import os
from copy import deepcopy
from datetime import datetime
from functools import reduce, partial
from operator import iadd
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from pymonad.either import Either
from torch import Tensor
from tqdm import tqdm

from fedcore.api.utils.data import DataLoaderHandler
from fedcore.data.data import CompressionInputData
from fedcore.losses.utils import _get_loss_metric
from fedcore.repository.constanst_repository import default_device, Hooks
from fedcore.architecture.abstraction.accessor import Accessor


def now_for_file():
    return datetime.now().strftime("%m-%d-%Y_%H-%M-%S")


class BaseNeuralModel:
    """Class responsible for NN model implementation.

    Attributes:
        self.num_features: int, the number of features.

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot_ind.tools.loader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('resnet_model').add_node('rf').build()
                input_data = init_input_data(train_data[0], train_data[1])
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)
                print(features)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        self.params = params or {}
        self.epochs = self.params.get("epochs", 1)
        self.batch_size = self.params.get("batch_size", 16)
        self.learning_rate = self.params.get("learning_rate", 0.001)
        self.custom_loss = self.params.get(
            "custom_loss", None
        )  # loss which evaluates model structure
        self.enforced_training_loss = self.params.get("enforced_training_loss", None)
        self.device = self.params.get('device', default_device())
        self._optimizer_gen = partial(torch.optim.Adam, lr=self.learning_rate)

        self.is_operation = self.params.get('is_operation', False) ###
        self.save_each = self.params.get('save_each', None)
        self.eval_each = self.params.get('eval_each', 5)
        self.checkpoint_folder = self.params.get('checkpoint_folder', None) ###
        self.batch_limit = self.params.get('batch_limit', None)
        self.calib_batch_limit = self.params.get('calib_batch_limit', None)
        self.batch_type = self.params.get('batch_type', None)
        self.name = self.params.get('name', '')

        self.label_encoder = None
        self.is_regression_task = False
        self.model = None
        self.target = None
        self.task_type = None

        # add hooks
        self._on_epoch_end = []
        self._on_epoch_start = []

    def __check_and_substitute_loss(self, train_data: InputData):
        if (
            train_data.supplementary_data.col_type_ids is not None
            and train_data.supplementary_data.col_type_ids.get("loss", None)
        ):
            criterion = train_data.supplementary_data.col_type_ids["loss"]
            try: 
                self.loss_fn = criterion()
            except:
                self.loss_fn = criterion
            print("Forcely substituted loss to", self.loss_fn)

    def __substitute_device_quant(self):
        if getattr(self.model, '_is_quantized', False):
            self.device = default_device('cpu')
            self.model.to(self.device)
            print('Quantized model inference supports CPU only')

    def fit(self, input_data: InputData, supplementary_data: dict = None, loader_type: Literal['train', 'calib'] = 'train'):
        custom_fit_process = supplementary_data is not None
        train_loader = getattr(input_data.features, f'{loader_type}_dataloader', 'train_dataloader')
        # (input_data.features.train_dataloader 
        #             if not finetune else 
        #                 input_data.features.calib_dataloader)
        val_loader = getattr(input_data.features, 'calib_dataloader', None)

        self.loss_fn = _get_loss_metric(input_data)
        self.__check_and_substitute_loss(input_data)
        if self.model is None:
            self.model = input_data.target
        self._init_hooks()
        self.optimised_model = self.model
        self.optimizer = self._optimizer_gen(self.model.parameters())
        self.model.to(self.device)

        fit_output = Either(
            value=supplementary_data, monoid=[self.custom_loss, custom_fit_process]
        ).either(
            left_function=lambda custom_loss: self._default_train(
                train_loader, self.model, custom_loss, val_loader=val_loader
            ),
            right_function=lambda sup_data: self._custom_train(
                train_loader, self.model, sup_data["callback"], val_loader=val_loader
            ),
        )
        self._clear_cache()
        return self.model
    
    @torch.no_grad()
    def _eval(self, model, val_dataloader, metrics=None, custom_loss=None):
        model.eval()
        loss_sum = 0
        total_iterations = 0
        val_dataloader = DataLoaderHandler.check_convert(dataloader=val_dataloader,
                                                       mode=self.batch_type,
                                                       max_batches=self.calib_batch_limit,
                                                       enumerate=False)
        for batch in tqdm(val_dataloader, desc='Batch #'):
            total_iterations += 1
            inputs, targets = batch
            output = self.model(inputs.to(self.device))
            if custom_loss:
                model_loss = {key: val(model) for key, val in custom_loss.items()}
                model_loss["metric_loss"] = self.loss_fn(
                    output, targets.to(self.device)
                )
                quality_loss = reduce(iadd, [loss for loss in model_loss.values()])
                loss_sum += model_loss["metric_loss"].item()
            else:
                quality_loss = self.loss_fn(output, targets.to(self.device))
                loss_sum += quality_loss.item()
                model_loss = quality_loss
        avg_loss = loss_sum / total_iterations
        return avg_loss

    def _train_loop(self, train_loader, model, val_loader=None, custom_loss: dict = None, need_eval=False):
        loss_sum = 0
        total_iterations = 0
        losses = None
        train_loader = DataLoaderHandler.check_convert(dataloader=train_loader,
                                                       mode=self.batch_type,
                                                       max_batches=self.batch_limit,
                                                       enumerate=False)
        for batch in tqdm(train_loader, desc='Batch #'):
            self.optimizer.zero_grad()
            total_iterations += 1
            inputs, targets = batch
            output = self.model(inputs.to(self.device))
            if custom_loss:
                model_loss = {key: val(model) for key, val in custom_loss.items()}
                model_loss["metric_loss"] = self.loss_fn(
                    output, targets.to(self.device)
                )
                quality_loss = reduce(iadd, [loss for loss in model_loss.values()])
                loss_sum += model_loss["metric_loss"].item()
            else:
                quality_loss = self.loss_fn(output, targets.to(self.device))
                loss_sum += quality_loss.item()
                model_loss = quality_loss
            quality_loss.backward()
            self.optimizer.step()
        avg_loss = loss_sum / total_iterations
        if custom_loss:
            losses = reduce(iadd, list(model_loss.items()))
            losses = [x.item() if not isinstance(x, str) else x for x in losses]
        return losses, avg_loss

    def _custom_train(self, train_loader,  model, callback: Callable, val_loader=None):
        # callback.callbacks.on_train_end()
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            model_loss, avg_loss = self._train_loop(train_loader, model)
            print("Epoch: {}, Average loss {}".format(epoch, avg_loss))
            if epoch % self.eval_each == 0 and val_loader is not None:
                print('Model Validation:' , self._eval(self.model, val_loader, ))
            if epoch > 3:
                # Freeze quantizer parameters
                self.model.apply(torch.quantization.disable_observer)
            if epoch > 2:
                # Freeze batch norm mean and variance estimates
                self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    def _default_train(self, train_loader, model, custom_loss: dict = None, val_loader=None):
        for epoch in range(1, self.epochs + 1):
            for hook in self._on_epoch_start:
                hook(epoch=epoch)
            self.model.train()
            model_loss, avg_loss = self._train_loop(train_loader, model, custom_loss)
            if model_loss is not None:
                print(
                    "Epoch: {}, Average loss {}, {}: {:.6f}, {}: {:.6f}, {}: {:.6f}".format(
                        epoch, avg_loss, *model_loss
                    )
                )
            else:
                print("Epoch: {}, Average loss {}".format(epoch, avg_loss))

            for hook in self._on_epoch_end:
                hook(epoch=epoch, val_loader=val_loader, custom_loss=custom_loss)         
    

    def predict(self, input_data: InputData, output_mode: str = "default"):
        """
        Method for feature generation for all series
        """
        self.__substitute_device_quant()
        return self._predict_model(input_data.features, output_mode)

    def predict_for_fit(self, input_data: InputData, output_mode: str = "default"):
        """
        Method for feature generation for all series
        """
        self.__substitute_device_quant()
        return self._predict_model(input_data.features, output_mode)

    def _predict_model(
        self, x_test: CompressionInputData, output_mode: str = "default"
    ):
        assert type(x_test) is CompressionInputData
        # print('### IS_QUANTIZED', getattr(self.model, '_is_quantized', False))
        model: torch.nn.Module = self.model or x_test.target
        model.eval()
        prediction = []
        dataloader = DataLoaderHandler.check_convert(x_test.calib_dataloader,
                                                     mode=self.batch_type,
                                                     max_batches=self.calib_batch_limit)
        for batch in tqdm(dataloader): ###TODO why calib_dataloader???
            inputs, targets = batch
            inputs = inputs.to(self.device)
            prediction.append(model(inputs))
        # print('### PREDICTION', prediction)
        return self._convert_predict(torch.concat(prediction), output_mode)

    def _convert_predict(self, pred: Tensor, output_mode: str = "labels"):
        have_encoder = all([self.label_encoder is not None, output_mode == "labels"])
        output_is_clf_labels = all(
            [not self.is_regression_task, output_mode == "labels"]
        )

        pred = (
            pred.cpu().detach().numpy()
            if self.is_regression_task
            else F.softmax(pred, dim=1)
        )
        y_pred = (
            torch.argmax(pred, dim=1).cpu().detach().numpy()
            if output_is_clf_labels
            else pred
        )
        y_pred = (
            self.label_encoder.inverse_transform(y_pred) if have_encoder else y_pred
        )

        predict = OutputData(
            idx=np.arange(len(y_pred)),
            task=self.task_type,
            predict=y_pred,
            target=self.target,
            data_type=DataTypesEnum.table,
        )
        return predict

    def _clear_cache(self):
        with torch.no_grad():
            torch.cuda.empty_cache()

    @staticmethod
    def get_validation_frequency(epoch, lr):
        if epoch < 10:
            return 1  # Validate frequently in early epochs
        elif lr < 0.01:
            return 5  # Validate less frequently after learning rate decay
        else:
            return 2  # Default validation frequency
        
    @property
    def is_quantised(self):
        return getattr(self, '_is_quantised', False)
    
    def _init_hooks(self):
        for hook_elem in Hooks:
            hook = hook_elem.value
            if not self.params.get(hook._SUMMON_KEY, None):
                continue
            hook = hook(self.params, self.model)
            if hook._hook_place == 'post':
                self._on_epoch_end.append(hook)
            else:
                self._on_epoch_start.append(hook)
