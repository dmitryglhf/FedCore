from fedcore.api.config_factory import ConfigFactory
from torchvision.models import resnet18, ResNet18_Weights
from fedcore.api.api_configs import (
    APIConfigTemplate, AutoMLConfigTemplate, FedotConfigTemplate, 
    LearningConfigTemplate, ModelArchitectureConfigTemplate, 
    NeuralModelConfigTemplate, PruningTemplate
)
from fedcore.architecture.dataset.api_loader import ApiLoader
from fedcore.tools.example_utils import get_scenario_for_api
from fedcore.api.main import FedCore


METRIC_TO_OPTIMISE = ['accuracy', 'latency']
LOSS = 'cross_entropy'
PROBLEM = 'classification'
PEFT_PROBLEM = 'pruning'
INITIAL_ASSUMPTION = resnet18(ResNet18_Weights)
INITIAL_MODEL = 'ResNet18'
PRETRAIN_SCENARIO = 'from_checkpoint'
SCRATCH = 'from_scratch'

POP_SIZE = 2

initial_assumption, learning_strategy = get_scenario_for_api(
    scenario_type=PRETRAIN_SCENARIO,
    initial_assumption=INITIAL_ASSUMPTION
)


def get_api_config():
    model_config = ModelArchitectureConfigTemplate(
        input_dim=None,
        output_dim=None,
        depth=6
    )

    pretrain_config = NeuralModelConfigTemplate(
        epochs=200,
        log_each=10,
        eval_each=15,
        save_each=50,
        criterion=LOSS,
        model_architecture=model_config,
        custom_learning_params=dict(
            use_early_stopping={
                'patience': 30,
                'maximise_task': False,
                'delta': 0.01
            }
        )
    )

    fedot_config = FedotConfigTemplate(
        problem=PROBLEM,
        metric=METRIC_TO_OPTIMISE,
        pop_size=POP_SIZE,
        timeout=5,
        initial_assumption=INITIAL_ASSUMPTION
    )

    automl_config = AutoMLConfigTemplate(fedot_config=fedot_config)

    finetune_config = NeuralModelConfigTemplate(
        epochs=3,
        log_each=3,
        eval_each=3,
        criterion=LOSS,
    )

    peft_config = PruningTemplate(
        importance="magnitude",
        pruning_ratio=0.8,
        finetune_params=finetune_config
    )

    learning_config = LearningConfigTemplate(
        criterion=LOSS,
        learning_strategy=learning_strategy,
        learning_strategy_params=pretrain_config,
        peft_strategy=PEFT_PROBLEM,
        peft_strategy_params=peft_config
    )

    api_template = APIConfigTemplate(
        automl_config=automl_config,
        learning_config=learning_config
    )

    APIConfig = ConfigFactory.from_template(api_template)
    api_config = APIConfig()
    return api_config


def get_input_data():
    al = ApiLoader('CIFAR10', {'split_ratio': [0.6, 0.4]})
    input_data = al._convert_to_fedcore(al._init_pretrain_dataset(al.source))
    return input_data


if __name__ == "__main__":
    api_config = get_api_config()
    input_data = get_input_data()
    
    fedcore_compressor = FedCore(api_config)
    fedcore_compressor.fit(input_data)
