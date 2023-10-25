from models import *
import yaml
import argparse
from dataloader import CustomDataset


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config('configs/config.yml')

parser = argparse.ArgumentParser()
parser.add_argument("--debug", "-d", action="store_true")

def model_init():
    pass

def run_experiment():
    pass

def wandb_setup(dataset):
    pass
    # run_experiment(group_name=date, dataset_name=dataset)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        wandb_mode = "disabled"
    else:
        wandb_mode = "online"
    
    if args.tvsum:
        wandb_setup(dataset="tvsum")
    elif args.summe:
        wandb_setup(dataset="summe")

    if args.count:
        wandb_config = config["wandb_setup"]
        model = model_init(config["model_name"], wandb_config["input_dim"], wandb_config["model_dim"], wandb_config["num_heads"],
                            wandb_config["num_layers"], wandb_config["output_dim"], wandb_config["max_seq_length"], wandb_config["hidden_dropout"], wandb_config["atten_dropout"])
        # print_trainable_parameters(model)