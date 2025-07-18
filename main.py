import yaml
from train import MultimodalTrainer


def main():
    config_path = "config.yaml"
    trainer = MultimodalTrainer(config_path)
    trainer.train()


if __name__ == "__main__":
    main()

