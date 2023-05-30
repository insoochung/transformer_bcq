import argparse
import json

from train_utils import train, SCRIPT_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="BCQ TF example")
    parser.add_argument("--hparams", default=SCRIPT_DIR / "hparams/small.json")
    args = parser.parse_args()
    with open(args.hparams, "r", encoding="utf-8") as f:
        hparams_dict = json.load(f)
    train(hparams_dict)
