"""Train State-Aware Interaction MIL with the paper protocol."""

import argparse

from experiments.common import add_common_args, prepare_run
from stateaware_mil.training import train_main_method


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg, fm, fm_name, df, store, input_dim, output_root, settings, device = prepare_run(args)
    summary = train_main_method(
        "StateAware", df, store, input_dim, output_root, "UNI2" if fm == "uni2" else "CONCH",
        cfg["biomarker_a"], cfg["biomarker_b"], settings, device, args.force,
    )
    print(f"\n{cfg['display_name']} | {fm_name} | State-Aware")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
