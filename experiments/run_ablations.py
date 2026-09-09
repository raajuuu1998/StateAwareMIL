"""Run the component ablations reported in Table 5."""

import argparse

from experiments.common import add_common_args, prepare_run
from stateaware_mil.training import train_ablation

VARIANTS = ["full", "no_interaction", "no_auxiliary", "binary_joint"]


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--variant", default="all", choices=["all", *VARIANTS])
    args = parser.parse_args()
    cfg, fm, fm_name, df, store, input_dim, output_root, settings, device = prepare_run(args)
    fm_folder = "UNI2" if fm == "uni2" else "CONCH"
    variants = VARIANTS if args.variant == "all" else [args.variant]
    for variant in variants:
        summary = train_ablation(
            variant, df, store, input_dim, output_root, fm_folder,
            cfg["biomarker_a"], cfg["biomarker_b"], settings, device, args.force,
        )
        print(f"\n{cfg['display_name']} | {fm_name} | {variant}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
