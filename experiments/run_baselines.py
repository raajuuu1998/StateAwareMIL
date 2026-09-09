"""Train the comparison methods reported in Table 2."""

import argparse

from experiments.common import add_common_args, prepare_run
from stateaware_mil.training import train_main_method

METHODS = ["DirectJoint", "IndependentPair", "NaiveMTL", "PostHocLR"]


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--method",
        default="all",
        choices=["all", "directjoint", "independentpair", "naivemtl", "posthoc_lr"],
    )
    args = parser.parse_args()
    cfg, fm, fm_name, df, store, input_dim, output_root, settings, device = prepare_run(args)
    fm_folder = "UNI2" if fm == "uni2" else "CONCH"
    selected = METHODS if args.method == "all" else [{
        "directjoint": "DirectJoint",
        "independentpair": "IndependentPair",
        "naivemtl": "NaiveMTL",
        "posthoc_lr": "PostHocLR",
    }[args.method]]

    # PostHoc-LR depends on the IndependentPair checkpoints. If all baselines
    # are requested, the order below guarantees that dependency is available.
    for method in selected:
        summary = train_main_method(
            method, df, store, input_dim, output_root, fm_folder,
            cfg["biomarker_a"], cfg["biomarker_b"], settings, device, args.force,
        )
        print(f"\n{cfg['display_name']} | {fm_name} | {method}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
