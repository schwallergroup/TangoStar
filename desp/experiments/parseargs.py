import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Python script with different options."
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="f2e",
        choices=["f2e", "f2f", "retro", "retro_sd", "retro_tango", "random", "bfs", "bi-bfs" , "f2f_tango", "f2e_tango"],
        help="Strategy for predict_one() function",
    )
    parser.add_argument(
        "--test_set",
        type=str,
        required=True,
        choices=[
            "pistachio_reachable",
            "pistachio_hard",
            "uspto_190",
            "who",
        ],
        help="Test set to use",
    )
    parser.add_argument("--test_path", type=str, help="Path to the test data file")
    parser.add_argument("--retro_model", type=str, help="Path to the retro model")
    parser.add_argument(
        "--retro_templates", type=str, help="Path to the retro templates"
    )
    parser.add_argument(
        "--bb_mol2idx", type=str, help="Path to the building block mol2idx file"
    )
    parser.add_argument("--fwd_model", type=str, help="Path to the forward model")
    parser.add_argument(
        "--fwd_templates", type=str, help="Path to the forward templates"
    )
    parser.add_argument("--bb_model", type=str, help="Path to the building block model")
    parser.add_argument(
        "--bb_tensor", type=str, help="Path to the building block tensor file"
    )
    parser.add_argument(
        "--bb_idx2mol", type=str, help="Path to the building block idx2mol file"
    )
    parser.add_argument(
        "--sd_model", type=str, help="Path to the synthetic distance model"
    )
    parser.add_argument(
        "--value_model", type=str, help="Path to the value model"
        )
    parser.add_argument(
        "--device", type=str, help="Device to load index and models to"
        )
    parser.add_argument(
        "--tango_weight", type=float, help="tango weight, ranges between 20-30 work well"
        )
    parser.add_argument(
        "--iteration_limit", type=int, help="number of expansions"
        )
    parser.add_argument(
        "--tanimoto_weight", type=float, help="Weight of Tanimoto Similarity in the Tango reward, bounded between 0 and 1"
        )
    parser.add_argument(
        "--mcs_weight", type=float, help="Weight of maximum common substructure in the Tango Reward"
        )
    parser.add_argument(
        "--task_id" , type=int, help="Task id"
    )

    args = parser.parse_args()
    return args