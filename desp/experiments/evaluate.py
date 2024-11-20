import json
import networkx as nx
import os
import pickle
import sys
import time
from networkx.algorithms.dag import dag_longest_path
from parseargs import parse_args
from rdkit.Chem import Descriptors, MolFromSmiles
from tqdm import tqdm
import ast

sys.path.append("../..")
from desp.search.desp_search import DespSearch
from desp.inference.retro_predictor import RetroPredictor
from desp.inference.syn_dist_predictor import SynDistPredictor
from desp.inference.retro_value import ValuePredictor
from desp.inference.forward_predictor import ForwardPredictor
from desp.inference.tango_value import TangoValue


def zero(smiles_1, smiles_2):
    return 0


def predict_one(target, starting):

    searcher = DespSearch(
        target,
        starting,
        retro_predictor,
        fwd_predictor,
        building_blocks,
        strategy=args.strategy,
        heuristic_fn=value_predictor.predict,
        distance_fn=distance_fn,
        iteration_limit=args.iteration_limit,
        top_m=25,
        top_k=1,
        max_depth_top=21,
        max_depth_bot=11,
        stop_on_first_solution=True,
        must_use_sm=False,
        retro_only=False if args.strategy in ["f2e", "f2f", "bi-bfs"] else True,
    )
    print(f"Starting search towards {target} from {starting} using {args.iteration_limit} expansions")
    result = searcher.run_search()
    print(f"Result for {target} from {starting}: {result}")
    return target, starting, result, searcher.search_graph


if __name__ == "__main__":
    args = parse_args()

    # Load retro predictor
    retro_predictor = RetroPredictor(
        model_path=args.retro_model, templates_path=args.retro_templates
    )

    # Load building blocks
    with open(args.bb_mol2idx, "r") as f:
        building_blocks = json.load(f)

    if args.strategy in ["f2e", "f2f", "bi-bfs", "f2f_tango", "f2e_tango"]:
        # Load fwd predictor
        fwd_predictor = ForwardPredictor(
            forward_model_path=args.fwd_model,
            templates_path=args.fwd_templates,
            bb_model_path=args.bb_model,
            bb_tensor_path=args.bb_tensor,
            bb_mol2idx=building_blocks,
            device=args.device,
        )
    else:
        fwd_predictor = None
    print(f"running for iteration_limit {args.iteration_limit} on {args.test_path}")
    # Load synthetic distance and value models
    device = args.device if args.strategy == "f2f" else "cpu"
    sd_predictor = SynDistPredictor(args.sd_model, device)
    value_predictor = ValuePredictor(args.value_model)
    tango_value = TangoValue(args.tango_weight)

    if args.strategy in ["retro_tango", "f2f_tango" ,"f2e_tango"]:
        tango_value.weight = args.tango_weight
        tango_value.mcs_weight = args.mcs_weight
        tango_value.tanimoto_weight = args.tanimoto_weight
    # Load test set
    targets = []
    with open(args.test_path, "r") as f:
        if "who" in args.test_path:
            line = ast.literal_eval(f.readline())
            targ = line[0][0]
            sm = line[1]
            if targ in building_blocks:
                del building_blocks[targ]
            # sm = [s for s in sm if len(s) > 10]
            building_blocks = {s : idx for idx, s in enumerate(sm)}
            targets.append((targ, sm))
            


    if args.strategy == "f2f":
        distance_fn = sd_predictor.predict_batch
    elif args.strategy in ["f2e", "retro_sd"]:
        distance_fn = sd_predictor.predict
    else:
        distance_fn = zero

    results = []
    graphs = []

    # Construct the directory path
    directory = os.path.join("/your/dir/", args.test_set)
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"tango")

    strat = args.strategy
    if strat == "retro_tango":
        args.strategy = "retro_sd"
    if strat == "f2e_tango":
        args.strategy = "f2e"
    if strat == "f2f_tango":
        args.strategy = "f2f"

    with open(file_path + ".txt", "a") as f:
        start_time = time.time()
        for target, starting in tqdm(targets):
            if strat in ["retro_tango", "f2e_tango"]:
                distance_fn = tango_value.predict
            if strat in ["f2f_tango"]:
                distance_fn = tango_value.predict_batch

            starting_search = time.time()
            target, starting, result, graph = predict_one(target, starting)
            search_time = time.time() - starting_search
            results.append((target, starting, result, search_time))
            graphs.append(graph)
            f.write(f"('{target}', {result}, {search_time})\n")
            print(f"('{target}', {result}, {search_time})\n")
            f.flush()

    # Save graphs to pickle file
    with open(file_path+ "tango" + ".pkl", "wb") as f2:
        pickle.dump(graphs, f2)
