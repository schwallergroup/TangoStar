#!/bin/bash

# You can run evaluation of the algorithms like so.
# evaluate.sh uspto_190 retro 500 25 0.7.
# evaluate.sh  {test_set} {algorithm} {expansion_budget} {tango_weight} {tanimoto_weight}
case "$1" in
"pistachio_reachable")
DATA_PATH="../data/pistachio_reachable_targets.txt"
;;
"pistachio_hard")
DATA_PATH="../data/pistachio_hard_targets.txt"
;;
"uspto_190")
DATA_PATH="../data/uspto_190_targets.txt"
;;
*)
echo "Invalid dataset: $DATASET"
exit 1
;;
esac
# Set environment variables
export RETRO_MODEL="../data/model_retro.pt"
export RETRO_TEMPLATES="../data/idx2template_retro.json"
export BB_MOL2IDX="../data/canon_building_block_mol2idx_no_isotope.json"
export FWD_MODEL="../data/model_fwd.pt"
export FWD_TEMPLATES="../data/idx2template_fwd.json"
export BB_MODEL="../data/model_bb.pt"
export BB_TENSOR="../data/building_block_fps.npz"
export SD_MODEL="../data/syn_dist.pt"
export VALUE_MODEL="../data/retro_value.pt"
export DEVICE=0

python3 evaluate.py \
--test_set "$1" \
--strategy "$2" \
--test_path "$DATA_PATH" \
--retro_model "$RETRO_MODEL" \
--retro_templates "$RETRO_TEMPLATES" \
--bb_mol2idx "$BB_MOL2IDX" \
--fwd_model "$FWD_MODEL" \
--fwd_templates "$FWD_TEMPLATES" \
--bb_model "$BB_MODEL" \
--bb_tensor "$BB_TENSOR" \
--bb_idx2mol "$BB_IDX2MOL" \
--sd_model "$SD_MODEL" \
--value_model "$VALUE_MODEL" \
--device "$DEVICE" \
--iteration_limit $3 \
--tango_weight $4 \
--tanimoto_weight $5 \
