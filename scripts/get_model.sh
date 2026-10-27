#!/bin/bash
# Downloads unsloth's Ornith-1.5-122B UD-IQ2_XXS (42.0 GB), used as is. Usage: scripts/get_model.sh <dir>
set -euo pipefail
D=${1:?dir}; F=Ornith-1.5-122B-UD-IQ2_XXS.gguf
[ -s "$D/$F" ] || curl -fL -C - -o "$D/$F" "https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF/resolve/main/$F"
ls -l "$D/$F"   # expected size: 41967395072 bytes
