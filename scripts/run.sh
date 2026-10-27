#!/bin/bash
# Runs the patched strata-q36 engine as a server on stdin/stdout (Strata's own front-ends talk to it the same way).
# Usage: scripts/run.sh <strata build dir> <model .gguf> <profile> [extra engine args]
# Long-prompt mode (agents): LONG=1 scripts/run.sh ... (streamed layers, borrowed staging, RAM tier 16 GB).
# The first run writes <model>.q36pack (about 33 GB) next to the model.
set -euo pipefail
B=${1:?strata build dir}; M=${2:?model}; P=${3:?profile}; shift 3
export STRATA_Q8_PACKED=0                 # the packed Q8 decode copies would take the experts' VRAM
export STRATA_Q36_EMBD_HOST=1             # token embeddings in host memory (zero-copy)
export STRATA_Q36_VRAM_RESERVE_MB=256     # VRAM left free besides the engine's own buffers
export STRATA_Q36_PROFILE="$P"            # most used experts -> VRAM slots, next -> RAM tier
if [ "${LONG:-0}" = 1 ]; then
  export STRATA_Q36_PACK_STREAM=1 STRATA_Q36_BORROW_STAGING=1 STRATA_Q36_GATHER_THREADS=4
  export STRATA_Q36_RAM_GB=${STRATA_Q36_RAM_GB:-16}
else
  export STRATA_Q36_RAM_GB=${STRATA_Q36_RAM_GB:-18}
fi
# on a 32 GB machine also run once: sudo sysctl vm.swappiness=1
exec "$B/strata-q36" --model "$M" --max-context "${CTX:-12288}" --mtp 0 --pool-workers 3 --serve "$@"
