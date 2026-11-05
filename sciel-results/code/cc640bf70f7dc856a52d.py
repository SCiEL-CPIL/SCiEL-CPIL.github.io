"""XOR-circuit synthesis with global CSE and frequency-based precomputation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import Counter

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = 85.0

    A = np.array(matrix, dtype=np.int8)
    n, m = A.shape

    # Compute output bitsets
    out_vals = []
    for j in range(m):
        v = 0
        for i in range(n):
            if A[i, j]:
                v |= (1 << i)
        out_vals.append(v)

    # Count 2-bit pairs that co-occur in outputs
    pair_counts = Counter()
    for v in out_vals:
        bits = [i for i in range(n) if (v >> i) & 1]
        for idx in range(len(bits)):
            for jdx in range(idx + 1, len(bits)):
                pair_counts[(bits[idx], bits[jdx])] += 1

    # Count intermediate values from sequential decomposition
    intermediate_counts = Counter()
    for v in out_vals:
        bits = sorted([i for i in range(n) if (v >> i) & 1])
        cur = 0
        for i in bits:
            cur ^= (1 << i)
            if cur != 0:
                intermediate_counts[cur] += 1

    # Build circuit with CSE
    wire_val = {}
    val_to_wire = {}
    for i in range(n):
        wire_val[i] = (1 << i)
        val_to_wire[1 << i] = i
    wire_val[n] = 0
    val_to_wire[0] = n

    gates = []
    next_id = n + 1

    def create_gate(a, b):
        nonlocal next_id
        v = wire_val[a] ^ wire_val[b]
        new_id = next_id
        next_id += 1
        gates.append((a, b))
        wire_val[new_id] = v
        val_to_wire[v] = new_id
        return new_id

    def ensure_wire(v):
        nonlocal next_id
        if v in val_to_wire:
            return val_to_wire[v]
        # Find existing pair that XORs to v
        for a, va in list(wire_val.items()):
            b_val = v ^ va
            if b_val in val_to_wire:
                return create_gate(a, val_to_wire[b_val])
        # Build from inputs sequentially
        bits = [i for i in range(n) if (v >> i) & 1]
        if not bits:
            return n
        wire = n
        for i in bits:
            new_v = wire ^ (1 << i)
            if new_v not in val_to_wire:
                wire = create_gate(wire, i)
            else:
                wire = val_to_wire[new_v]
        return wire

    # Pre-create frequent 2-bit XORs
    for pair, count in pair_counts.most_common():
        if time.time() - start_time > DEADLINE - 5:
            break
        if count >= 2:
            i, j = pair
            ensure_wire((1 << i) | (1 << j))

    # Pre-create frequent intermediates
    for v, count in intermediate_counts.most_common():
        if time.time() - start_time > DEADLINE - 3:
            break
        if count >= 2:
            ensure_wire(v)

    # Final: get output wires
    outputs = []
    for v in out_vals:
        outputs.append(ensure_wire(v))

    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
