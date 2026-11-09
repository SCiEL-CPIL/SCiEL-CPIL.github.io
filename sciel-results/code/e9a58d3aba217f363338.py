import time
import numpy as np
from collections import defaultdict

def _row_masks(matrix, n_cols):
    """Convert matrix rows to integer bitmasks."""
    masks = []
    for i in range(matrix.shape[0]):
        val = 0
        row = matrix[i]
        for j in range(n_cols):
            if row[j] != 0:
                val |= (1 << j)
        masks.append(val)
    return masks

def _popcount(x):
    """Count set bits in integer."""
    return bin(x).count('1')

def _xor_tree(masks, next_wire, gates, wire_values):
    """Build XOR trees for each mask, returning list of output wire IDs."""
    outputs = []
    const_zero = len(wire_values) - 2  # n
    for mask in masks:
        if mask == 0:
            outputs.append(const_zero)
            continue
        if _popcount(mask) == 1:
            # Single bit: find which input
            j = mask.bit_length() - 1
            outputs.append(j)
            continue
        # Build XOR tree
        bits = []
        m = mask
        while m:
            lsb = m & (-m)
            j = lsb.bit_length() - 1
            bits.append(j)
            m ^= lsb
        result = bits[0]
        for b in bits[1:]:
            v1 = wire_values[result]
            v2 = wire_values[b]
            res = v1 ^ v2
            # Check if already exists
            found = None
            for wid, val in wire_values.items():
                if val == res:
                    found = wid
                    break
            if found is None:
                found = next_wire[0]
                next_wire[0] += 1
                gates.append([result, b])
                wire_values[found] = res
            result = found
        outputs.append(result)
    return outputs

def _build_greedy_cse(masks, n):
    """
    Greedy common-subexpression elimination.
    Repeatedly find the pair of existing wires whose XOR appears most frequently
    among remaining targets, add it as a gate, and simplify targets.
    """
    const_zero = n
    # State: list of target masks (what each output still needs)
    targets = list(masks)
    # Available wires: wire_id -> bitmask
    wire_vals = {}
    for j in range(n):
        wire_vals[j] = (1 << j)
    wire_vals[const_zero] = 0
    next_w = n + 1
    gates = []
    
    # Map from bitmask to wire_id for quick lookup
    val_to_wire = {}
    for wid, v in wire_vals.items():
        val_to_wire[v] = wid
    
    # Count how many targets each bit appears in
    # We'll iteratively:
    # 1. For each pair of existing wires, compute XOR value
    # 2. Count how many targets would be simplified if we add that XOR
    # 3. Add the best one
    
    # For efficiency, track which targets are "done" (single bit or zero)
    # and which still need work
    
    n_targets = len(targets)
    
    while True:
        # Check if all targets are trivial (0 or single bit)
        all_done = True
        for t in targets:
            if t != 0 and _popcount(t) > 1:
                all_done = False
                break
        if all_done:
            break
        
        # Find best pair to add
        best_gain = 0
        best_pair = None
        best_val = None
        
        wire_ids = list(wire_vals.keys())
        # Only consider pairs where XOR is non-trivial and not already a wire
        for idx_a in range(len(wire_ids)):
            wa = wire_ids[idx_a]
            va = wire_vals[wa]
            for idx_b in range(idx_a + 1, len(wire_ids)):
                wb = wire_ids[idx_b]
                vb = wire_vals[wb]
                vx = va ^ vb
                if vx == 0:
                    continue
                if vx in val_to_wire:
                    continue
                # Count how many targets contain both va and vb as subsets
                # Actually, a target t can use wire vx if vx is a subset of t
                # i.e., (t & vx) == vx
                # The gain is the number of targets where vx is a proper subset
                # and using it reduces the popcount
                gain = 0
                for t in targets:
                    if t == 0:
                        continue
                    if (t & vx) == vx and vx != t:
                        # vx is a proper subset of t, so we can use it
                        gain += 1
                    elif t == vx:
                        # t equals vx, so we can directly use it
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (wa, wb)
                    best_val = vx
        
        if best_gain <= 1:
            # No significant sharing benefit; just build direct trees
            break
        
        # Add the best gate
        wa, wb = best_pair
        gates.append([wa, wb])
        wire_vals[best_val] = next_w
        val_to_wire[best_val] = next_w
        next_w += 1
    
    # Now build final outputs using available wires
    outputs = []
    for t in targets:
        if t == 0:
            outputs.append(const_zero)
            continue
        if _popcount(t) == 1:
            j = t.bit_length() - 1
            outputs.append(j)
            continue
        # Need to decompose t using available wires
        # Greedy: repeatedly pick an available wire that is a subset of t
        remaining = t
        parts = []
        while remaining != 0:
            if _popcount(remaining) == 1:
                j = remaining.bit_length() - 1
                parts.append(j)
                break
            # Find best wire that is a subset of remaining
            best_w = None
            best_pc = 0
            for wid, v in wire_vals.items():
                if v == 0:
                    continue
                if (remaining & v) == v:
                    pc = _popcount(v)
                    if pc > best_pc:
                        best_pc = pc
                        best_w = wid
            if best_w is None:
                # Fallback: use single bits
                m = remaining
                while m:
                    lsb = m & (-m)
                    j = lsb.bit_length() - 1
                    parts.append(j)
                    m ^= lsb
                break
            else:
                parts.append(best_w)
                remaining ^= wire_vals[best_w]
        
        # Combine parts
        if len(parts) == 1:
            outputs.append(parts[0])
        else:
            result = parts[0]
            for p in parts[1:]:
                v1 = wire_vals[result]
                v2 = wire_vals[p]
                res = v1 ^ v2
                found = None
                for wid, val in wire_vals.items():
                    if val == res:
                        found = wid
                        break
                if found is None:
                    found = next_w
                    next_w += 1
                    gates.append([result, p])
                    wire_vals[found] = res
                    val_to_wire[res] = found
                result = found
            outputs.append(result)
    
    return gates, outputs

def _build_direct(masks, n):
    """Direct XOR tree per row (baseline)."""
    const_zero = n
    wire_vals = {}
    for j in range(n):
        wire_vals[j] = (1 << j)
    wire_vals[const_zero] = 0
    next_w = n + 1
    gates = []
    outputs = []
    
    for mask in masks:
        if mask == 0:
            outputs.append(const_zero)
            continue
        if _popcount(mask) == 1:
            j = mask.bit_length() - 1
            outputs.append(j)
            continue
        bits = []
        m = mask
        while m:
            lsb = m & (-m)
            j = lsb.bit_length() - 1
            bits.append(j)
            m ^= lsb
        result = bits[0]
        for b in bits[1:]:
            v1 = wire_vals[result]
            v2 = wire_vals[b]
            res = v1 ^ v2
            found = None
            for wid, val in wire_vals.items():
                if val == res:
                    found = wid
                    break
            if found is None:
                found = next_w
                next_w += 1
                gates.append([result, b])
                wire_vals[found] = res
            result = found
        outputs.append(result)
    
    return gates, outputs

def _local_search(gates, outputs, masks, n, deadline):
    """
    Local search: try to replace pairs of gates with shorter sequences.
    """
    if time.time() > deadline:
        return gates, outputs
    
    const_zero = n
    # Build wire values
    wire_vals = {}
    for j in range(n):
        wire_vals[j] = (1 << j)
    wire_vals[const_zero] = 0
    gw = n + 1
    for g in gates:
        v1 = wire_vals[g[0]]
        v2 = wire_vals[g[1]]
        wire_vals[gw] = v1 ^ v2
        gw += 1
    
    # Try to find redundant gates or shorter paths
    # For each output, check if we can reach its target mask with fewer gates
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_gates = list(gates)
        new_outputs = list(outputs)
        
        # Try removing each gate and checking if circuit still works
        for gi in range(len(new_gates) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Check if gate gi is used by any output or other gate
            used = False
            for out in new_outputs:
                if out == n + 1 + gi:
                    used = True
                    break
            if not used:
                for g in new_gates:
                    if g[0] == n + 1 + gi or g[1] == n + 1 + gi:
                        used = True
                        break
            if not used:
                new_gates.pop(gi)
                improved = True
                break
        gates = new_gates
        outputs = new_outputs
    
    return gates, outputs

def synthesize(matrix, depth_limit):
    """
    Synthesize an exact XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D binary array (n_rows x n_cols) over GF(2).
        depth_limit: maximum allowed depth (<=16).
    
    Returns:
        dict with 'gates' (list of [wire, wire]) and 'outputs' (list of wires).
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Handle edge cases
    if matrix is None:
        return {'gates': [], 'outputs': []}
    
    matrix = np.array(matrix, dtype=np.uint8)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    
    n_rows, n_cols = matrix.shape
    n = n_cols
    const_zero = n
    
    if n_rows == 0:
        return {'gates': [], 'outputs': []}
    
    masks = _row_masks(matrix, n)
    
    # Try greedy CSE first
    try:
        gates_cse, outputs_cse = _build_greedy_cse(masks, n)
    except Exception:
        gates_cse, outputs_cse = _build_direct(masks, n)
    
    # Also build direct for comparison
    gates_dir, outputs_dir = _build_direct(masks, n)
    
    # Pick the one with fewer gates
    if len(gates_cse) <= len(gates_dir):
        gates, outputs = gates_cse, outputs_cse
    else:
        gates, outputs = gates_dir, outputs_dir
    
    # Local search to remove redundant gates
    gates, outputs = _local_search(gates, outputs, masks, n, deadline)
    
    # Verify correctness
    const_zero = n
    wire_vals = {}
    for j in range(n):
        wire_vals[j] = (1 << j)
    wire_vals[const_zero] = 0
    gw = n + 1
    for g in gates:
        v1 = wire_vals[g[0]]
        v2 = wire_vals[g[1]]
        wire_vals[gw] = v1 ^ v2
        gw += 1
    
    # Check outputs
    valid = True
    for i, out in enumerate(outputs):
        if out not in wire_vals:
            valid = False
            break
        if wire_vals[out] != masks[i]:
            valid = False
            break
    
    if not valid:
        # Fallback to direct
        gates, outputs = _build_direct(masks, n)
    
    return {'gates': gates, 'outputs': outputs}
