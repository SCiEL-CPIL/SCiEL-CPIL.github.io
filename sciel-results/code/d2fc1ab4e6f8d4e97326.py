import time
import numpy as np
from collections import defaultdict
import random

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

def _verify_circuit(gates, outputs, masks, n):
    """Verify circuit correctness."""
    const_zero = n
    wire_vals = {}
    for j in range(n):
        wire_vals[j] = (1 << j)
    wire_vals[const_zero] = 0
    gw = n + 1
    for g in gates:
        if g[0] not in wire_vals or g[1] not in wire_vals:
            return False
        wire_vals[gw] = wire_vals[g[0]] ^ wire_vals[g[1]]
        gw += 1
    for i, out in enumerate(outputs):
        if out not in wire_vals:
            return False
        if wire_vals[out] != masks[i]:
            return False
    return True

def _build_greedy_cse(masks, n, deadline):
    """
    Greedy common-subexpression elimination with time budget.
    """
    const_zero = n
    targets = list(masks)
    wire_vals = {}
    for j in range(n):
        wire_vals[j] = (1 << j)
    wire_vals[const_zero] = 0
    next_w = n + 1
    gates = []
    
    val_to_wire = {}
    for wid, v in wire_vals.items():
        val_to_wire[v] = wid
    
    # Sort targets by popcount descending for better sharing
    target_order = sorted(range(len(targets)), key=lambda i: _popcount(targets[i]), reverse=True)
    
    while time.time() < deadline:
        all_done = True
        for t in targets:
            if t != 0 and _popcount(t) > 1:
                all_done = False
                break
        if all_done:
            break
        
        best_gain = 0
        best_pair = None
        best_val = None
        
        wire_ids = list(wire_vals.keys())
        # Limit search for efficiency
        if len(wire_ids) > 60:
            # Only consider most recently added + inputs
            wire_ids = list(wire_vals.keys())[:n + 30] + list(wire_vals.keys())[-30:]
            # Deduplicate
            seen = set()
            unique_ids = []
            for wid in wire_ids:
                if wid not in seen:
                    seen.add(wid)
                    unique_ids.append(wid)
            wire_ids = unique_ids
        
        for idx_a in range(len(wire_ids)):
            wa = wire_ids[idx_a]
            va = wire_vals[wa]
            if va == 0:
                continue
            for idx_b in range(idx_a + 1, len(wire_ids)):
                wb = wire_ids[idx_b]
                vb = wire_vals[wb]
                if vb == 0:
                    continue
                vx = va ^ vb
                if vx == 0 or vx in val_to_wire:
                    continue
                gain = 0
                for t in targets:
                    if t == 0:
                        continue
                    if (t & vx) == vx:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (wa, wb)
                    best_val = vx
        
        if best_gain <= 1:
            break
        
        wa, wb = best_pair
        gates.append([wa, wb])
        wire_vals[best_val] = next_w
        val_to_wire[best_val] = next_w
        next_w += 1
    
    # Build final outputs
    outputs = []
    for t in targets:
        if t == 0:
            outputs.append(const_zero)
            continue
        if _popcount(t) == 1:
            j = t.bit_length() - 1
            outputs.append(j)
            continue
        remaining = t
        parts = []
        while remaining != 0:
            if _popcount(remaining) == 1:
                j = remaining.bit_length() - 1
                parts.append(j)
                break
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
        
        if len(parts) == 1:
            outputs.append(parts[0])
        else:
            result = parts[0]
            for p in parts[1:]:
                v1 = wire_vals[result]
                v2 = wire_vals[p]
                res = v1 ^ v2
                found = val_to_wire.get(res)
                if found is None:
                    found = next_w
                    next_w += 1
                    gates.append([result, p])
                    wire_vals[found] = res
                    val_to_wire[res] = found
                result = found
            outputs.append(result)
    
    return gates, outputs

def _build_basis_greedy(masks, n, deadline):
    """
    Basis-aware greedy: maintain a basis of wires, extend it greedily
    to minimize total XOR distance to targets.
    """
    const_zero = n
    wire_vals = {}
    for j in range(n):
        wire_vals[j] = (1 << j)
    wire_vals[const_zero] = 0
    val_to_wire = {0: const_zero}
    for j in range(n):
        val_to_wire[1 << j] = j
    next_w = n + 1
    gates = []
    
    # Greedy: repeatedly find the pair (a,b) in current wires whose XOR
    # minimizes the sum of popcount(t ^ best_available_subset) over all targets
    # Simplified: find pair whose XOR reduces total target complexity most
    
    remaining_targets = list(masks)
    
    while time.time() < deadline:
        all_done = True
        for t in remaining_targets:
            if t != 0 and _popcount(t) > 1:
                all_done = False
                break
        if all_done:
            break
        
        best_gain = 0
        best_pair = None
        best_val = None
        
        wire_ids = list(wire_vals.keys())
        if len(wire_ids) > 50:
            wire_ids = list(wire_vals.keys())[:n + 20] + list(wire_vals.keys())[-20:]
            seen = set()
            unique_ids = []
            for wid in wire_ids:
                if wid not in seen:
                    seen.add(wid)
                    unique_ids.append(wid)
            wire_ids = unique_ids
        
        for idx_a in range(len(wire_ids)):
            wa = wire_ids[idx_a]
            va = wire_vals[wa]
            if va == 0:
                continue
            for idx_b in range(idx_a + 1, len(wire_ids)):
                wb = wire_ids[idx_b]
                vb = wire_vals[wb]
                if vb == 0:
                    continue
                vx = va ^ vb
                if vx == 0 or vx in val_to_wire:
                    continue
                # Gain: how much this new wire helps decompose targets
                gain = 0
                for t in remaining_targets:
                    if t == 0:
                        continue
                    if (t & vx) == vx:
                        gain += _popcount(vx)
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (wa, wb)
                    best_val = vx
        
        if best_gain <= 2:
            break
        
        wa, wb = best_pair
        gates.append([wa, wb])
        wire_vals[best_val] = next_w
        val_to_wire[best_val] = next_w
        next_w += 1
    
    # Build outputs
    outputs = []
    for t in masks:
        if t == 0:
            outputs.append(const_zero)
            continue
        if _popcount(t) == 1:
            j = t.bit_length() - 1
            outputs.append(j)
            continue
        remaining = t
        parts = []
        while remaining != 0:
            if _popcount(remaining) == 1:
                j = remaining.bit_length() - 1
                parts.append(j)
                break
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
        
        if len(parts) == 1:
            outputs.append(parts[0])
        else:
            result = parts[0]
            for p in parts[1:]:
                v1 = wire_vals[result]
                v2 = wire_vals[p]
                res = v1 ^ v2
                found = val_to_wire.get(res)
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
    val_to_wire = {0: const_zero}
    for j in range(n):
        val_to_wire[1 << j] = j
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
            found = val_to_wire.get(res)
            if found is None:
                found = next_w
                next_w += 1
                gates.append([result, b])
                wire_vals[found] = res
                val_to_wire[res] = found
            result = found
        outputs.append(result)
    
    return gates, outputs

def _local_search(gates, outputs, masks, n, deadline):
    """
    Local search: remove unused gates and try gate replacements.
    """
    if time.time() > deadline:
        return gates, outputs
    
    const_zero = n
    wire_vals = {}
    for j in range(n):
        wire_vals[j] = (1 << j)
    wire_vals[const_zero] = 0
    gw = n + 1
    for g in gates:
        wire_vals[gw] = wire_vals[g[0]] ^ wire_vals[g[1]]
        gw += 1
    
    # Remove unused gates (iterate until stable)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Build usage map
        used_wires = set(outputs)
        for g in gates:
            used_wires.add(g[0])
            used_wires.add(g[1])
        
        new_gates = []
        new_wire_vals = {}
        for j in range(n):
            new_wire_vals[j] = (1 << j)
        new_wire_vals[const_zero] = 0
        gw = n + 1
        old_to_new = {}
        for gi, g in enumerate(gates):
            wire_id = n + 1 + gi
            if wire_id in used_wires:
                new_gates.append([old_to_new.get(g[0], g[0]), old_to_new.get(g[1], g[1])])
                new_wire_vals[gw] = wire_vals[wire_id]
                old_to_new[wire_id] = gw
                gw += 1
            else:
                old_to_new[wire_id] = None
        
        if len(new_gates) < len(gates):
            # Remap outputs
            new_outputs = []
            for out in outputs:
                if out < n + 1:
                    new_outputs.append(out)
                else:
                    mapped = old_to_new.get(out)
                    if mapped is not None:
                        new_outputs.append(mapped)
                    else:
                        # Should not happen if we only remove unused
                        new_outputs.append(out)
            gates = new_gates
            outputs = new_outputs
            wire_vals = new_wire_vals
            improved = True
    
    # Try replacing individual gates with existing wires
    if time.time() < deadline:
        val_to_wire = {}
        for wid, v in wire_vals.items():
            if wid not in val_to_wire:
                val_to_wire[v] = wid
        
        new_gates = list(gates)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for gi in range(len(new_gates)):
                if time.time() > deadline:
                    break
                w = n + 1 + gi
                v = wire_vals[w]
                # Check if this wire's value already exists in earlier wires
                for other_w, other_v in wire_vals.items():
                    if other_w == w:
                        continue
                    if other_v == v and other_w < w:
                        # Can potentially remove this gate
                        # Check if w is used by any later gate or output
                        used = False
                        for out in outputs:
                            if out == w:
                                used = True
                                break
                        if not used:
                            for g in new_gates:
                                if g[0] == w or g[1] == w:
                                    used = True
                                    break
                        if not used:
                            new_gates.pop(gi)
                            changed = True
                            break
            gates = new_gates
    
    return gates, outputs

def synthesize(matrix, depth_limit):
    """
    Synthesize an exact XOR circuit for the given binary matrix.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin (90s kill, leave 10s)
    
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
    
    # Build all candidates
    candidates = []
    
    # Direct
    gates_dir, outputs_dir = _build_direct(masks, n)
    candidates.append((len(gates_dir), gates_dir, outputs_dir))
    
    # Greedy CSE
    if time.time() < deadline:
        try:
            gates_cse, outputs_cse = _build_greedy_cse(masks, n, deadline)
            if _verify_circuit(gates_cse, outputs_cse, masks, n):
                candidates.append((len(gates_cse), gates_cse, outputs_cse))
        except Exception:
            pass
    
    # Basis greedy
    if time.time() < deadline:
        try:
            gates_basis, outputs_basis = _build_basis_greedy(masks, n, deadline)
            if _verify_circuit(gates_basis, outputs_basis, masks, n):
                candidates.append((len(gates_basis), gates_basis, outputs_basis))
        except Exception:
            pass
    
    # Pick best
    candidates.sort(key=lambda x: x[0])
    best_count, gates, outputs = candidates[0]
    
    # Local search on best
    if time.time() < deadline:
        gates, outputs = _local_search(gates, outputs, masks, n, deadline)
    
    # Final verification
    if not _verify_circuit(gates, outputs, masks, n):
        gates, outputs = _build_direct(masks, n)
    
    return {'gates': gates, 'outputs': outputs}
