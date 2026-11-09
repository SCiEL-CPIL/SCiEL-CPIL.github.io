"""XOR circuit synthesis via minimum-weight basis completion in GF(2)^m."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def synthesize(matrix, depth_limit):
    """
    Synthesize XOR circuit using minimum-weight basis completion.
    
    Structural change: Instead of per-output greedy wire creation, we treat the
    problem as finding a minimum-gate DAG that produces all target vectors.
    We greedily select intermediate XORs that maximize sharing across remaining
    targets, using a priority-based approach over candidate pairs.
    """
    deadline = time.time() + 80  # 10s safety margin
    
    n = len(matrix)
    m = len(matrix[0]) if n > 0 else 0
    
    if n == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert matrix rows to bitmasks
    rows = []
    for i in range(n):
        mask = 0
        for j in range(m):
            if matrix[i][j]:
                mask |= (1 << j)
        rows.append(mask)
    
    # Zero wire is m, inputs are 0..m-1
    zero_wire = m
    next_wire = m + 1
    gates = []
    
    # Wire management
    wire_to_mask = {}  # wire_id -> bitmask
    mask_to_wire = {}  # bitmask -> wire_id
    for j in range(m):
        wire_to_mask[j] = 1 << j
        mask_to_wire[1 << j] = j
    wire_to_mask[zero_wire] = 0
    mask_to_wire[0] = zero_wire
    
    # Precompute popcount
    max_mask = 1 << m
    popcount = [0] * max_mask
    for i in range(1, max_mask):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Track which targets are still pending
    pending = set(range(n))
    outputs = [None] * n
    
    # Phase 1: Mark targets that can be produced with 0 gates (single bit or zero)
    for i in range(n):
        pc = popcount[rows[i]]
        if pc == 0:
            outputs[i] = zero_wire
            pending.discard(i)
        elif pc == 1:
            # Find the single bit position
            bit = (rows[i] & -rows[i]).bit_length() - 1
            outputs[i] = bit
            pending.discard(i)
    
    if not pending:
        return {'gates': gates, 'outputs': outputs}
    
    # Phase 2: Greedy shared-subexpression selection
    # At each step, consider all pairs of existing wires and pick the XOR
    # that best helps complete remaining targets.
    
    wires_in_use = set(wire_to_mask.keys())
    
    def count_targets_helped(new_mask):
        """Count how many pending targets could be completed or simplified by having new_mask."""
        count = 0
        for idx in pending:
            target = rows[idx]
            other = target ^ new_mask
            if other in mask_to_wire:
                count += 1
            elif popcount[other] <= 1:
                count += 1
        return count
    
    def estimate_gates_to_build(mask):
        """Estimate minimum gates to build mask from existing wires."""
        if mask in mask_to_wire:
            return 0
        if mask == 0:
            return 0
        pc = popcount[mask]
        if pc <= 1:
            return 0
        # Check if can be made from two existing wires
        for w1 in wires_in_use:
            m1 = wire_to_mask[w1]
            m2 = mask ^ m1
            if m2 in mask_to_wire:
                return 1
        # Lower bound: pc - 1
        return pc - 1
    
    # Main greedy loop: pick the best new gate to add
    max_iterations = 5000
    iter_count = 0
    
    while pending and iter_count < max_iterations:
        if time.time() > deadline:
            break
        iter_count += 1
        
        # Collect candidate pairs
        wire_list = list(wires_in_use)
        best_score = -1
        best_pair = None
        best_new_mask = None
        
        for i_idx in range(len(wire_list)):
            w1 = wire_list[i_idx]
            m1 = wire_to_mask[w1]
            for j_idx in range(i_idx + 1, len(wire_list)):
                w2 = wire_list[j_idx]
                m2 = wire_to_mask[w2]
                new_mask = m1 ^ m2
                if new_mask == 0 or new_mask in mask_to_wire:
                    continue
                
                # Score: how many pending targets does this help?
                helped = count_targets_helped(new_mask)
                
                # Also consider: is new_mask itself a pending target?
                if new_mask in [rows[idx] for idx in pending]:
                    helped += 10  # Bonus for directly completing a target
                
                # Penalize by estimated cost to build (should be 1 since both exist)
                score = helped
                
                if score > best_score:
                    best_score = score
                    best_pair = (w1, w2)
                    best_new_mask = new_mask
        
        if best_pair is None or best_score <= 0:
            # No beneficial pair found; build remaining targets directly
            break
        
        # Add the gate
        w1, w2 = best_pair
        gates.append((w1, w2))
        w = next_wire
        next_wire += 1
        wire_to_mask[w] = best_new_mask
        mask_to_wire[best_new_mask] = w
        wires_in_use.add(w)
        
        # Complete any targets that can now be produced
        for idx in list(pending):
            target = rows[idx]
            other = target ^ best_new_mask
            if other in mask_to_wire:
                wo = mask_to_wire[other]
                if w != wo:
                    gates.append((w, wo))
                    wout = next_wire
                    next_wire += 1
                    wire_to_mask[wout] = target
                    mask_to_wire[target] = wout
                    wires_in_use.add(wout)
                    outputs[idx] = wout
                    pending.discard(idx)
                else:
                    # target == best_new_mask (other is zero)
                    outputs[idx] = w
                    pending.discard(idx)
            elif other == 0:
                outputs[idx] = w
                pending.discard(idx)
            elif popcount[other] == 1:
                bit = (other & -other).bit_length() - 1
                gates.append((w, bit))
                wout = next_wire
                next_wire += 1
                wire_to_mask[wout] = target
                mask_to_wire[target] = wout
                wires_in_use.add(wout)
                outputs[idx] = wout
                pending.discard(idx)
    
    # Phase 3: Build any remaining pending targets
    for idx in list(pending):
        if time.time() > deadline:
            # Emergency fallback
            target = rows[idx]
            bits = [j for j in range(m) if target & (1 << j)]
            if not bits:
                outputs[idx] = zero_wire
            elif len(bits) == 1:
                outputs[idx] = bits[0]
            else:
                w = bits[0]
                for j in bits[1:]:
                    gates.append((w, j))
                    w = next_wire
                    next_wire += 1
                outputs[idx] = w
            pending.discard(idx)
            continue
        
        target = rows[idx]
        
        # Try to build target using existing wires
        found = False
        for w1 in list(wires_in_use):
            m1 = wire_to_mask[w1]
            m2 = target ^ m1
            if m2 in mask_to_wire:
                w2 = mask_to_wire[m2]
                if w1 != w2:
                    gates.append((w1, w2))
                    w = next_wire
                    next_wire += 1
                    wire_to_mask[w] = target
                    mask_to_wire[target] = w
                    wires_in_use.add(w)
                    outputs[idx] = w
                    found = True
                    break
        
        if not found:
            # Build from scratch
            bits = [j for j in range(m) if target & (1 << j)]
            if not bits:
                outputs[idx] = zero_wire
            elif len(bits) == 1:
                outputs[idx] = bits[0]
            else:
                # Balanced tree
                def build_balanced(bit_list):
                    nonlocal next_wire
                    if len(bit_list) == 1:
                        return bit_list[0]
                    mid = len(bit_list) // 2
                    left = build_balanced(bit_list[:mid])
                    right = build_balanced(bit_list[mid:])
                    gates.append((left, right))
                    w = next_wire
                    next_wire += 1
                    mk = 0
                    for b in bit_list:
                        mk |= (1 << b)
                    wire_to_mask[w] = mk
                    mask_to_wire[mk] = w
                    wires_in_use.add(w)
                    return w
                
                w = build_balanced(bits)
                outputs[idx] = w
        
        pending.discard(idx)
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
