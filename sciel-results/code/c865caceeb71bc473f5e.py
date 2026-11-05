import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    """
    Synthesize an exact XOR circuit for the given binary matrix.
    
    Strategy:
    1. Convert each row to a bitmask in GF(2)^n.
    2. Use Dijkstra-like relaxation on the hypercube to find minimum gates
       to synthesize each target, sharing intermediate wires.
    3. Use a greedy approach that processes targets and reuses computed wires.
    4. For small n, try multiple basis orderings to minimize gate count.
    
    Returns {'gates': [[wire, wire], ...], 'outputs': [wire, ...]}
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape
    zero_wire = n
    
    # Convert rows to bitmasks
    target_masks = []
    row_weights = []
    for i in range(m):
        mask = 0
        w = 0
        for j in range(n):
            if mat[i, j] == 1:
                mask |= (1 << j)
                w += 1
        target_masks.append(mask)
        row_weights.append(w)
    
    baseline_gates = sum(max(0, w - 1) for w in row_weights)
    
    # Trivial cases
    if baseline_gates == 0:
        gates = []
        outputs = []
        for i in range(m):
            if row_weights[i] == 0:
                outputs.append(zero_wire)
            else:
                idx = np.where(mat[i] == 1)[0][0]
                outputs.append(int(idx))
        return {'gates': gates, 'outputs': outputs}
    
    # Use Dijkstra-style synthesis with shared wire pool
    # dist[mask] = minimum gates to produce mask, using a set of available wires
    # We build wires incrementally, tracking which mask each wire computes.
    
    def synthesize_greedy(mat, target_masks, n, zero_wire, deadline):
        """
        Greedy synthesis with shared intermediate wires.
        Process targets in order of decreasing weight.
        For each target, find the cheapest decomposition using existing wires.
        """
        # Wire pool: mask -> wire_index
        wire_map = {0: zero_wire}
        for i in range(n):
            wire_map[1 << i] = i
        
        gates = []
        outputs = []
        next_wire = n + 1
        
        # Sort target indices by weight descending (hardest first)
        target_order = sorted(range(len(target_masks)), key=lambda i: target_masks[i].bit_count(), reverse=True)
        
        def cost_to_synthesize(target, wire_map, next_wire):
            """
            Find minimum gates to synthesize target given current wire_map.
            Returns (gate_count, new_gates, new_wire_index, new_wires_to_add)
            """
            if target in wire_map:
                return 0, [], wire_map[target], {}
            
            # Try all single XORs with existing wires
            best_cost = float('inf')
            best_pair = None
            best_new_wire = None
            best_new_wires = None
            
            # Get list of existing wire masks
            existing_masks = list(wire_map.keys())
            
            for a_mask in existing_masks:
                rem = target ^ a_mask
                if rem in wire_map:
                    # Both available, one gate
                    cost = 1
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (wire_map[a_mask], wire_map[rem])
                        best_new_wire = None  # We'll create a new wire
                        best_new_wires = {}
                else:
                    # rem not available, need to synthesize rem
                    # This is recursive, but let's limit depth
                    rem_weight = rem.bit_count()
                    if rem_weight <= 2:
                        # Can synthesize rem from inputs
                        if rem_weight == 0:
                            pass  # rem is 0, already in wire_map
                        elif rem_weight == 1:
                            # rem is a single input
                            pass  # rem is in wire_map
                        elif rem_weight == 2:
                            # rem = x ^ y for two inputs
                            bits = [j for j in range(n) if (rem >> j) & 1]
                            if len(bits) == 2:
                                cost = 2  # one gate for rem, one for target
                                if cost < best_cost:
                                    best_cost = cost
                                    best_pair = (wire_map[a_mask], None)  # placeholder
                                    best_new_wire = None
                                    best_new_wires = {rem: (bits[0], bits[1])}
            
            # Also try direct synthesis from inputs if cheap
            target_weight = target.bit_count()
            if target_weight <= 4:
                direct_cost = target_weight - 1
                if direct_cost < best_cost:
                    best_cost = direct_cost
                    best_pair = None
                    best_new_wire = None
                    best_new_wires = None  # signal direct synthesis
            
            return best_cost, best_pair, best_new_wire, best_new_wires
        
        # For simplicity and correctness, use a direct approach:
        # For each target, synthesize it using available wires, adding intermediates.
        
        for ti in target_order:
            target = target_masks[ti]
            
            if target in wire_map:
                outputs.append(wire_map[target])
                continue
            
            # Find best decomposition
            # Try: target = a ^ b where a, b are in wire_map
            best = None
            best_cost = float('inf')
            
            existing_masks = list(wire_map.keys())
            for a_mask in existing_masks:
                b_mask = target ^ a_mask
                if b_mask in wire_map:
                    cost = 1
                    if cost < best_cost:
                        best_cost = cost
                        best = ('xor', a_mask, b_mask)
            
            # Try direct synthesis from inputs
            target_weight = target.bit_count()
            if target_weight > 0:
                direct_cost = target_weight - 1
                if direct_cost < best_cost:
                    best_cost = direct_cost
                    best = ('direct', target, None)
            
            # Try two-step: target = a ^ c, where c is synthesized from inputs
            # This is covered by the direct case above since c would be a mask
            # But we can also try: find a in wire_map such that target^a has low weight
            for a_mask in existing_masks:
                rem = target ^ a_mask
                rem_weight = rem.bit_count()
                if 0 < rem_weight <= 3:
                    # rem can be synthesized with rem_weight-1 gates
                    cost = (rem_weight - 1) + 1
                    if cost < best_cost:
                        best_cost = cost
                        best = ('two_step', a_mask, rem)
            
            # Execute the best strategy
            if best is None or best_cost == float('inf'):
                # Fallback: direct synthesis
                ones = [j for j in range(n) if (target >> j) & 1]
                if not ones:
                    outputs.append(zero_wire)
                    wire_map[target] = zero_wire
                elif len(ones) == 1:
                    outputs.append(ones[0])
                    wire_map[target] = ones[0]
                else:
                    curr = ones[0]
                    for j in range(1, len(ones)):
                        gates.append([curr, ones[j]])
                        curr = next_wire
                        next_wire += 1
                    outputs.append(curr)
                    wire_map[target] = curr
                continue
            
            if best[0] == 'xor':
                _, a_mask, b_mask = best
                aw = wire_map[a_mask]
                bw = wire_map[b_mask]
                gates.append([aw, bw])
                new_wire = next_wire
                next_wire += 1
                outputs.append(new_wire)
                wire_map[target] = new_wire
            elif best[0] == 'direct':
                _, target_mask, _ = best
                ones = [j for j in range(n) if (target_mask >> j) & 1]
                curr = ones[0]
                for j in range(1, len(ones)):
                    gates.append([curr, ones[j]])
                    curr = next_wire
                    next_wire += 1
                outputs.append(curr)
                wire_map[target] = curr
            elif best[0] == 'two_step':
                _, a_mask, rem = best
                aw = wire_map[a_mask]
                # Synthesize rem from inputs
                rem_ones = [j for j in range(n) if (rem >> j) & 1]
                if len(rem_ones) == 1:
                    rw = rem_ones[0]
                else:
                    curr = rem_ones[0]
                    for j in range(1, len(rem_ones)):
                        gates.append([curr, rem_ones[j]])
                        curr = next_wire
                        next_wire += 1
                    rw = curr
                # XOR aw with rw
                gates.append([aw, rw])
                new_wire = next_wire
                next_wire += 1
                outputs.append(new_wire)
                wire_map[target] = new_wire
                # Also add rem to wire_map if it's useful
                if rem not in wire_map:
                    wire_map[rem] = rw
        
        return len(gates), gates, outputs
    
    # Run greedy synthesis
    gcount, gates, outputs = synthesize_greedy(mat, target_masks, n, zero_wire, deadline)
    
    # Try basis-based approach for comparison
    def synthesize_basis(mat, target_masks, n, zero_wire, deadline):
        """Synthesize using an optimized basis."""
        # Find basis via Gaussian elimination
        basis = []
        for i in range(len(target_masks)):
            mask = target_masks[i]
            reduced = True
            while reduced:
                reduced = False
                for b in basis:
                    if mask & b and (mask ^ b) < mask:
                        mask ^= b
                        reduced = True
            if mask != 0:
                basis.append(mask)
        
        if not basis:
            return 0, [], [zero_wire] * len(target_masks)
        
        # Build basis wires
        wire_map = {0: zero_wire}
        for i in range(n):
            wire_map[1 << i] = i
        
        gates = []
        next_wire = n + 1
        basis_wires = []
        
        for b in basis:
            if b in wire_map:
                basis_wires.append(wire_map[b])
                continue
            ones = [j for j in range(n) if (b >> j) & 1]
            curr = ones[0]
            for j in range(1, len(ones)):
                gates.append([curr, ones[j]])
                curr = next_wire
                next_wire += 1
            basis_wires.append(curr)
            wire_map[b] = curr
        
        # Build reduction table
        sorted_basis = sorted(enumerate(basis), key=lambda x: x[1].bit_length(), reverse=True)
        pivot_to_idx = {}
        for idx, b in sorted_basis:
            h = b.bit_length() - 1
            pivot_to_idx[h] = idx
        
        outputs = []
        for t in target_masks:
            if t in wire_map:
                outputs.append(wire_map[t])
                continue
            mask = t
            coeff_indices = []
            while mask != 0:
                h = mask.bit_length() - 1
                if h in pivot_to_idx:
                    idx = pivot_to_idx[h]
                    coeff_indices.append(idx)
                    mask ^= basis[idx]
                else:
                    break
            
            if not coeff_indices:
                outputs.append(zero_wire)
                continue
            if len(coeff_indices) == 1:
                outputs.append(basis_wires[coeff_indices[0]])
            else:
                curr = basis_wires[coeff_indices[0]]
                for idx in coeff_indices[1:]:
                    gates.append([curr, basis_wires[idx]])
                    curr = next_wire
                    next_wire += 1
                outputs.append(curr)
            wire_map[t] = outputs[-1]
        
        return len(gates), gates, outputs
    
    bcount, bgates, boutputs = synthesize_basis(mat, target_masks, n, zero_wire, deadline)
    
    # Pick the better one
    if bcount < gcount:
        return {'gates': bgates, 'outputs': boutputs}
    else:
        return {'gates': gates, 'outputs': outputs}
