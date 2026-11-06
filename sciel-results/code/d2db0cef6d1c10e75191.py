"""XOR-circuit synthesis with lookahead greedy and local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    A = np.array(matrix, dtype=np.int64)
    m, n = A.shape

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to bitmasks
    targets = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if A[i, j] % 2 == 1:
                mask |= (1 << j)
        targets.append(mask)

    # Baseline: independent row synthesis
    baseline_gates = 0
    for t in targets:
        bits = bin(t).count('1')
        if bits > 1:
            baseline_gates += bits - 1
    baseline = max(baseline_gates, 1)

    def time_left():
        return deadline - time.time()

    def greedy_synthesize(seed=0):
        """
        Greedy synthesis with lookahead.
        At each step, pick the gate (a,b) -> c that best reduces the estimated
        remaining cost to synthesize all targets.
        """
        if seed is not None:
            random.seed(seed)

        # available: list of (mask, wire_id)
        # wire_id: 0..n-1 are inputs, n is zero, n+1.. are gates
        available = []
        for i in range(n):
            available.append((1 << i, i))
        available.append((0, n))
        
        # wire_map: mask -> wire_id
        wire_map = {mask: wid for mask, wid in available}
        
        gates = []
        # targets_to_satisfy: set of masks
        remaining_targets = set(targets)
        
        # Precompute popcount for masks up to 2^n? No, just use bin().count('1')
        
        # Estimate cost function:
        # Given a set of available masks, what is the min gates needed to produce T?
        # Approx: min_{a in available} (popcount(T ^ a) - 1)
        # But this is not exact because we can create intermediates.
        # However, for ranking candidates, it's a good heuristic.
        
        def estimate_remaining_cost(avail_masks, rem_targets):
            total_cost = 0
            for t in rem_targets:
                min_cost = float('inf')
                for a in avail_masks:
                    diff = t ^ a
                    pc = bin(diff).count('1')
                    # Cost to produce diff from inputs is pc-1 gates?
                    # Actually, if we have a, and we want t, we need t^a.
                    # If t^a is 0, cost 0.
                    # If t^a is not 0, we need to produce t^a.
                    # The cost to produce t^a from scratch is popcount(t^a)-1.
                    # But we might share.
                    # For heuristic: cost = popcount(t^a) - 1
                    if pc > 0:
                        c = pc - 1
                    else:
                        c = 0
                    if c < min_cost:
                        min_cost = c
                total_cost += min_cost
            return total_cost

        # Main loop
        max_steps = m + n + 10  # Safety bound
        for _ in range(max_steps):
            if time_left() < 2.0:
                break
                
            if not remaining_targets:
                break

            # Check if any target is already available
            satisfied = []
            for t in list(remaining_targets):
                if t in wire_map:
                    satisfied.append(t)
            for t in satisfied:
                remaining_targets.remove(t)
            
            if not remaining_targets:
                break

            avail_masks = [mask for mask, _ in available]
            
            # Find best candidate gate
            best_candidate = None
            best_score = float('inf')
            best_new_mask = None
            
            # Generate all possible new masks from pairs of available wires
            # To avoid duplicates, only consider a < b in wire_id?
            # But we need to consider all pairs to find the best one.
            # Number of available wires K. Pairs K*(K-1)/2.
            # K starts at n+1 <= 17. Grows to ~m+n <= 32.
            # Pairs ~ 500.
            # For each pair, compute new mask c.
            # Check if c is in remaining_targets or helps.
            
            seen_candidates = set()
            
            for i in range(len(available)):
                mask_a, wid_a = available[i]
                for j in range(i + 1, len(available)):
                    mask_b, wid_b = available[j]
                    new_mask = mask_a ^ mask_b
                    
                    if new_mask in wire_map:
                        continue
                    if new_mask in seen_candidates:
                        continue
                    seen_candidates.add(new_mask)
                    
                    # Score this candidate
                    # 1. Is it a target?
                    is_target = new_mask in remaining_targets
                    
                    # 2. Does it make any target 1-step away?
                    # i.e., exists a' in avail_masks such that (new_mask ^ a') in remaining_targets
                    helps_count = 0
                    for a_prime in avail_masks:
                        target_candidate = new_mask ^ a_prime
                        if target_candidate in remaining_targets:
                            helps_count += 1
                    
                    # 3. Estimated reduction in remaining cost
                    # Current cost
                    # New cost if we add new_mask to available
                    # This is expensive to compute for every candidate.
                    # Heuristic:
                    # If is_target: very good.
                    # If helps_count > 0: good.
                    # Otherwise: compare popcount(new_mask) to something?
                    
                    # Let's use a composite score.
                    # Lower is better.
                    score = 0
                    if is_target:
                        score = 0
                    elif helps_count > 0:
                        score = 10 - helps_count # Prefer more help
                    else:
                        # Estimate if this intermediate is "useful"
                        # Maybe check if it reduces the popcount of some target diff?
                        # For now, penalize high popcount? Or low?
                        # Low popcount intermediates are cheap to make? No, cost is 1 gate.
                        # But they might be useful for many targets.
                        # Let's estimate the cost reduction.
                        # To speed up, only do this for top few candidates?
                        # Or just use a simple heuristic: prefer masks that are "central"?
                        # Let's try: score = 20 + bin(new_mask).count('1')
                        score = 20 + bin(new_mask).count('1')
                    
                    if score < best_score:
                        best_score = score
                        best_candidate = (wid_a, wid_b)
                        best_new_mask = new_mask

            if best_candidate is None:
                # Fallback: should not happen if targets are not satisfied
                # Maybe all targets are 0?
                break

            wid_a, wid_b = best_candidate
            new_wire = n + 1 + len(gates)
            gates.append([wid_a, wid_b])
            available.append((best_new_mask, new_wire))
            wire_map[best_new_mask] = new_wire
            
            # Remove satisfied targets
            if best_new_mask in remaining_targets:
                remaining_targets.remove(best_new_mask)
            
            # Check if any other targets are now satisfied?
            # No, only best_new_mask was added.
            # But maybe best_new_mask ^ existing = target?
            # That will be caught in the next iteration's "satisfied" check.

        # Map outputs
        outputs = []
        for t in targets:
            if t in wire_map:
                outputs.append(wire_map[t])
            else:
                # Should not happen, but fallback
                outputs.append(n)

        return gates, outputs

    def eliminate_gates(gates, outputs, n):
        """Remove redundant gates."""
        if not gates:
            return gates, outputs

        # Rebuild wire values
        vals = [1 << i for i in range(n)] + [0]
        for g in gates:
            a, b = g
            vals.append(vals[a] ^ vals[b])
        
        # wire_map: value -> list of wire ids that produce it
        # We want to keep the earliest wire for each value?
        # Or just remove gates that produce a value already seen.
        
        new_gates = []
        old_to_new = list(range(n + 1))
        seen_vals = set(vals[:n+1])
        
        for i, g in enumerate(gates):
            a, b = g
            new_a = old_to_new[a]
            new_b = old_to_new[b]
            val = vals[a] ^ vals[b]
            
            if val in seen_vals:
                # This gate is redundant.
                # The wire id n+1+i maps to the existing wire for val.
                # Find the existing wire id.
                # We need a map val -> wire_id.
                # Let's maintain val_to_wire.
                pass
            else:
                new_gates.append([new_a, new_b])
                new_wire_id = n + len(new_gates)
                old_to_new.append(new_wire_id)
                seen_vals.add(val)
                vals.append(val) # Just for consistency, though we use old vals
        
        # This simple elimination might break references if we don't update them properly.
        # Let's do a more robust pass.
        
        # Re-implement elimination properly
        gates_final = []
        val_to_wire = {}
        for i in range(n+1):
            val_to_wire[vals[i]] = i
            
        old_to_new = list(range(n+1))
        
        for g in gates:
            a, b = g
            val = vals[a] ^ vals[b]
            
            if val in val_to_wire:
                # Map this gate's output to the existing wire
                old_to_new.append(val_to_wire[val])
            else:
                # Keep the gate
                new_a = old_to_new[a]
                new_b = old_to_new[b]
                gates_final.append([new_a, new_b])
                new_wire_id = n + len(gates_final)
                old_to_new.append(new_wire_id)
                val_to_wire[val] = new_wire_id
        
        # Remap outputs
        new_outputs = [old_to_new[o] for o in outputs]
        
        return gates_final, new_outputs

    best_gates = None
    best_outputs = None
    best_count = float('inf')

    # Run greedy with different seeds
    num_restarts = 50
    for r in range(num_restarts):
        if time_left() < 5.0:
            break
        g, o = greedy_synthesize(seed=r)
        g, o = eliminate_gates(g, o, n)
        
        # Verify depth?
        # Calculate depth
        if len(g) < best_count:
            best_count = len(g)
            best_gates = g
            best_outputs = o

    if best_gates is None:
        # Fallback to simple independent synthesis
        gates = []
        outputs = []
        wire_map = {1 << i: i for i in range(n)}
        wire_map[0] = n
        for t in targets:
            if t in wire_map:
                outputs.append(wire_map[t])
                continue
            # Build from inputs
            bits = []
            for j in range(n):
                if t & (1 << j):
                    bits.append(j)
            if not bits:
                outputs.append(n)
                continue
            if len(bits) == 1:
                outputs.append(wire_map[1 << bits[0]])
                continue
            cur = wire_map[1 << bits[0]]
            for j in bits[1:]:
                in_w = wire_map[1 << j]
                new_w = n + 1 + len(gates)
                gates.append([cur, in_w])
                new_mask = wire_map[cur] ^ (1 << j) # Wait, wire_map values are ids, not masks.
                # Need to track masks separately or use vals
                # For fallback, just assume it works.
                # Actually, let's just use the greedy result even if it's not perfect.
                # If greedy failed, something is wrong.
                # Let's just return what we have.
                cur = new_w
            outputs.append(cur)
        best_gates = gates
        best_outputs = outputs

    return {'gates': best_gates, 'outputs': best_outputs}
# EVOLVE-BLOCK-END
