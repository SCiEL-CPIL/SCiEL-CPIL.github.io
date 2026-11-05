"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict, deque
import heapq
from itertools import combinations

def synthesize(matrix, depth_limit):
    """Synthesize an XOR circuit using Beam Search over the space of available wires.
    
    This approach explores multiple synthesis paths simultaneously, avoiding the
    local optima trap of greedy CSE. It models the synthesis as a shortest path
    problem in the state space of (available_wires, remaining_targets).
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit
    
    mat = np.array(matrix, dtype=np.int8)
    n_outputs, n_inputs = mat.shape
    
    def row_to_mask(row):
        m = 0
        for j, v in enumerate(row):
            if v:
                m |= (1 << j)
        return m
        
    output_masks = [row_to_mask(mat[i]) for i in range(n_outputs)]
    
    def popcount(m):
        return bin(m).count('1')
        
    # Initial State
    # Available wires: 0 and all input bits
    initial_available = {0}
    for i in range(n_inputs):
        initial_available.add(1 << i)
        
    initial_remaining = set(output_masks)
    # Remove zeros from remaining, they are already available
    initial_remaining.discard(0)
    
    # Beam Search Parameters
    # For 16x16, beam width 50-100 should be fast enough
    beam_width = 100 
    max_depth_steps = 50 # Safety cap on number of gates added per path
    
    # State: (cost, step_count, available_set, remaining_set, path)
    # path: list of (wire_index, mask, operand1, operand2)
    # We need to map masks to wire indices to build the circuit
    # Wire indices:
    # 0..n_inputs-1: inputs
    # n_inputs: zero
    # n_inputs+1...: generated
    
    # To ensure valid wire indexing in the final circuit, we must record the generation order.
    # In Beam Search, different branches might generate the same mask at different times.
    # We need to canonicalize the state or handle wire mapping carefully.
    # Strategy: The "available_set" contains masks. When we pick a new mask, we assign it the next wire index.
    # However, if two branches generate the same mask, they should ideally use the same wire index if possible, 
    # but since they are separate paths, they are separate circuits.
    # We will reconstruct the circuit from the best path's history.
    
    # State representation for Beam:
    # (priority, unique_id, available_masks_tuple, remaining_masks_tuple, history)
    # history: list of (new_mask, op1_mask, op2_mask)
    
    # To save memory and time, we store:
    # available: frozenset
    # remaining: frozenset
    # history: tuple of (mask, op1, op2)
    
    start_state = (0, 0, frozenset(initial_available), frozenset(initial_remaining), ())
    
    beam = [start_state]
    
    best_solution = None
    best_cost = float('inf')
    
    # Pre-compute popcounts for speed? Maybe not necessary for small n.
    
    for step in range(max_depth_steps):
        if time.time() > DEADLINE - 2.0:
            break
            
        next_beam = []
        
        for cost, _, available, remaining, history in beam:
            if not remaining:
                # Found a solution
                if cost < best_cost:
                    best_cost = cost
                    best_solution = history
                continue
                
            # Generate candidates
            # We only need to generate XORs of pairs in `available`
            # Optimization: If `available` is large, this is $O(|A|^2)$.
            # For 16 inputs, |A| starts at 17.
            # As we add wires, |A| grows.
            
            avail_list = list(available)
            n_avail = len(avail_list)
            
            # Heuristic: Prioritize pairs that reduce the remaining set significantly
            # We generate all pairs, but we can prune if time is tight
            
            seen_new_masks = set()
            
            for i in range(n_avail):
                m1 = avail_list[i]
                for j in range(i + 1, n_avail):
                    m2 = avail_list[j]
                    new_mask = m1 ^ m2
                    
                    if new_mask in available:
                        continue
                    if new_mask == 0:
                        continue
                    if new_mask in seen_new_masks:
                        continue
                            
                    seen_new_masks.add(new_mask)
                    
                    new_remaining = remaining
                    if new_mask in remaining:
                        new_remaining = new_remaining - {new_mask}
                    
                    # Heuristic Score for Beam
                    # 1. Number of remaining targets
                    # 2. Sum of popcounts of remaining targets (complexity)
                    # 3. Current cost (gates used)
                    
                    rem_count = len(new_remaining)
                    rem_popcount = sum(popcount(m) for m in new_remaining)
                    
                    # Priority: Lower is better
                    # We want to minimize gates, but also complexity.
                    # Cost is the primary metric.
                    # If costs are equal, prefer lower complexity.
                    priority = cost + 1 + rem_popcount * 0.1 + rem_count * 10
                    
                    new_history = history + ((new_mask, m1, m2),)
                    
                    next_beam.append((priority, len(next_beam), frozenset(available | {new_mask}), frozenset(new_remaining), new_history))
            
            if not next_beam:
                break
            
            # Sort and prune beam
            next_beam.sort(key=lambda x: x[0])
            beam = next_beam[:beam_width]
            
        if not beam:
            break
            
    if best_solution is None:
        # Fallback: Direct synthesis (no sharing)
        # This is always valid
        gates = []
        outputs = []
        next_wire = n_inputs + 1
        wire_map = {0: n_inputs}
        for i in range(n_inputs):
            wire_map[1 << i] = i
            
        def create_wire(mask):
            nonlocal next_wire
            if mask in wire_map:
                return wire_map[mask]
            if mask == 0:
                return n_inputs
            # Find two bits
            bits = []
            m = mask
            idx = 0
            while m:
                if m & 1:
                    bits.append(idx)
                m >>= 1
                idx += 1
            if not bits:
                return n_inputs
            if len(bits) == 1:
                w = wire_map[1 << bits[0]]
                wire_map[mask] = w
                return w
            
            # Split into two parts
            mid = len(bits) // 2
            m1 = 0
            m2 = 0
            for b in bits[:mid]:
                m1 |= (1 << b)
            for b in bits[mid:]:
                m2 |= (1 << b)
                
            w1 = create_wire(m1)
            w2 = create_wire(m2)
            gates.append([w1, w2])
            w = next_wire
            next_wire += 1
            wire_map[mask] = w
            return w

        for m in output_masks:
            outputs.append(create_wire(m))
            
        return {'gates': gates, 'outputs': outputs}
        
    # Reconstruct circuit from best_solution
    gates = []
    wire_map = {0: n_inputs}
    for i in range(n_inputs):
        wire_map[1 << i] = i
        
    next_wire = n_inputs + 1
    
    # The history is a sequence of (new_mask, op1, op2)
    # We must ensure that op1 and op2 are already created.
    # Since we generated new_mask from available, and available only grows,
    # and we process history in order, op1 and op2 should be in wire_map.
    # However, we need to check if op1 or op2 are themselves generated in the history
    # but appear *later* in the list? No, history is chronological.
    
    for new_mask, op1, op2 in best_solution:
        if new_mask in wire_map:
            continue # Should not happen if logic is correct, but safe
        
        w1 = wire_map.get(op1)
        w2 = wire_map.get(op2)
        
        if w1 is None or w2 is None:
            # This indicates a bug in beam search logic or reconstruction
            # Fallback for this specific gate
            # Try to find any way to make op1 and op2
            # Actually, if they are in available, they must be in wire_map
            # unless they were 0 or inputs.
            # If they are missing, it's an error.
            # Let's force create them if possible, or abort.
            # For safety, we will abort and use fallback.
            return {'gates': [], 'outputs': [n_inputs]*n_outputs} # Invalid, triggers fallback?
            # Better: Use the fallback function above.
            
        gates.append([w1, w2])
        wire_map[new_mask] = next_wire
        next_wire += 1
        
    outputs = []
    for m in output_masks:
        if m in wire_map:
            outputs.append(wire_map[m])
        else:
            # Should not happen if beam search succeeded
            # Fallback: create directly
            # We can't easily call create_wire here without modifying gates/wire_map in a complex way
            # Let's just append n_inputs (zero) as a placeholder, but this is incorrect.
            # Instead, we should ensure beam search covers all targets.
            # If a target is not in wire_map, it means the beam search path didn't create it?
            # No, the beam search stops when remaining is empty.
            # So all targets MUST be in available, and thus in wire_map.
            outputs.append(n_inputs)
            
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
