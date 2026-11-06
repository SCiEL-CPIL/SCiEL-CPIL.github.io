"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
import sys

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 55.0  # Safety margin for 60s limit
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert rows to bitmasks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)
        
    # Unique targets
    targets = set(row_masks)
    # Remove 0 if it's a target, as it's always available
    if 0 in targets:
        targets.discard(0)
        
    if not targets:
        # All rows are zero
        return {'gates': [], 'outputs': [n] * m}

    # Initial state: all inputs and zero are available
    initial_wires = set()
    for j in range(n):
        initial_wires.add(1 << j)
    initial_wires.add(0)
    
    initial_state = frozenset(initial_wires)
    
    # Check if already solved
    if targets.issubset(initial_state):
        # Map targets to wire IDs
        wire_id_map = {0: n}
        for j in range(n):
            wire_id_map[1 << j] = j
            
        outputs = []
        for mask in row_masks:
            outputs.append(wire_id_map[mask])
            
        return {'gates': [], 'outputs': outputs}

    # A* / Beam Search
    # State: frozenset of available masks
    # Priority: Number of gates used (cost)
    # We store: (cost, state)
    
    # To allow reconstruction, we store parent info
    # parent_map[state] = (prev_state, a, b, new_mask)
    parent_map = {}
    parent_map[initial_state] = None
    
    # Heuristic: Number of targets not yet in state
    def heuristic(state):
        return len(targets - state)
    
    # Beam search parameters
    BEAM_WIDTH = 2000
    MAX_STATES = 50000
    
    # Priority queue: (f_score, g_score, state)
    # f = g + h
    # g = number of gates
    open_list = []
    heapq.heappush(open_list, (heuristic(initial_state), 0, initial_state))
    
    closed_set = set()
    best_cost = float('inf')
    best_state = None
    
    while open_list:
        if time.time() > DEADLINE:
            break
            
        f, g, state = heapq.heappop(open_list)
        
        if state in closed_set:
            continue
            
        closed_set.add(state)
        
        # Check goal
        if targets.issubset(state):
            best_cost = g
            best_state = state
            break
            
        # If we have expanded too many states, stop
        if len(closed_set) > MAX_STATES:
            break
            
        # Generate successors
        # We need to find pairs (a, b) in state such that a^b is not in state
        # To limit branching factor, we can restrict to "promising" new wires
        # A new wire w is promising if:
        # 1. w is a target
        # 2. w reduces the heuristic (i.e., w is "close" to a target)
        
        # Optimization: Only consider pairs where a^b is "useful"
        # Useful if:
        # - a^b is in targets
        # - a^b has a smaller popcount than some target (potential simplification)
        
        # To avoid O(N^2) per state, we can sample or limit.
        # Given small N (<=16), |state| is small initially, grows later.
        # Max |state| is 2^16 = 65536. Pairs ~ 2^32 which is too many.
        # We MUST prune.
        
        # Pruning Strategy:
        # Only add a new wire w if:
        # 1. w is a target.
        # 2. w is a "basis" element for the remaining targets?
        
        # Alternative: Only generate wires that are linear combinations of 
        # the current state that are also linear combinations of the targets?
        # This is hard to check quickly.
        
        # Simpler Pruning:
        # Only consider new wires w if w is in the span of the targets?
        # Actually, any wire in the state is in the span of inputs.
        # We only need wires that help form the targets.
        
        # Let's restrict the beam to only keep states where the number of 
        # "extra" wires (wires not in targets and not inputs) is bounded.
        
        # For now, let's just generate all a^b but limit the number of 
        # successors added to the beam.
        
        successors = []
        
        # Get list of wires in state
        wires_list = list(state)
        L = len(wires_list)
        
        # If L is large, sampling might be needed, but for n<=16, 
        # the number of *useful* wires is small.
        
        # To speed up, we only check pairs if L is not too large, 
        # or we use a heuristic to pick pairs.
        
        # Heuristic: Pick pairs (a,b) such that a^b has low popcount 
        # or is a target.
        
        # Let's just iterate, but break if we find too many candidates
        candidate_count = 0
        max_candidates_per_state = 100
        
        for i in range(L):
            if candidate_count >= max_candidates_per_state:
                break
            a = wires_list[i]
            for j in range(i + 1, L):
                b = wires_list[j]
                w = a ^ b
                if w in state:
                    continue
                
                # Check if w is promising
                # 1. Is w a target?
                # 2. Does w help? (e.g., popcount(w) < min_popcount(remaining_targets))
                
                is_target = w in targets
                # Simple heuristic: prefer low popcount
                pc = w.bit_count()
                
                # Only consider if it's a target or has very low popcount
                if not is_target and pc > 4: 
                    continue
                
                new_state = state | {w}
                
                # Check if this new state is already closed
                if new_state in closed_set:
                    continue
                
                # Calculate new cost
                new_g = g + 1
                new_h = heuristic(new_state)
                new_f = new_g + new_h
                
                # Store parent info
                parent_map[new_state] = (state, a, b, w)
                
                successors.append((new_f, new_g, new_state))
                candidate_count += 1
                
                if candidate_count >= max_candidates_per_state:
                    break
            
            if candidate_count >= max_candidates_per_state:
                break
        
        # Add successors to open list, respecting beam width
        # We can just push all and let the heap handle it, but limit size
        for item in successors:
            heapq.heappush(open_list, item)
            
        # Trim open list to beam width
        if len(open_list) > BEAM_WIDTH:
            # Keep the best BEAM_WIDTH items
            # This is O(N log N), acceptable for 2000
            open_list.sort()
            open_list = open_list[:BEAM_WIDTH]
            # Re-heapify
            heapq.heapify(open_list)

    # If we found a solution, reconstruct
    if best_state is not None:
        # Reconstruct gates
        # We need to map masks to wire IDs
        # Trace back from best_state to initial_state
        
        gates = []
        # We need to store the sequence of (a, b, w) in reverse
        seq = []
        curr = best_state
        while parent_map[curr] is not None:
            prev_state, a, b, w = parent_map[curr]
            seq.append((a, b, w))
            curr = prev_state
            
        # seq is in reverse order (last gate to first)
        # Reverse it
        seq.reverse()
        
        # Assign wire IDs
        # Inputs: 0..n-1
        # Zero: n
        # Gates: n+1, n+2, ...
        
        wire_id_map = {}
        for j in range(n):
            wire_id_map[1 << j] = j
        wire_id_map[0] = n
        
        next_wire_id = n + 1
        
        # We need to ensure that when we process a gate (a, b, w), 
        # a and b already have wire IDs.
        # Since we built the state by adding one wire at a time, 
        # and a,b were in the previous state, they should have IDs.
        
        for a, b, w in seq:
            # Get IDs for a and b
            # Note: a and b are masks
            id_a = wire_id_map[a]
            id_b = wire_id_map[b]
            
            # Create gate
            gates.append([id_a, id_b])
            
            # Assign ID to w
            wire_id_map[w] = next_wire_id
            next_wire_id += 1
            
        # Map outputs
        outputs = []
        for mask in row_masks:
            outputs.append(wire_id_map[mask])
            
        return {'gates': gates, 'outputs': outputs}
        
    # Fallback: Greedy Tree Construction
    # If search fails or times out, use the greedy approach from the seed
    next_wire = n + 1
    gates = []
    memo = {0: n}
    for j in range(n):
        memo[1 << j] = j
        
    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w
        
    def build_from_bits(bits_list):
        if not bits_list:
            return n
        if len(bits_list) == 1:
            return bits_list[0]
            
        wires = []
        masks = []
        for b in bits_list:
            wires.append(b)
            masks.append(1 << b)
            
        while len(wires) > 1:
            new_wires = []
            new_masks = []
            for i in range(0, len(wires), 2):
                if i + 1 < len(wires):
                    a_w = wires[i]
                    a_m = masks[i]
                    b_w = wires[i+1]
                    b_m = masks[i+1]
                    
                    new_m = a_m ^ b_m
                    if new_m in memo:
                        new_wires.append(memo[new_m])
                        new_masks.append(new_m)
                    else:
                        new_w = add_gate(a_w, b_w)
                        memo[new_m] = new_w
                        new_wires.append(new_w)
                        new_masks.append(new_m)
                else:
                    new_wires.append(wires[i])
                    new_masks.append(masks[i])
            wires = new_wires
            masks = new_masks
            
        return wires[0]
        
    outputs = []
    for mask in row_masks:
        if mask in memo:
            outputs.append(memo[mask])
        else:
            bits = [j for j in range(n) if (mask >> j) & 1]
            w = build_from_bits(bits)
            memo[mask] = w
            outputs.append(w)
            
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
