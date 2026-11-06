import time
import numpy as np

def xor_baseline(matrix, depth_limit):
    """
    Fallback: Synthesize each row independently.
    This is not optimal in terms of shared gates, but always valid.
    """
    m, n = matrix.shape
    gates = []
    outputs = []
    # wire_map: mask -> wire_index
    # Inputs: 0..n-1
    # Zero: n
    # Gates start at n+1
    
    # We need to synthesize each row.
    # For a row mask `target`, we can decompose it.
    # Simple approach: if target is 0, output n.
    # If target is 1<<i, output i.
    # Else, we can just add gates to build it from inputs.
    # To be efficient, we can use a simple greedy:
    # Pick two inputs that are in the target, XOR them? No, that removes them.
    # We need to construct the sum.
    # Actually, for baseline, we can just create a gate for every pair of bits in the target?
    # No, that's O(k^2).
    # Better: Chain XORs.
    # Find all set bits in target.
    # If none, output n.
    # If one, output that input index.
    # If multiple, take the first bit index i.
    # Create gate: input i ^ input j -> new wire.
    # Then new wire ^ input k -> new wire.
    # Etc.
    
    # However, we can share gates between rows if we use the main search.
    # For baseline, we just do it independently to guarantee correctness.
    
    # To make baseline slightly better, we can memoize within the baseline call?
    # No, let's keep it simple and correct.
    
    # Actually, a better baseline is to use the same beam search but with width 1?
    # Or just a greedy construction.
    
    # Let's implement a simple greedy construction for each row.
    # This ensures we have a valid solution even if beam search fails or times out.
    
    # We will maintain a global wire_map for the baseline to avoid re-computing identical sub-circuits?
    # No, independent means independent. But we can still reuse the *inputs*.
    # We cannot reuse gates from other rows in an "independent" synthesis unless we explicitly build a shared circuit.
    # The task asks to minimize mean gate count. A shared circuit is better.
    # The main `synthesize` function does the shared search.
    # This fallback is just for safety.
    
    # Let's create a simple circuit for each row.
    # We'll use a local wire map for the whole circuit, but we won't try to share across rows in this function.
    # Wait, if I return a circuit with gates, the gates are global.
    # If I synthesize row 0 and use gates 1..5, then synthesize row 1, I should ideally reuse gates 1..5 if possible.
    # Since this is a fallback, I will just append gates for each row.
    
    global_wire_map = {}
    for i in range(n):
        global_wire_map[1 << i] = i
    global_wire_map[0] = n
    
    next_wire_id = n + 1
    
    for i in range(m):
        target = 0
        for j in range(n):
            if matrix[i, j]:
                target |= (1 << j)
        
        if target in global_wire_map:
            outputs.append(global_wire_map[target])
            continue
        
        # Decompose target
        bits = []
        t = target
        while t:
            lsb = t & -t
            bits.append(lsb)
            t ^= lsb
        
        if not bits:
            outputs.append(n)
            continue
        
        if len(bits) == 1:
            # Single bit
            idx = bits[0].bit_length() - 1
            outputs.append(idx)
            continue
        
        # Chain XOR
        # Start with first bit
        current_wire = global_wire_map[bits[0]]
        current_mask = bits[0]
        
        for k in range(1, len(bits)):
            next_bit_mask = bits[k]
            next_bit_idx = next_bit_mask.bit_length() - 1
            
            # Create gate: current_wire XOR input(next_bit_idx)
            in1 = current_wire
            in2 = next_bit_idx
            
            # Add gate
            gates.append([in1, in2])
            
            # New wire
            new_wire = next_wire_id
            next_wire_id += 1
            
            # Update map
            new_mask = current_mask ^ next_bit_mask
            global_wire_map[new_mask] = new_wire
            
            current_wire = new_wire
            current_mask = new_mask
        
        outputs.append(current_wire)
    
    return {'gates': gates, 'outputs': outputs}


def synthesize(matrix, depth_limit):
    start_time = time.time()
    # 90s limit, keep 5s buffer for safety
    deadline = start_time + 85.0
    
    m, n = matrix.shape
    
    # Convert rows to bitmasks
    rows = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        rows.append(mask)
    
    # Unique targets (we only need to synthesize unique masks)
    unique_targets = list(set(rows))
    
    # Initial state
    # Available wires: inputs (1<<0 ... 1<<(n-1)) and zero (0)
    initial_available = set([1 << i for i in range(n)] + [0])
    initial_targets = set(unique_targets)
    
    # Wire map: mask -> wire_index
    # Inputs: 0..n-1
    # Zero: n
    initial_wire_map = {}
    for i in range(n):
        initial_wire_map[1 << i] = i
    initial_wire_map[0] = n
    
    # Beam Search
    # State: (available_frozenset, targets_frozenset, gates_tuple, wire_map_tuple)
    # To make it hashable and comparable, we use tuples.
    # However, wire_map is a dict. We can store it as a tuple of (mask, wire) pairs sorted by mask.
    
    def map_to_tuple(wmap):
        return tuple(sorted(wmap.items()))
    
    def tuple_to_map(tup):
        return dict(tup)
    
    # Beam width
    beam_width = 50
    max_depth = depth_limit
    
    # Initial state
    init_avail_fs = frozenset(initial_available)
    init_targets_fs = frozenset(initial_targets)
    init_gates = ()
    init_wmap_t = map_to_tuple(initial_wire_map)
    
    current_beam = [(init_avail_fs, init_targets_fs, init_gates, init_wmap_t)]
    
    best_solution = None
    best_cost = float('inf')
    
    # Check initial state (if no targets, cost 0)
    if not init_targets_fs:
        best_cost = 0
        best_solution = (init_gates, tuple_to_map(init_wmap_t))
    
    for depth in range(max_depth):
        if time.time() > deadline - 2.0:
            break
        
        next_beam_candidates = []
        seen_states = set()
        
        for state in current_beam:
            avail_fs, targets_fs, gates, wmap_t = state
            wmap = tuple_to_map(wmap_t)
            
            if not targets_fs:
                cost = len(gates)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = (gates, wmap)
                continue
            
            avail_list = list(avail_fs)
            L = len(avail_list)
            
            # Generate all possible new wires
            # To reduce branching, we can prioritize wires that help targets.
            # But let's generate all and sort.
            
            for i in range(L):
                a = avail_list[i]
                for j in range(i + 1, L):
                    b = avail_list[j]
                    u = a ^ b
                    if u in avail_fs:
                        continue
                    
                    # New state
                    new_avail_fs = avail_fs | {u}
                    new_targets_fs = targets_fs
                    progress = 0
                    if u in new_targets_fs:
                        new_targets_fs = new_targets_fs - {u}
                        progress = 1
                    
                    new_gates = gates + ((wmap[a], wmap[b]),)
                    
                    new_wmap = dict(wmap)
                    new_wmap[u] = n + 1 + len(gates)
                    new_wmap_t = map_to_tuple(new_wmap)
                    
                    # Heuristic score
                    # Primary: number of gates (lower is better)
                    # Secondary: number of targets remaining (lower is better)
                    # Tertiary: number of available wires (lower is better? or higher?)
                    # Usually, having more available wires is good for future, but cost is high.
                    # Let's stick to (gates, targets_remaining)
                    
                    score = (len(new_gates), len(new_targets_fs))
                    
                    # Deduplication key
                    # If we have same avail and targets, we keep the one with fewer gates.
                    # Since we are expanding in layers of depth, and sorting by gates,
                    # the first one we see for a specific (avail, targets) pair will have the min gates
                    # IF we process candidates in order of gates.
                    # But we generate candidates from multiple parents.
                    # So we should collect all, sort, then dedup.
                    
                    next_beam_candidates.append((score, new_avail_fs, new_targets_fs, new_gates, new_wmap_t))
            
            # Prune and select
            if not next_beam_candidates:
                break
            
            # Sort by score
            next_beam_candidates.sort(key=lambda x: x[0])
            
            current_beam = []
            seen_states = set()
            
            for c in next_beam_candidates:
                score, avail_fs, targets_fs, gates, wmap_t = c
                key = (avail_fs, targets_fs)
                if key in seen_states:
                    continue
                seen_states.add(key)
                current_beam.append((avail_fs, targets_fs, gates, wmap_t))
                
                if len(current_beam) >= beam_width:
                    break
            
            if not current_beam:
                break
            
            # Check if any state in current_beam is a solution
            for state in current_beam:
                avail_fs, targets_fs, gates, wmap_t = state
                if not targets_fs:
                    cost = len(gates)
                    if cost < best_cost:
                        best_cost = cost
                        best_solution = (gates, tuple_to_map(wmap_t))

    # Construct output
    if best_solution:
        gates, wmap = best_solution
        # Map original rows to wire indices
        outputs = []
        for r in rows:
            outputs.append(wmap[r])
        return {'gates': [list(g) for g in gates], 'outputs': outputs}
    else:
        # Fallback to baseline
        return xor_baseline(matrix, depth_limit)
