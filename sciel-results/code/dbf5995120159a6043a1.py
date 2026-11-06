"""
Exact XOR-circuit synthesis using greedy sub-expression reuse.
Instead of choosing a basis and building trees per output, we build all
output wires simultaneously in a single pass, greedily selecting XOR pairs
that maximize reuse across all remaining output masks.
"""
import time
import numpy as np
from collections import Counter, defaultdict

def _greedy_xor_synthesis(M, n_cols, deadline):
    """
    Greedy sub-expression reuse synthesis.
    
    Each wire (input or gate) has an associated mask (bitmask of which inputs it
    depends on). We maintain the set of available wires and their masks.
    At each step, we look at all pairs of available wires and pick the pair
    whose XOR mask appears in the most remaining output expressions.
    
    For efficiency with up to 16x16 matrices, we use integer bitmasks.
    """
    n_rows, n_cols = M.shape
    
    # Convert M rows to integer bitmasks
    output_masks = []
    for i in range(n_rows):
        mask = 0
        for j in range(n_cols):
            if M[i, j]:
                mask |= (1 << j)
        output_masks.append(mask)
    
    # Zero wire
    zero_mask = 0
    zero_wire = n_cols
    
    # Available wires: wire_id -> mask
    wire_mask = {}
    for j in range(n_cols):
        wire_mask[j] = (1 << j)
    wire_mask[zero_wire] = zero_mask
    
    # Track which outputs still need to be resolved
    # remaining[i] = mask of inputs still needed for output i
    remaining = list(output_masks)
    # Wire that produces the current partial result for each output
    # Initially each output is just its full mask (no gates yet)
    # We'll track: for each output, the set of wires that XOR to give it
    # Actually, let's think differently.
    
    # Better approach: we maintain a set of "available" wires (inputs + gates created so far).
    # For each output, we maintain a multiset (or set) of available wires whose XOR equals the output.
    # Initially, each output j has the set of input wires where M[j,i]=1.
    # When we create a new gate w = a XOR b, for any output whose wire set contains both a and b,
    # we can replace {a, b} with {w} in that output's wire set.
    
    # This is expensive to track. Let me use a different formulation.
    
    # Alternative: We process outputs one at a time but share a global cache.
    # For each output, we compute its XOR using a greedy algorithm that reuses
    # previously computed sub-expressions.
    
    # Actually, the cleanest approach for small matrices:
    # 1. Collect all unique masks that appear as rows of M.
    # 2. Use a priority-based approach: start with all input wires available.
    # 3. Repeatedly find the pair (a, b) of available wires such that
    #    mask(a) XOR mask(b) appears in the most "active" outputs.
    # 4. Create gate for that pair, add to available wires, update active outputs.
    
    # Let me implement a simpler but effective version:
    # Process outputs in order of decreasing Hamming weight.
    # For each output, use a recursive/memoized approach to build its XOR tree,
    # but share sub-expressions via a global cache.
    
    # Global cache: mask -> wire_id
    cache = {zero_mask: zero_wire}
    for j in range(n_cols):
        cache[1 << j] = j
    
    gates = []
    next_wire_id = n_cols + 1
    
    def get_wire(mask):
        """Get or create a wire for the given mask."""
        nonlocal next_wire_id
        if mask in cache:
            return cache[mask]
        if mask == 0:
            return zero_wire
        if mask == (1 << 0) * 0:  # shouldn't happen
            return zero_wire
        
        # Find a decomposition: split mask into two parts
        # Try to find a sub-mask that's already cached
        best_split = None
        best_saved = 0
        
        # Try all possible splits: pick a bit to separate
        # For efficiency, try a few strategies:
        # Strategy 1: split by lowest set bit
        # Strategy 2: try to find a cached sub-mask
        
        # First, check if any proper sub-mask is in cache
        sub = mask
        while sub:
            if sub in cache and sub != mask:
                complement = mask ^ sub
                if complement in cache and complement != mask:
                    # Both parts cached!
                    w1 = cache[sub]
                    w2 = cache[complement]
                    gates.append([w1, w2])
                    w_id = next_wire_id
                    next_wire_id += 1
                    cache[mask] = w_id
                    return w_id
            sub = (sub - 1) & mask
        
        # No cached sub-mask found. Split mask in half by bits.
        # Count bits
        bits = []
        m = mask
        b = 0
        while m:
            if m & 1:
                bits.append(b)
            m >>= 1
            b += 1
        
        if len(bits) == 1:
            # Single bit, should be in cache
            w_id = next_wire_id
            next_wire_id += 1
            cache[mask] = w_id
            # This shouldn't happen for input wires, but handle gracefully
            gates.append([zero_wire, zero_wire])  # placeholder, shouldn't occur
            return w_id
        
        # Split into two roughly equal halves
        mid = len(bits) // 2
        part1 = 0
        for i in range(mid):
            part1 |= (1 << bits[i])
        part2 = mask ^ part1
        
        w1 = get_wire(part1)
        w2 = get_wire(part2)
        
        gates.append([w1, w2])
        w_id = next_wire_id
        next_wire_id += 1
        cache[mask] = w_id
        return w_id
    
    # Process outputs
    # Sort by decreasing weight to maximize reuse
    order = sorted(range(n_rows), key=lambda i: bin(output_masks[i]).count('1'), reverse=True)
    
    outputs = [0] * n_rows
    for idx in order:
        if time.time() > deadline:
            # Fallback: use remaining outputs with direct computation
            for i in range(n_rows):
                if outputs[i] == 0 and i not in [o for o in range(n_rows) if outputs[o] != 0]:
                    pass
            # Just fill in with what we have
            for i in range(n_rows):
                if outputs[i] == 0:
                    # Direct: XOR all inputs
                    ms = output_masks[i]
                    if ms == 0:
                        outputs[i] = zero_wire
                    else:
                        outputs[i] = get_wire(ms)
            break
        outputs[idx] = get_wire(output_masks[idx])
    
    return {'gates': gates, 'outputs': outputs}

def _greedy_pair_synthesis(M, n_cols, deadline):
    """
    Alternative greedy: simultaneously build all outputs by repeatedly
    creating the gate that reduces total remaining work the most.
    
    State: for each output i, a set S_i of available wires whose XOR = output i.
    Initially S_i = {j : M[i,j] = 1}.
    At each step, find pair (a,b) available such that replacing a,b with a^b
    in as many S_i as possible.
    """
    n_rows, n_cols = M.shape
    
    zero_wire = n_cols
    
    # Available wires: id -> mask
    wire_mask = {}
    for j in range(n_cols):
        wire_mask[j] = (1 << j)
    wire_mask[zero_wire] = 0
    
    # For each output, track the set of wires (as frozenset for hashing)
    # Use a list of sets for mutability
    output_wires = []
    for i in range(n_rows):
        ws = set()
        for j in range(n_cols):
            if M[i, j]:
                ws.add(j)
        output_wires.append(ws)
    
    gates = []
    next_id = n_cols + 1
    available = set(wire_mask.keys())
    
    # Precompute: for each pair of available wires, which outputs contain both
    # This is expensive. Instead, for each output, count how many pairs it has.
    
    max_steps = n_rows * n_cols  # safety
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Check if all outputs are resolved (size <= 1)
        all_done = all(len(ws) <= 1 for ws in output_wires)
        if all_done:
            break
        
        # Find the best pair to create
        # For each pair (a, b) in available, a < b:
        #   new_mask = wire_mask[a] ^ wire_mask[b]
        #   count how many outputs contain both a and b
        # Pick the pair with max count (tiebreak: by mask weight or arbitrary)
        
        best_pair = None
        best_count = 0
        best_new_mask = 0
        
        avail_list = sorted(available)
        n_avail = len(avail_list)
        
        # For efficiency, only consider pairs where at least one output has both
        # Build: for each output, if len(ws) >= 2, consider all pairs in ws
        # But we need to count across outputs
        
        # Use a counter: pair -> count of outputs containing it
        pair_count = Counter()
        for i in range(n_rows):
            ws = output_wires[i]
            if len(ws) < 2:
                continue
            ws_list = sorted(ws)
            for a_idx in range(len(ws_list)):
                a = ws_list[a_idx]
                if a not in wire_mask:
                    continue
                for b_idx in range(a_idx + 1, len(ws_list)):
                    b = ws_list[b_idx]
                    if b not in wire_mask:
                        continue
                    pair_count[(a, b)] += 1
        
        if not pair_count:
            break
        
        # Find pair with max count
        for pair, cnt in pair_count.items():
            if cnt > best_count:
                best_count = cnt
                best_pair = pair
                a, b = pair
                best_new_mask = wire_mask[a] ^ wire_mask[b]
        
        if best_pair is None:
            break
        
        a, b = best_pair
        new_mask = best_new_mask
        
        # Check if this mask already has a wire
        if new_mask in wire_mask.values():
            # Find existing wire with this mask
            for wid, wm in wire_mask.items():
                if wm == new_mask:
                    new_wire = wid
                    break
        else:
            new_wire = next_id
            next_id += 1
            wire_mask[new_wire] = new_mask
            gates.append([a, b])
            available.add(new_wire)
        
        # Update outputs: replace {a, b} with {new_wire}
        for i in range(n_rows):
            ws = output_wires[i]
            if a in ws and b in ws:
                ws.remove(a)
                ws.remove(b)
                ws.add(new_wire)
        
        # Remove a and b from available if no longer needed by any output
        # Actually, keep them available since they might be used in future pairs
        # But we can clean up wires not in any output
        # For now, keep all wires available
    
    # Resolve outputs
    outputs = []
    for i in range(n_rows):
        ws = output_wires[i]
        if len(ws) == 0:
            outputs.append(zero_wire)
        elif len(ws) == 1:
            outputs.append(next(iter(ws)))
        else:
            # Multiple wires remain: XOR them
            # This shouldn't happen if we ran enough steps, but handle it
            ws_list = list(ws)
            # Build XOR tree
            result = ws_list[0]
            for w in ws_list[1:]:
                gates.append([result, w])
                result = next_id
                next_id += 1
            outputs.append(result)
    
    return {'gates': gates, 'outputs': outputs}

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0
    
    M = np.array(matrix, dtype=np.uint8)
    n_rows, n_cols = M.shape
    
    # Handle trivial cases
    if n_rows == 0:
        return {'gates': [], 'outputs': []}
    
    # Check if all rows are zero
    if not M.any():
        return {'gates': [], 'outputs': [n_cols] * n_rows}
    
    # For small problems, use the greedy pair synthesis (simultaneous)
    # For larger, use the per-output cache approach
    if n_rows * n_cols <= 256:
        # Small: use greedy pair synthesis
        circuit = _greedy_pair_synthesis(M, n_cols, deadline)
    else:
        # Larger: use per-output with cache
        circuit = _greedy_xor_synthesis(M, n_cols, deadline)
    
    # Verify correctness
    # Build wire masks from the circuit
    if len(circuit['gates']) > 0 or any(o != n_cols for o in circuit['outputs']):
        wire_masks = {}
        for j in range(n_cols):
            wire_masks[j] = (1 << j)
        wire_masks[n_cols] = 0  # zero wire
        
        for idx, (a, b) in enumerate(circuit['gates']):
            wire_id = n_cols + 1 + idx
            wire_masks[wire_id] = wire_masks.get(a, 0) ^ wire_masks.get(b, 0)
        
        # Verify outputs
        for i in range(n_rows):
            expected = 0
            for j in range(n_cols):
                if M[i, j]:
                    expected |= (1 << j)
            actual = wire_masks.get(circuit['outputs'][i], 0)
            if expected != actual:
                # Fallback to simple per-output synthesis
                circuit = _simple_synthesis(M, n_cols)
                break
    
    return circuit

def _simple_synthesis(M, n_cols):
    """Simple fallback: build each output as a flat XOR of inputs."""
    n_rows, n_cols = M.shape
    zero_wire = n_cols
    gates = []
    next_id = n_cols + 1
    outputs = []
    
    for i in range(n_rows):
        ones = [j for j in range(n_cols) if M[i, j]]
        if not ones:
            outputs.append(zero_wire)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            # Build XOR tree
            result = ones[0]
            for j in ones[1:]:
                gates.append([result, j])
                result = next_id
                next_id += 1
            outputs.append(result)
    
    return {'gates': gates, 'outputs': outputs}
