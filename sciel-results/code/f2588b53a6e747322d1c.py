"""XOR-circuit synthesis using iterative greedy pairing to minimize intermediate weights."""

def synthesize(matrix, depth_limit):
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert to list of tuples
    rows = [tuple(matrix[i]) for i in range(m)]
    
    # Wire mapping
    # 0..n-1: inputs
    # n: zero
    # n+1...: gates
    
    gates = []
    wire_to_vec = {}
    vec_to_wire = {}
    
    # Initialize inputs
    for i in range(n):
        vec = [0] * n
        vec[i] = 1
        v = tuple(vec)
        wire_to_vec[i] = v
        vec_to_wire[v] = i
    
    # Initialize zero
    zero_vec = tuple([0] * n)
    wire_to_vec[n] = zero_vec
    vec_to_wire[zero_vec] = n
    
    next_wire = n + 1
    
    def xor_vec(a, b):
        return tuple(x ^ y for x, y in zip(a, b))
    
    def add_gate(w1, w2):
        nonlocal next_wire
        v1 = wire_to_vec[w1]
        v2 = wire_to_vec[w2]
        new_v = xor_vec(v1, v2)
        
        # Check if we already have this vector
        if new_v in vec_to_wire:
            return vec_to_wire[new_v]
        
        gates.append([w1, w2])
        wire_to_vec[next_wire] = new_v
        vec_to_wire[new_v] = next_wire
        new_wire = next_wire
        next_wire += 1
        return new_wire

    # We will maintain a list of "current targets" (vectors we still need to compute).
    # We also maintain a mapping from original output index to its current representation.
    # Representation: (base_vector, list_of_xors) where base_vector is a vector in the current target set
    # and list_of_xors are vectors from the original target set that have been XORed out.
    
    # Actually, a simpler way:
    # We start with targets = rows.
    # We maintain a list `current_targets` of vectors.
    # We maintain a list `orig_to_current` mapping original index to current index in `current_targets`.
    # We maintain a list `current_to_orig` mapping current index to original index? No, because we merge.
    
    # Let's use the strategy from the prompt:
    # 1. Start with targets = rows.
    # 2. Repeatedly find a pair (i, j) such that wt(targets[i] ^ targets[j]) < wt(targets[i]).
    # 3. Replace targets[i] with targets[i] ^ targets[j].
    # 4. Record that original_output[i] = new_targets[i] ^ original_output[j].
    # 5. Continue until no more improvement or max iterations.
    # 6. Compute all final targets using the recursive get_wire.
    # 7. Reconstruct outputs using the recorded relationships.
    
    # This is tricky to track dependencies correctly.
    
    # Alternative approach:
    # We build the circuit greedily.
    # We maintain a set of "available" vectors (wires).
    # We have a set of "required" vectors (outputs).
    # We repeatedly pick a required vector r.
    # If r is available, mark it done.
    # Else, find an available vector a such that r ^ a has lower weight than r.
    # Add r ^ a to the required set (if not already there).
    # Add a gate r = a ^ (r ^ a).
    # This creates a dependency graph.
    
    # This is essentially what the current code does, but it's recursive and might not be optimal in ordering.
    
    # Let's try the "Pairing on Targets" approach more carefully.
    
    # We will maintain a list of "jobs". Each job is a vector.
    # Initially, jobs = rows.
    # We also maintain a dictionary `dependency` mapping original_index to a list of (xor_with, other_index) or similar.
    
    # Actually, let's just use the recursive get_wire but with a better ordering and a limit on recursion depth to avoid stack overflow, falling back to direct computation if needed.
    
    # The current code's issue might be that `get_wire` can recurse deeply or inefficiently.
    
    # Let's refine the `get_wire` function:
    # 1. Check cache.
    # 2. Find best split `vec = a ^ b` where `a` is in cache and `b` has minimal weight.
    # 3. If `b` is in cache, add gate and return.
    # 4. Else, if `weight(b) < weight(vec)`, recurse on `b`.
    # 5. Else, compute `vec` directly from inputs.
    
    # This is basically what the current code does.
    
    # Let's try to improve the "best split" selection.
    # Instead of just minimizing `weight(b)`, we can look ahead.
    # But that's expensive.
    
    # Let's try a different heuristic:
    # Sort outputs by weight.
    # For each output, if not in cache, find the available wire `a` that minimizes `weight(vec ^ a)`.
    # Let `b = vec ^ a`.
    # If `b` is in cache, add gate.
    # Else, add `b` to the "to_compute" queue and add gate later?
    # No, we need to compute `b` first.
    
    # The recursive approach is fine if we ensure termination.
    # Termination is guaranteed if `weight(b) < weight(vec)`.
    # If `weight(b) >= weight(vec)`, we compute directly.
    
    # Let's optimize the direct computation:
    # Instead of XORing all inputs one by one, we can use a faster method?
    # For small n, it's fine.
    
    # Let's try to improve the score by:
    # 1. Trying multiple orderings of outputs (e.g., weight, random) and picking the best.
    # 2. Using a more sophisticated split heuristic.
    
    # Given the time limit, let's stick to one robust method.
    
    # Let's try the "Pairing" method properly.
    
    # We maintain a list of vectors `targets`.
    # We maintain a list `orig_indices` corresponding to `targets`.
    # Initially, targets = rows, orig_indices = [0, ..., m-1].
    
    # We also maintain a dictionary `expr` mapping original_index to a vector in the *final* basis?
    # No, that's hard.
    
    # Let's use the following:
    # We will generate a set of intermediate wires.
    # We will assign each output to a wire.
    
    # Let's go back to the recursive get_wire, but make it iterative to avoid recursion limits and improve performance.
    
    # Actually, the recursive get_wire is clean. Let's just ensure it's efficient.
    
    # One improvement: when choosing `a` to minimize `weight(vec ^ a)`, we can restrict `a` to be one of the recently added wires or low-weight wires.
    
    # Let's try to improve the baseline by using a better initial set of wires.
    # Currently, we only have inputs and zero.
    # We could pre-compute some common intermediates?
    
    # Let's just run the current logic but with a small optimization:
    # In `get_wire`, when scanning `wire_to_vec`, we can skip wires that are "far" from `vec`.
    # But we need to scan all to find the min.
    
    # Let's try to parallelize? No, standard library only.
    
    # Let's try to improve the score by trying a few different strategies and picking the best.
    # Strategy 1: Sort by weight.
    # Strategy 2: Random order.
    # Strategy 3: Sort by weight, but compute "heavy" outputs first?
    
    # Let's implement Strategy 1 and 2 and pick the one with fewer gates.
    
    import random
    
    def try_synthesize(order):
        nonlocal gates, wire_to_vec, vec_to_wire, next_wire
        
        # Reset state
        gates = []
        wire_to_vec = {}
        vec_to_wire = {}
        
        for i in range(n):
            vec = [0] * n
            vec[i] = 1
            v = tuple(vec)
            wire_to_vec[i] = v
            vec_to_wire[v] = i
            
        zero_vec = tuple([0] * n)
        wire_to_vec[n] = zero_vec
        vec_to_wire[zero_vec] = n
        
        next_wire = n + 1
        
        import sys
        sys.setrecursionlimit(10000)
        
        def get_wire(vec):
            vec = tuple(vec)
            if vec in vec_to_wire:
                return vec_to_wire[vec]
            
            # Find best a in wire_to_vec
            best_a = None
            best_diff = None
            min_wt = float('inf')
            
            for w, v in wire_to_vec.items():
                diff = xor_vec(vec, v)
                wt = sum(diff)
                if wt < min_wt:
                    min_wt = wt
                    best_a = w
                    best_diff = diff
                    
            if best_diff is None:
                return n
                
            wt_vec = sum(vec)
            if min_wt >= wt_vec:
                # Direct computation
                idxs = [i for i in range(n) if vec[i] == 1]
                if not idxs:
                    return n
                w_res = idxs[0]
                for idx in idxs[1:]:
                    w_res = add_gate(w_res, idx)
                return w_res
                
            w_diff = get_wire(best_diff)
            w_res = add_gate(best_a, w_diff)
            return w_res

        outputs = [0] * m
        for idx in order:
            outputs[idx] = get_wire(rows[idx])
            
        return {'gates': gates, 'outputs': outputs}

    # Try a few orders
    orders = []
    
    # Order 1: By weight
    orders.append(sorted(range(m), key=lambda i: sum(rows[i])))
    
    # Order 2: Random
    for _ in range(5):
        o = list(range(m))
        random.shuffle(o)
        orders.append(o)
    
    best_result = None
    best_gate_count = float('inf')
    
    for order in orders:
        result = try_synthesize(order)
        if len(result['gates']) < best_gate_count:
            best_gate_count = len(result['gates'])
            best_result = result
            
    return best_result
