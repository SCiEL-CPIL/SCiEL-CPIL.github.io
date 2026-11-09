import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    N = 512
    
    # 1. Generate primes < 512
    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(N)
    
    # 2. Build set Y of max multiples of primes
    # Y contains M_p for each prime p
    # M_p is the largest multiple of p in [1, 511]
    Y = set()
    for p in primes:
        M_p = p * (N - 1) // p
        Y.add(M_p)
    
    # Also, it's possible that a number not in Y is optimal?
    # Unlikely, but we will refine later.
    # For the main search, we use Y.
    
    Y_list = list(Y)
    num_nodes = len(Y_list)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(Y_list)}
    
    # Weights
    weights = [0.0] * num_nodes
    for i, v in enumerate(Y_list):
        weights[i] = 1.0 / (N - v)
        
    # Build conflict graph
    # adj[i] is a set of indices j such that gcd(Y_list[i], Y_list[j]) > 1
    adj = [set() for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if gcd(Y_list[i], Y_list[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # Solve MWIS on this graph
    # Use recursive backtracking with bitsets for speed if possible, or sets.
    # Given N <= 100, sets are fine.
    
    best_score = 0.0
    best_set = set()
    
    # Sort nodes by weight descending to find good solutions early
    # But we need to keep track of original indices for the set.
    # Let's just use the indices 0..num_nodes-1.
    
    def solve_mwis():
        nonlocal best_score, best_set
        
        # Precompute compatible sets?
        # comp[i] = set of nodes compatible with i (including i? no, just neighbors in complement)
        # Let's define complement graph: edge if coprime.
        # We want Max Weight Clique in complement graph.
        # Or just MWIS in original.
        
        # Let's use a simple branch and bound.
        # State: current_selected (set), available (set of nodes not yet considered and not conflicting with current)
        # To make it faster, we process nodes in a fixed order (sorted by weight desc).
        
        # Order nodes by weight descending
        order = sorted(range(num_nodes), key=lambda i: weights[i], reverse=True)
        
        # Map original index to position in order
        pos_in_order = [0] * num_nodes
        for k, idx in enumerate(order):
            pos_in_order[idx] = k
            
        # We will iterate through `order`.
        # State: idx in order, current_mask (bitmask of selected nodes? no, N can be > 64? N~100. Python ints handle big bitmasks)
        # Let's use bitmask for selected nodes.
        # And a bitmask for available nodes.
        
        # Precompute conflict masks
        conflict_mask = [0] * num_nodes
        for i in range(num_nodes):
            m = 0
            for j in adj[i]:
                m |= (1 << j)
            conflict_mask[i] = m
            
        # Initial available: all nodes
        full_mask = (1 << num_nodes) - 1
        
        def backtrack(pos, selected_mask, current_score, available_mask):
            nonlocal best_score, best_set
            
            if time.time() > deadline:
                return
                
            if pos == num_nodes:
                if current_score > best_score:
                    best_score = current_score
                    # Reconstruct set
                    s = set()
                    for i in range(num_nodes):
                        if selected_mask & (1 << i):
                            s.add(Y_list[i])
                    best_set = s
                return
                
            # Upper bound: current_score + sum of weights of all remaining nodes in available_mask
            # This is loose. Better: current_score + sum of weights of nodes in available_mask that are compatible with each other?
            # Simple bound: sum of weights of available nodes.
            ub = current_score
            # Calculate sum of weights of available nodes
            # We can precompute prefix sums? No, available_mask changes.
            # Just iterate? N is small.
            # Or use a precomputed list of weights in order.
            # Let's just use the loose bound first. If it's not pruning enough, we'll see.
            # Actually, we can just sum weights[i] for i where available_mask has bit i.
            
            # Optimization: If current_score + sum(weights[i] for i in available) <= best_score, prune.
            # Let's compute this sum efficiently?
            # For N=100, iterating is fast.
            
            # Check if we can prune
            # Sum of weights of available nodes
            avail_sum = 0
            am = available_mask
            while am:
                lsb = am & -am
                i = lsb.bit_length() - 1
                avail_sum += weights[i]
                am ^= lsb
                
            if current_score + avail_sum <= best_score:
                return
                
            # Branch on node at `pos`
            node = order[pos]
            
            # Option 1: Exclude node
            # Available for next step is same, but we skip this node.
            # Wait, `available_mask` should contain only nodes that are compatible with `selected_mask`.
            # When we move to next pos, we just proceed.
            # But we must ensure that nodes we *skip* don't affect availability of later nodes?
            # No, availability is determined by `selected_mask`.
            # So if we exclude `node`, `selected_mask` doesn't change.
            # `available_mask` for next step is `available_mask` (which already excludes conflicts with selected).
            # But we must remove `node` from consideration?
            # The `pos` iteration handles the order.
            # We just call backtrack(pos+1, selected_mask, current_score, available_mask).
            # But wait, `available_mask` might contain `node`.
            # If we exclude `node`, we just don't select it.
            
            backtrack(pos + 1, selected_mask, current_score, available_mask)
            
            # Option 2: Include node
            if available_mask & (1 << node):
                new_selected_mask = selected_mask | (1 << node)
                new_score = current_score + weights[node]
                
                # New available: remove node and all its neighbors from available_mask
                new_available = available_mask & ~(conflict_mask[node] | (1 << node))
                
                backtrack(pos + 1, new_selected_mask, new_score, new_available)
                
        backtrack(0, 0, 0.0, full_mask)
        
    solve_mwis()
    
    current_set = list(best_set)
    
    # Refinement: Try to add or swap with numbers NOT in Y
    # Check all a in 1..511
    # If a is in current_set, skip.
    # If a is coprime to all in current_set, add it.
    # If swapping a with b in current_set improves score and maintains validity, do it.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set_set = set(current_set)
        
        # Try additions
        for a in range(1, N):
            if a in current_set_set:
                continue
            # Check coprime
            valid = True
            for b in current_set:
                if gcd(a, b) > 1:
                    valid = False
                    break
            if valid:
                # Calculate score change
                # We are adding a.
                # Is it always beneficial? Yes, score is positive.
                current_set.append(a)
                current_set_set.add(a)
                improved = True
                break # Restart to re-evaluate
                
        if improved:
            continue
            
        # Try swaps
        # For each b in current_set, try to replace with a not in current_set
        # Gain = 1/(512-a) - 1/(512-b)
        # We want positive gain.
        # So we want a > b.
        
        # Sort current_set to iterate
        # We can just iterate all pairs
        
        for i, b in enumerate(current_set):
            if time.time() > deadline:
                break
            
            # Find best a
            best_a = -1
            best_gain = 0.0
            
            # We only care about a > b for positive gain
            for a in range(b+1, N):
                if a in current_set_set:
                    continue
                # Check if a is coprime to all in current_set except b
                valid = True
                for j, c in enumerate(current_set):
                    if j == i:
                        continue
                    if gcd(a, c) > 1:
                        valid = False
                        break
                if valid:
                    gain = 1.0/(N-a) - 1.0/(N-b)
                    if gain > best_gain:
                        best_gain = gain
                        best_a = a
                        
            if best_a != -1:
                current_set[i] = best_a
                current_set_set.remove(b)
                current_set_set.add(best_a)
                improved = True
                break # Restart
        
    return current_set
