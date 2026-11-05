import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _valid_row(row):
    """Check if a row has distinct internal differences."""
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d = row[j] - ri
            if d in diffs:
                return False
            diffs.append(d)
    return True

def _row_to_mask(row):
    """Convert a valid row to a bitmask of its differences."""
    n = len(row)
    mask = 0
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d = row[j] - ri
            mask |= (1 << d)
    return mask

def _generate_candidates(k, scope, rng):
    """
    Generate a diverse set of candidate rows of size k+1, starting at 0,
    strictly increasing, max <= scope, with distinct internal diffs.
    """
    num_elements = k + 1
    candidates = []
    seen_masks = set()
    
    # Limit candidates to keep memory and time manageable
    # For k=5, num_elements=6. 
    # We need enough candidates to have a high probability of finding a clique.
    # If we aim for scope ~110, and we need 7 rows.
    # Let's generate a few thousand candidates.
    
    max_candidates = 20000
    attempts = 0
    max_attempts = max_candidates * 10
    
    while len(candidates) < max_candidates and attempts < max_attempts:
        attempts += 1
        # Randomly pick k values from 1..scope
        if scope < k:
            break
        vals = rng.sample(range(1, scope + 1), k)
        vals.sort()
        row = [0] + vals
        
        if not _valid_row(row):
            continue
            
        mask = _row_to_mask(row)
        if mask in seen_masks:
            continue
            
        seen_masks.add(mask)
        candidates.append((row, mask))
        
    return candidates

def _find_clique(candidates, n_rows, time_limit, rng):
    """
    Find a clique of size n_rows in the compatibility graph of candidates.
    Nodes: candidates.
    Edge: (i, j) if mask_i & mask_j == 0.
    """
    n = len(candidates)
    if n < n_rows:
        return None
        
    # Build adjacency bitmasks
    # adj[i] is a bitmask of candidates j (j > i) that are compatible with i
    # Using Python ints for bitsets (arbitrary precision)
    adj = [0] * n
    
    for i in range(n):
        mask_i = candidates[i][1]
        adj_i = 0
        for j in range(i + 1, n):
            mask_j = candidates[j][1]
            if (mask_i & mask_j) == 0:
                adj_i |= (1 << j)
        adj[i] = adj_i
        
    # Heuristic: Sort candidates by degree (number of compatible neighbors)
    # Higher degree first is often better for clique finding (Tomita heuristic)
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    # We want to search in an order that maximizes the chance of finding a clique quickly
    # But we also need to preserve the indices for the adjacency bitmasks?
    # If we reorder nodes, we must rebuild adj.
    # Let's reorder.
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    new_candidates = [candidates[i] for i in order]
    new_adj = [0] * n
    
    # Map old index to new index
    old_to_new = [0] * n
    for new_idx, old_idx in enumerate(order):
        old_to_new[old_idx] = new_idx
        
    for i in range(n):
        old_i = order[i]
        adj_old = adj[old_i]
        adj_new = 0
        # Iterate over bits in adj_old
        # Only keep bits for nodes that come after i in the new order
        # Actually, standard Tomita algorithm uses a list of candidates and filters.
        # Let's just rebuild adj for the new order.
        for j in range(n):
            if i == j: continue
            if (adj_old >> order[j]) & 1:
                # old_i is compatible with order[j]
                adj_new |= (1 << j)
        new_adj[i] = adj_new
        
    candidates = new_candidates
    adj = new_adj
    
    best_clique = []
    
    # Recursive backtracking
    # P: list of candidate indices available
    # clique: current list of indices in the clique
    def backtrack(P, clique, depth):
        nonlocal best_clique
        if len(clique) == n_rows:
            best_clique = clique[:]
            return True
            
        if not P:
            return False
            
        # Branch and bound: if len(clique) + len(P) <= len(best_clique), prune
        # But we want to find ANY clique of size n_rows, so we don't have a "best" yet in terms of size.
        # We just need to find one.
        
        # Heuristic: pick a node from P with max degree in the subgraph induced by P
        # For speed, just pick the first one? Or random?
        # Let's pick the one with max degree in P to reduce branching factor.
        
        # Calculate degrees in P
        # This is O(|P|^2) which might be slow if |P| is large.
        # But |P| shrinks quickly.
        
        # Optimization: if time is running out, return False
        if time.time() > time_limit:
            return False
            
        # Select pivot
        # Simple heuristic: pick node with highest degree in P
        max_deg = -1
        pivot = P[0]
        for u in P:
            deg = 0
            # Count neighbors in P
            # adj[u] has bits for all compatible nodes. We need to count bits that are in P.
            # Creating a mask for P is expensive.
            # Alternative: just pick P[0]. It's a standard simple heuristic.
            # Or random.
            pass
            
        # Let's just pick P[0] for simplicity and speed in the inner loop.
        # The ordering of P matters.
        # We initialized P as all nodes.
        # Let's sort P by degree initially?
        # In the recursive step, we filter P.
        
        u = P[0]
        
        # New P: nodes in P[1:] that are compatible with u
        mask_u = adj[u]
        new_P = []
        for v in P[1:]:
            if (mask_u >> v) & 1:
                new_P.append(v)
                
        # Recurse
        if backtrack(new_P, clique + [u], depth + 1):
            return True
            
        # If we didn't find a clique with u, try next node
        # But we must remove u from P for subsequent tries?
        # Standard algorithm:
        # while P:
        #   u = P.pop()
        #   if compatible(u, clique):
        #      recurse(P, clique+u)
        
        # My implementation above picks P[0] and removes it implicitly by slicing?
        # No, `new_P` is derived from `P[1:]`. So `u` is effectively removed.
        # But if the recursion fails, we need to try the next node in the ORIGINAL P.
        # The structure `u = P[0]; recurse(P[1:])` implies we are trying u first.
        # If it fails, we need to try P[1] with the same clique?
        # No, the standard algorithm is:
        #   while P:
        #      u = P.pop(0) # or pop()
        #      if compatible:
        #         if recurse(P, clique+u): return True
        #      # else or after fail, u is removed from P, so next iteration tries next node
        
        # So we need a loop.
        
    # Let's rewrite backtrack properly with a loop
    
    def backtrack_loop(P, clique):
        nonlocal best_clique
        if len(clique) == n_rows:
            best_clique = clique[:]
            return True
        if not P:
            return False
        if time.time() > time_limit:
            return False
            
        # Copy P to allow mutation (pop)
        # P is a list of indices
        while P:
            # Heuristic: pick node with max degree in current P
            # For speed, just pop the first one?
            # Better: pop the one with highest degree in the subgraph induced by P
            # But calculating degree is expensive.
            # Let's stick to the order provided.
            u = P.pop(0) # O(N)
            
            # Check compatibility with all nodes in clique?
            # We only passed compatible nodes in P, so u is compatible with the LAST node added?
            # No, P contains nodes compatible with the LAST node added in the previous step?
            # Let's trace:
            # Initial P: all nodes.
            # Step 1: pick u1. New P: nodes compatible with u1.
            # Step 2: pick u2 from New P. u2 is compatible with u1.
            # Step 3: pick u3 from NewNew P. u3 is compatible with u2.
            # Is u3 compatible with u1?
            # If u3 is in NewNew P, it was in New P (compatible with u1) AND compatible with u2.
            # So yes, all nodes in P are compatible with ALL nodes in the current clique.
            # This is the key property of this recursive formulation.
            
            # So we don't need to check compatibility with the whole clique, just that u is in P.
            
            if backtrack_loop(P, clique + [u]):
                return True
                
        return False

    # Initial P: all nodes
    # To improve performance, sort initial P by degree (descending)
    # We already sorted candidates by degree globally.
    # So P = [0, 1, ..., n-1] is sorted by degree.
    
    P = list(range(n))
    
    # To avoid modifying the original list if we restart or something, we pass a copy?
    # backtrack_loop pops from P.
    # If it fails completely, P is empty.
    
    found = backtrack_loop(P, [])
    
    if found:
        return [candidates[i][0] for i in best_clique]
    return None

def construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 145.0  # 5s safety margin
    
    rng = random.Random(42)
    
    # Estimate scope
    # Counting bound: M >= n_rows * k(k+1)/2
    num_pairs = k * (k + 1) // 2
    counting_bound = n_rows * num_pairs
    
    # Best known for (7,5) is 112.
    # Let's try scopes from counting_bound upwards.
    # Start with a margin.
    current_scope = counting_bound + 10
    
    best_solution = None
    
    # We will try to find a solution for increasing scopes.
    # If we find one, we record it.
    # We can stop if we find one at the counting bound? No, we want the MINIMUM scope.
    # So we MUST try counting_bound first.
    
    # However, generating candidates for small scopes might yield too few candidates.
    # If we can't find a clique, we increase scope.
    
    min_scope_to_try = counting_bound
    max_scope_to_try = counting_bound + 50 # Don't go too high
    
    while current_scope <= max_scope_to_try and time.time() < DEADLINE - 5:
        # Generate candidates
        # Time limit for generation + search
        gen_time_limit = time.time() + 20.0
        search_time_limit = DEADLINE - 2.0
        
        candidates = _generate_candidates(k, current_scope, rng)
        
        if len(candidates) < n_rows:
            current_scope += 1
            continue
            
        # Find clique
        solution = _find_clique(candidates, n_rows, search_time_limit, rng)
        
        if solution:
            best_solution = solution
            # If we found a solution, is it optimal?
            # We started from counting_bound.
            # If current_scope == counting_bound, it's optimal (or close to it).
            # If we found it at a higher scope, we should try lower scopes?
            # But we are iterating upwards.
            # So the FIRST solution found is the one with the smallest scope in our search range.
            # Since we start at counting_bound, if we find one, it's the best we can do with this method.
            # We can break.
            break
            
        current_scope += 1
        
    if best_solution is None:
        # Fallback: Greedy construction
        # This might produce a larger scope, but guarantees a valid output.
        rows = []
        used_mask = 0
        for _ in range(n_rows):
            found = False
            # Try to find a row
            # Use a large scope for fallback
            fb_scope = counting_bound + 100
            for _ in range(1000):
                if fb_scope < k:
                    break
                vals = rng.sample(range(1, fb_scope + 1), k)
                vals.sort()
                row = [0] + vals
                if not _valid_row(row):
                    continue
                mask = _row_to_mask(row)
                if mask & used_mask:
                    continue
                rows.append(row)
                used_mask |= mask
                found = True
                break
            if not found:
                # Sequential fallback
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & set(bin(used_mask).replace('0b','').split('1') if used_mask else []):
                         # This check is broken. Let's use a set of used diffs.
                         pass
                # Proper sequential fallback
                used_diffs = set()
                for d in range(1, fb_scope+1):
                    if used_mask & (1 << d):
                        used_diffs.add(d)
                
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs |= new_diffs
                rows.append(row)
        best_solution = rows

    # Final validation
    # Check if best_solution is valid
    all_diffs = []
    for row in best_solution:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                all_diffs.append(row[j] - row[i])
                
    if len(all_diffs) != len(set(all_diffs)):
        # Invalid, force fallback
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs |= new_diffs
            rows.append(row)
        return rows
        
    return best_solution

# EVOLVE-BLOCK-END
