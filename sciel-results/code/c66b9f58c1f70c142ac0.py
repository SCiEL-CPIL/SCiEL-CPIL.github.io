"""Bootstrap search program: Large graphs with no regular induced subgraph on six vertices."""
import time
import random
import itertools
import numpy as np

DEADLINE = 150
SAFETY_MARGIN = 15
START_TIME = None

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    deadline = START_TIME + DEADLINE - SAFETY_MARGIN
    best_n = 0
    best_edges = []
    best_valid = False
    rng = random.Random(12345)
    
    # Strategy 1: Try 1-factorization approach for n=22, 20, 18, 16, 14
    for n in [22, 20, 18, 16, 14]:
        if time.time() > deadline:
            break
            
        # For 1-factorization, n must be even
        if n % 2 != 0:
            continue
            
        max_attempts = 50 if n <= 18 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Generate a random 1-factorization of K_n
            # A 1-factorization has n-1 perfect matchings that partition the edges
            one_factors = random_one_factorization(n, rng)
            
            # Select a random subset of 1-factors to include
            # We need to be careful: we want to avoid regular induced subgraphs
            # Try different numbers of 1-factors
            num_factors = rng.randint(3, n - 3)
            selected_factors = rng.sample(range(n - 1), num_factors)
            
            # Build the graph from selected 1-factors
            adj_masks = [0] * n
            for f_idx in selected_factors:
                for (u, v) in one_factors[f_idx]:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    
            # Check for violations
            valid, violations = check_and_get_violations(adj_masks, n)
            
            if valid:
                if n > best_n:
                    best_n = n
                    best_edges = get_edges_from_masks(adj_masks, n)
                    best_valid = True
                break  # Found valid for this n
            
            # If not valid, try local search to fix violations
            if not valid and n >= 6:
                new_adj, new_valid, new_violations = local_search_one_factor(adj_masks, one_factors, n, rng, deadline)
                if new_valid:
                    if n > best_n:
                        best_n = n
                        best_edges = get_edges_from_masks(new_adj, n)
                        best_valid = True
                    break
    
    # Strategy 2: If 1-factorization didn't work well, try random graphs
    if not best_valid or best_n < 14:
        for n in range(min(22, best_n + 4), 5, -1):
            if time.time() > deadline:
                break
                
            max_attempts = 10 if n >= 18 else 20
            
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                    
                # Generate random graph
                p = rng.uniform(0.3, 0.7)
                adj_masks = [0] * n
                for u in range(n):
                    for v in range(u + 1, n):
                        if rng.random() < p:
                            adj_masks[u] |= (1 << v)
                            adj_masks[v] |= (1 << u)
                            
                # Check for violations
                valid, violations = check_and_get_violations(adj_masks, n)
                
                if valid:
                    if n > best_n:
                        best_n = n
                        best_edges = get_edges_from_masks(adj_masks, n)
                        best_valid = True
                    break
    
    if not best_valid:
        # Fallback: construct a simple non-regular graph on 6 vertices
        # Path P6: 0-1-2-3-4-5
        edges = [[0,1],[1,2],[2,3],[3,4],[4,5]]
        return {'n': 6, 'edges': edges}
        
    return {'n': best_n, 'edges': best_edges}

def random_one_factorization(n, rng):
    """
    Generate a random 1-factorization of K_n.
    Returns a list of n-1 perfect matchings.
    Each matching is a list of (u, v) pairs.
    """
    # Use the standard algorithm for 1-factorization of K_{2m}
    # Fix vertex 0, rotate the others
    
    vertices = list(range(1, n))
    rng.shuffle(vertices)
    
    one_factors = []
    for i in range(n - 1):
        # In round i, vertex 0 is matched with vertices[i]
        # The remaining n-2 vertices are paired up
        remaining = [v for j, v in enumerate(vertices) if j != i]
        remaining.insert(0, vertices[i])  # Put the "opposite" vertex first
        
        # Pair up: (0, remaining[0]), (remaining[1], remaining[2]), ...
        matching = [(0, remaining[0])]
        for j in range(1, len(remaining) - 1, 2):
            matching.append((remaining[j], remaining[j + 1]))
        
        one_factors.append(matching)
        
        # Rotate vertices for next round
        vertices = [vertices[0]] + vertices[1:]  # This is wrong, need proper rotation
        
    # Actually, let me use a cleaner approach
    # Standard 1-factorization: label vertices 0, 1, 2, ..., n-2, and ∞
    # In round i (0 to n-2), ∞ is matched with i
    # The others are paired: (i+1, i-1), (i+2, i-2), etc. (mod n-1)
    
    one_factors = []
    for i in range(n - 1):
        matching = [(0, i + 1)]  # 0 matched with i+1
        for j in range(1, (n - 1) // 2 + 1):
            a = (i + j) % (n - 1)
            b = (i - j) % (n - 1)
            matching.append((a + 1, b + 1))
        one_factors.append(matching)
    
    # Shuffle the order of matchings
    rng.shuffle(one_factors)
    
    return one_factors

def get_edges_from_masks(adj_masks, n):
    e = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                e.append([u, v])
    return e

def check_and_get_violations(adj_masks, n):
    """
    Returns (is_valid, list_of_violating_subsets)
    Only checks 6-subsets (larger regular subgraphs contain 6-subsets)
    """
    violations = []
    # Precompute masks for all 6-subsets
    subsets = list(itertools.combinations(range(n), 6))
    
    for combo in subsets:
        mask = 0
        for v in combo:
            mask |= (1 << v)
        
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        ok = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                ok = False
                break
        if ok:
            violations.append(combo)
            
    is_valid = len(violations) == 0
    return is_valid, violations

def local_search_one_factor(adj_masks, one_factors, n, rng, deadline):
    """
    Local search that only flips entire 1-factors (perfect matchings)
    This preserves the 1-factorization structure.
    """
    cur = adj_masks[:]
    
    # Track which 1-factors are currently included
    factor_included = [False] * len(one_factors)
    
    # Determine which factors are currently in the graph
    for f_idx, matching in enumerate(one_factors):
        # Check if all edges of this matching are in the graph
        all_present = True
        for (u, v) in matching:
            if not (cur[u] & (1 << v)):
                all_present = False
                break
        if all_present:
            factor_included[f_idx] = True
    
    # Precompute all 6-subsets
    all_subsets = list(itertools.combinations(range(n), 6))
    
    # Get initial violations
    valid, violations = check_and_get_violations(cur, n)
    if valid:
        return cur, True, []
        
    current_violation_count = len(violations)
    
    max_iters = 200
    it = 0
    while it < max_iters:
        if time.time() > deadline:
            break
            
        if current_violation_count == 0:
            return cur, True, []
            
        # Try flipping a random 1-factor (add or remove it)
        f_idx = rng.randint(0, len(one_factors) - 1)
        
        # Store state
        old_masks = cur[:]
        
        # Flip the 1-factor
        for (u, v) in one_factors[f_idx]:
            cur[u] ^= (1 << v)
            cur[v] ^= (1 << u)
        
        # Check new violations
        new_valid, new_violations = check_and_get_violations(cur, n)
        new_count = len(new_violations)
        
        if new_count < current_violation_count:
            # Improvement, keep the flip
            current_violation_count = new_count
            factor_included[f_idx] = not factor_included[f_idx]
        else:
            # Revert
            cur = old_masks
        
        it += 1
    
    valid, violations = check_and_get_violations(cur, n)
    return cur, valid, violations
