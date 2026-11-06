"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 85.0  # Safety margin before 90s kill
    
    n = 256
    max_sum = 2 * n
    
    # 1. Precompute squares
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
        
    # 2. Identify valid nodes (diagonal safe: 2*a not in squares)
    all_candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    M = len(all_candidates)
    
    # 3. Build Conflict Graph
    # conflicts[v] = set of nodes u such that v+u is a square
    conflicts = {a: set() for a in all_candidates}
    # To speed up, we can precompute for each a, which b in candidates conflict
    # But building the full graph is O(M^2). M ~ 245. 245^2 ~ 60k ops. Very fast.
    for idx, a in enumerate(all_candidates):
        for b in all_candidates[idx+1:]:
            if (a + b) in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)

    # Helper: Check if node v can be added to set s
    def try_add_fast(s, v):
        # If v conflicts with any node in s, it cannot be added
        # s is a set of nodes
        # conflicts[v] is a set of nodes
        return len(conflicts[v] & s) == 0

    def greedy_min_degree(candidates):
        remaining = set(candidates)
        independent_set = set()
        while remaining:
            min_deg = float('inf')
            chosen = None
            for v in remaining:
                deg = len(conflicts[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
            independent_set.add(chosen)
            remaining.remove(chosen)
            # Remove neighbors of chosen from remaining
            for neighbor in conflicts[chosen]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
        return independent_set

    def local_search_improve(s):
        s = set(s)
        improved = True
        # Limit iterations to prevent infinite loops in case of oscillation
        max_iters = 500
        iters = 0
        
        while improved and iters < max_iters:
            iters += 1
            improved = False
            
            # 1. Try to add any single node
            for v in all_candidates:
                if v not in s and try_add_fast(s, v):
                    s.add(v)
                    improved = True
            
            # 2. Try 1-swap: Remove u, add v.
            # Goal: Maintain size, but improve "slack" to allow future adds.
            # Or, if we can remove u and add v such that v has fewer conflicts with the rest, 
            # we might be able to add more later.
            # However, a simpler and more effective move is "Remove u, Add v" if it allows 
            # adding *another* node w later? That's a 2-step lookahead.
            
            # Let's try "Remove u, Add v" where v is not in s.
            # We accept if len(s) increases (impossible for 1-1) 
            # OR if the new configuration allows adding a 3rd node?
            # Simpler: Just try to find a swap that reduces the total degree of the set.
            
            # Actually, let's try "Remove u, Add v" if it doesn't decrease size, 
            # and the new node v has a lower "conflict count" with the remaining set than u did.
            # This is a heuristic for "better" packing.
            
            best_swap = None
            best_score = -1
            
            # Optimization: Only consider u that have high degree?
            # For simplicity, iterate all u in s
            for u in list(s):
                s.remove(u)
                
                # Find all v not in s that can be added
                addable = []
                for v in all_candidates:
                    if v not in s and v != u:
                        if try_add_fast(s, v):
                            addable.append(v)
                            
                if addable:
                    # Calculate "score" for each v. 
                    # Score = - (number of conflicts v has with current s) 
                    # Lower conflicts is better.
                    for v in addable:
                        # v is compatible with s
                        # How many conflicts does v have with s? 0 by definition of try_add_fast.
                        # So we need a different metric.
                        # Maybe: How many *other* nodes are now blocked by v?
                        # i.e., size of (conflicts[v] & (all_candidates - s - {u}))?
                        # If v blocks fewer nodes, it's better.
                        blocked = len(conflicts[v] & (set(all_candidates) - s - {u}))
                        # We want to minimize blocked.
                        score = -blocked
                        if score > best_score:
                            best_score = score
                            best_swap = (u, v)
                            
                s.add(u) # Restore u for next iteration of u loop
                
            if best_swap:
                u, v = best_swap
                s.remove(u)
                s.add(v)
                improved = True
                
            # 3. Try 2-swap: Remove u1, u2, Add v1, v2.
            # This is O(|S|^2 * |C|^2). Too slow for large S.
            # We can restrict this to a small number of candidates or skip if S is large.
            # Given the time limit, let's skip 2-swap in the main loop and rely on restarts.
            
        return s

    # Main Search Loop
    best_set = greedy_min_degree(all_candidates)
    best_size = len(best_set)
    
    random.seed(42)
    
    # Try multiple random restarts
    # We can also try different greedy strategies
    strategies = [
        lambda: greedy_min_degree(all_candidates),
        lambda: set(random.sample(all_candidates, min(50, len(all_candidates)))), # Random small set
    ]
    
    # Add a "weighted" greedy: pick node with min degree, but break ties randomly
    def greedy_min_degree_random(candidates):
        remaining = set(candidates)
        independent_set = set()
        while remaining:
            min_deg = float('inf')
            candidates_for_choice = []
            for v in remaining:
                deg = len(conflicts[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    candidates_for_choice = [v]
                elif deg == min_deg:
                    candidates_for_choice.append(v)
            chosen = random.choice(candidates_for_choice)
            independent_set.add(chosen)
            remaining.remove(chosen)
            for neighbor in conflicts[chosen]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
        return independent_set

    strategies.append(greedy_min_degree_random)
    
    for trial in range(500): # Max 500 trials
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Choose a strategy
        strat = random.choice(strategies)
        current_set = strat()
        
        # Refine
        current_set = local_search_improve(current_set)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
            # If we hit a high score, we can tighten the time limit for subsequent checks?
            # Or just continue.
            
    result = sorted(best_set)
    
    # Final Verification
    for i, a in enumerate(result):
        for b in result[i:]:
            s = a + b
            root = math.isqrt(s)
            if root * root == s:
                # This should never happen if our logic is correct
                # But if it does, we have a bug.
                # For safety, we could remove b, but that breaks the set.
                # Raise error to signal invalidity.
                raise ValueError(f"Conflict: {a} + {b} = {s}")
                
    return result
# EVOLVE-BLOCK-END
