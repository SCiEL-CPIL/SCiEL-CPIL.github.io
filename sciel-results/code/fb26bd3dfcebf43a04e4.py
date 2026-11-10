# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # 1. Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build adjacency bitmasks
    # adj_mask[i] has bits set for all j such that (i-j) % m is in forbidden
    # Since the relation is symmetric (if d is square, -d is square for m=205),
    # adj_mask[i] includes j if (i-j)%m in F OR (j-i)%m in F.
    # Actually, if d in F, then -d in F is true for m=205?
    # Let's verify: 205 = 5 * 41.
    # Legendre symbol (-1/5) = 1, (-1/41) = 1. So -1 is a square mod 205.
    # Thus if d is a square, -d is a square.
    # So the graph is undirected and regular.
    
    adj_masks = [0] * m
    forb_list = list(forbidden)
    for i in range(m):
        mask = 0
        for d in forb_list:
            j1 = (i - d) % m
            j2 = (i + d) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        adj_masks[i] = mask

    def popcount(x):
        return bin(x).count('1')

    def is_valid(mask):
        # Check if mask represents an independent set
        # For every i in mask, (mask & adj_masks[i]) should be 0
        # Actually, if i and j are both in mask, bit j is set in adj_masks[i].
        # So we just need to check if for all i in mask, (mask & adj_masks[i]) == 0?
        # No, if i and j are in mask, adj_masks[i] has bit j. So mask & adj_masks[i] will have bit j.
        # So we check if (mask & adj_masks[i]) is 0 for all i in mask?
        # This is O(N^2) if done naively.
        # Better: iterate set bits.
        m_copy = mask
        while m_copy:
            lsb = m_copy & -m_copy
            i = lsb.bit_length() - 1
            if mask & adj_masks[i]:
                return False
            m_copy ^= lsb
        return True

    def greedy_init():
        """Construct an independent set using greedy algorithm with random order."""
        vertices = list(range(m))
        random.shuffle(vertices)
        mask = 0
        for v in vertices:
            if (mask & adj_masks[v]) == 0:
                mask |= (1 << v)
        return mask

    def local_search_sa(mask, time_limit_remaining):
        """
        Simulated Annealing local search.
        State: mask (independent set)
        Move: 1-1 swap (remove u, add v)
        """
        current_mask = mask
        current_size = popcount(current_mask)
        best_mask = current_mask
        best_size = current_size
        
        # Initial temperature
        T = 1.0
        cooling_rate = 0.995
        # We need to estimate steps per second to adjust T?
        # Or just run for a fixed number of iterations per restart?
        # Let's run until time runs out for this restart, or until convergence.
        
        start_time = time.time()
        
        # To speed up, we can precompute for each v, the neighbors.
        # But we need to check if v can be added after removing u.
        # Condition: 
        # 1. u is in current_mask
        # 2. v is NOT in current_mask
        # 3. (current_mask \ {u}) & adj_masks[v] == 0
        
        while time.time() - start_time < time_limit_remaining:
            improved = False
            
            # Try all 1-1 swaps
            # Iterate over u in current_mask
            u_iter = current_mask
            while u_iter:
                lsb_u = u_iter & -u_iter
                u = lsb_u.bit_length() - 1
                u_iter ^= lsb_u
                
                # Candidates for v: not in current_mask, and not adjacent to (current_mask \ {u})
                # Let M' = current_mask ^ (1 << u)
                # v is valid if (M' & adj_masks[v]) == 0 and v not in M'
                
                # We can iterate over all v not in current_mask
                # This is O(N^2) per step. 205^2 = 42025. 
                # If we do this for every u, it's O(N^3) = 8.6M ops per full pass.
                # This might be too slow in Python for many iterations.
                
                # Optimization: 
                # For a fixed u, the set of valid v is:
                # V_valid = { v | v not in current_mask, (current_mask ^ (1<<u)) & adj_masks[v] == 0 }
                
                # Let's just iterate v for each u.
                # To make it faster, we can break early if we find an improving move.
                
                for v in range(m):
                    if current_mask & (1 << v):
                        continue
                    # Check if v is compatible with current_mask minus u
                    # (current_mask ^ (1<<u)) & adj_masks[v] == 0
                    if ((current_mask ^ (1 << u)) & adj_masks[v]) == 0:
                        new_mask = (current_mask ^ (1 << u)) | (1 << v)
                        new_size = current_size # Size remains same in 1-1 swap
                        
                        # We only accept if size increases? No, 1-1 swap keeps size same.
                        # We need to accept non-improving moves to escape local optima.
                        # But how do we know if it's "better"? 
                        # We don't have a secondary objective. 
                        # Standard SA accepts with prob exp(-DeltaE/T).
                        # Here DeltaE = 0. So we accept with prob 1? 
                        # That would just be a random walk.
                        
                        # Actually, for MIS, 1-1 swaps don't change size.
                        # We need 2-1 swaps (remove 2, add 1? No, that decreases size) 
                        # or 1-2 swaps (remove 1, add 2? Increases size).
                        
                        # The current program's `local_search` tries to ADD a vertex.
                        # That is a 0-1 swap (add 1).
                        # If we can add a vertex, we should do it.
                        
                        # Let's re-evaluate the move set.
                        # Move 1: Add v (if valid). Size +1.
                        # Move 2: Remove u, Add v (1-1 swap). Size 0. Used to restructure.
                        # Move 3: Remove u, Remove w, Add v, Add z (2-2 swap). Size 0.
                        
                        # If we only do 0-1 (add) and 1-1 (swap), we can increase size.
                        
                        # Let's check if we can ADD any vertex to current_mask.
                        # If yes, we do it immediately (greedy improvement).
                        # If no, we do 1-1 swaps to try to create a space for a 0-1 add later.
                        
                        # This is a bit complex. 
                        # Alternative: Just use a simple heuristic.
                        # 1. Try to add any valid vertex.
                        # 2. If stuck, try all 1-1 swaps. Accept random ones to explore.
                        # 3. After some swaps, try to add again.
                        
                        pass # Placeholder for logic below

            # The loop above was just for structure. Let's implement a cleaner loop.
            break # Break out of while to implement cleaner logic

        return best_mask, best_size

    # Let's rewrite the search strategy more carefully.
    
    def search_round():
        """One round of greedy + local search."""
        mask = greedy_init()
        size = popcount(mask)
        
        # Phase 1: Greedy Add
        # Try to add any vertex that doesn't conflict.
        changed = True
        while changed:
            changed = False
            # Find a vertex v not in mask such that (mask & adj_masks[v]) == 0
            # We can iterate all v
            for v in range(m):
                if not (mask & (1 << v)):
                    if (mask & adj_masks[v]) == 0:
                        mask |= (1 << v)
                        size += 1
                        changed = True
                        break # Restart loop to check new state
            
        # Phase 2: Local Search (1-1 Swaps)
        # We want to escape local optima.
        # We perform a series of 1-1 swaps.
        # To avoid getting stuck in a cycle, we can use a tabu list or just random restarts.
        # Given the small size, we can afford to be aggressive.
        
        # Let's try a "Kick" strategy:
        # If we are stuck (no adds possible), we randomly remove k vertices (e.g., 2-5) 
        # and then re-greedy add. This is a form of perturbation.
        
        # However, a better approach for MIS is "Local Search with 2-opt" or similar.
        # But 1-1 swaps are the most common.
        
        # Let's implement a simple Simulated Annealing that accepts 1-1 swaps 
        # with a temperature, and also tries to add vertices (which are always accepted).
        
        T = 2.0
        cool = 0.99
        max_iter_per_sa = 5000 # Limit iterations to ensure we return
        
        for _ in range(max_iter_per_sa):
            # Try to add a vertex
            added = False
            # Pick a random vertex not in mask
            # Or iterate all? Iterating all is O(N).
            # Let's pick a random candidate to speed up
            candidates = [v for v in range(m) if not (mask & (1 << v))]
            if candidates:
                v = random.choice(candidates)
                if (mask & adj_masks[v]) == 0:
                    mask |= (1 << v)
                    size += 1
                    added = True
                    T = max(T * 1.1, 1.0) # Reheat slightly on improvement
            
            if not added:
                # Try a 1-1 swap
                # Pick random u in mask, random v not in mask
                u_list = [i for i in range(m) if (mask & (1 << i))]
                if u_list and candidates:
                    u = random.choice(u_list)
                    v = random.choice(candidates)
                    
                    # Check validity of swap: remove u, add v
                    # New mask = (mask ^ (1<<u)) | (1<<v)
                    # Must be independent.
                    # (mask ^ (1<<u)) must not conflict with v.
                    # i.e., ((mask ^ (1<<u)) & adj_masks[v]) == 0
                    if ((mask ^ (1 << u)) & adj_masks[v]) == 0:
                        new_mask = (mask ^ (1 << u)) | (1 << v)
                        # Size is same.
                        # Accept with probability 1 (random walk) or based on some heuristic?
                        # Since size doesn't change, we accept to explore.
                        # To avoid cycles, we can track recent states, but that's expensive.
                        # Just accept.
                        mask = new_mask
                        # size unchanged
                        T *= cool
                    else:
                        # Reject
                        T *= cool
                else:
                    break
            
            if T < 0.1:
                break
        
        return mask, size

    best_mask = 0
    best_size = 0
    
    start_time = time.time()
    time_limit = 590.0
    
    # Multiple restarts
    while time.time() - start_time < time_limit:
        mask, size = search_round()
        
        if size > best_size:
            best_size = size
            best_mask = mask
            
        # Optional: If we hit a theoretical upper bound, stop.
        # Upper bound for independent set in d-regular graph with N nodes:
        # Not a tight simple bound. 
        # But if best_size > 40, it's likely optimal or near optimal.
        
    result = []
    for i in range(m):
        if best_mask & (1 << i):
            result.append(i)
            
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
