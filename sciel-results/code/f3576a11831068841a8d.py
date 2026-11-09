import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses Bitmask-based Iterated Local Search (ILS) to maximize the size of the independent set
    in the conflict graph defined by forbidden differences.
    """
    m = 205
    k = 2
    
    # 1. Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # 2. Build adjacency bitmask structure
    # neighbors_mask[i] is a bitmask where bit j is 1 if i and j are connected (conflict)
    neighbors_mask = [0] * m
    for u in range(m):
        mask = 0
        for v in range(u + 1, m):
            d1 = (u - v) % m
            d2 = (v - u) % m
            if d1 in forbidden or d2 in forbidden:
                mask |= (1 << v)
                neighbors_mask[v] |= (1 << u)
        neighbors_mask[u] = mask
    
    # Helper to check if a candidate can be added to a set represented by a bitmask
    # current_mask: bitmask of current set
    # candidate: int
    def can_add(candidate, current_mask):
        # If candidate is already in set, return False (though logic prevents this usually)
        if current_mask & (1 << candidate):
            return False
        # Check if any neighbor of candidate is in current set
        # neighbors_mask[candidate] & current_mask != 0 means there's a conflict
        if neighbors_mask[candidate] & current_mask:
            return False
        return True

    def randomized_greedy():
        """
        Construct a maximal independent set using a randomized greedy approach.
        Returns the set as a bitmask.
        """
        current_mask = 0
        candidates = list(range(m))
        random.shuffle(candidates)
        for v in candidates:
            if can_add(v, current_mask):
                current_mask |= (1 << v)
        return current_mask

    def repair(current_mask):
        """
        Greedily fill the set again to reach a local optimum.
        We iterate through all vertices in a random order and add them if valid.
        """
        candidates = list(range(m))
        random.shuffle(candidates)
        
        # We can optimize this by only checking vertices not already in the mask,
        # but since m=205, checking all is fast enough and simpler.
        for v in candidates:
            if not (current_mask & (1 << v)):
                if can_add(v, current_mask):
                    current_mask |= (1 << v)
        return current_mask

    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    # Initialize with randomized greedy
    current_mask = randomized_greedy()
    best_mask = current_mask
    best_size = bin(best_mask).count('1')
    
    # ILS loop
    while time.time() < deadline:
        # 1. Perturbation: Remove a random vertex from current set
        if best_size > 0:
            # Get list of set bits
            set_bits = [i for i in range(m) if (current_mask >> i) & 1]
            if set_bits:
                to_remove = random.choice(set_bits)
                current_mask &= ~(1 << to_remove)
                
                # 2. Repair: Greedily fill again
                # We use a randomized order to explore different local optima
                current_mask = repair(current_mask)
                
                current_size = bin(current_mask).count('1')
                
                # 3. Acceptance: 
                # Simple ILS: If new size > best size, update best and restart from here.
                # If new size == best size, we might keep it or restart from best.
                # If new size < best size, we typically reject and restart from best 
                # (or accept with probability, but for MaxIS, greedy repair usually 
                # yields a local optimum that is either better or same size, rarely worse 
                # if the removal didn't break too many things, but it can be worse).
                
                # To escape local optima, we should sometimes accept worse solutions 
                # or perturb more strongly. 
                # Let's use a simple strategy: 
                # - If improved, update best, current = best.
                # - If equal, current = new (to vary the trajectory).
                # - If worse, current = best (revert).
                
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                    current_mask = best_mask
                elif current_size == best_size:
                    current_mask = current_mask
                else:
                    # Revert to best to avoid drifting into bad regions
                    current_mask = best_mask

    # Convert best_mask to numpy array
    result_list = [i for i in range(m) if (best_mask >> i) & 1]
    result = np.array(result_list, dtype=np.int64)
    return result

# EVOLVE-BLOCK-START
# (The code above is the entire block.)
# EVOLVE-BLOCK-END
