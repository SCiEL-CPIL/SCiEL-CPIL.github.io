import numpy as np
import random
import time

def construct():
    """
    Constructs a set of 32 distinct integers in 1..512 maximizing the number
    of distinct differences d > 0 such that a, a+d, a+2d are in the set.
    Uses Greedy Construction followed by Local Search (Hill Climbing) with Restarts.
    """
    
    def vectorized_score(mask):
        """
        mask: boolean array of size 513 (indices 0..512).
        Returns the number of distinct differences d > 0 forming a 3-AP.
        """
        found_d = 0
        # d can range from 1 to 255 (since max value 512, min 1, 2d <= 511)
        for d in range(1, 256):
            max_a = 512 - 2 * d
            if max_a < 1:
                break
            # Check if any a in [1, max_a] satisfies mask[a] & mask[a+d] & mask[a+2*d]
            if np.any(mask[1:max_a+1] & mask[1+d:max_a+1+d] & mask[1+2*d:max_a+1+2*d]):
                found_d += 1
        return found_d

    def greedy_construct():
        """
        Builds a set by greedily adding the element that maximizes the score.
        """
        current_set = set()
        current_mask = np.zeros(513, dtype=bool)
        current_score = 0
        
        for step in range(32):
            best_candidate = -1
            best_score = -1
            
            # Try all candidates
            for x in range(1, 513):
                if x in current_set:
                    continue
                
                # Tentatively add x
                current_mask[x] = True
                new_score = vectorized_score(current_mask)
                current_mask[x] = False # Revert for next trial
                
                if new_score > best_score:
                    best_score = new_score
                    best_candidate = x
            
            # Add the best candidate
            if best_candidate != -1:
                current_set.add(best_candidate)
                current_mask[best_candidate] = True
                current_score = best_score
            else:
                # Fallback: pick random
                x = random.randint(1, 512)
                while x in current_set:
                    x = random.randint(1, 512)
                current_set.add(x)
                current_mask[x] = True
                current_score = vectorized_score(current_mask)
                
        return current_set, current_mask, current_score

    def local_search(initial_set, initial_mask, initial_score, max_trials=5000):
        """
        First-ascent hill climbing.
        """
        current_set = set(initial_set)
        current_mask = initial_mask.copy()
        current_score = initial_score
        
        improved = True
        trials = 0
        start_time = time.time()
        
        while improved and trials < max_trials:
            improved = False
            trials += 1
            if time.time() - start_time > 110.0:
                break
            
            # Pick a random element to remove
            to_remove = random.choice(list(current_set))
            # Pick a random element to add (not in set)
            to_add = random.randint(1, 512)
            while to_add in current_set:
                to_add = random.randint(1, 512)
                
            # Apply swap
            current_mask[to_remove] = False
            current_mask[to_add] = True
            
            new_score = vectorized_score(current_mask)
            
            if new_score > current_score:
                # Accept
                current_set.remove(to_remove)
                current_set.add(to_add)
                current_score = new_score
                improved = True
            else:
                # Revert
                current_mask[to_remove] = True
                current_mask[to_add] = False
                
        return current_set, current_mask, current_score

    random.seed(42)
    
    best_set = None
    best_score = -1
    
    # 1. Greedy Construction
    g_set, g_mask, g_score = greedy_construct()
    if g_score > best_score:
        best_score = g_score
        best_set = set(g_set)
        
    # Local search from Greedy
    ls_set, ls_mask, ls_score = local_search(g_set, g_mask, g_score, max_trials=5000)
    if ls_score > best_score:
        best_score = ls_score
        best_set = set(ls_set)
        
    # 2. Random Restarts with Local Search
    for i in range(5):
        r_set = set(random.sample(range(1, 513), 32))
        r_mask = np.zeros(513, dtype=bool)
        for x in r_set:
            r_mask[x] = True
        r_score = vectorized_score(r_mask)
        
        if r_score > best_score:
            best_score = r_score
            best_set = set(r_set)
            
        # Local search from Random
        ls_set, ls_mask, ls_score = local_search(r_set, r_mask, r_score, max_trials=3000)
        if ls_score > best_score:
            best_score = ls_score
            best_set = set(ls_set)
            
    return sorted(best_set)
