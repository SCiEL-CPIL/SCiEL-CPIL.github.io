"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 78.0  # safety margin before 90s kill
    N = 4096
    MAX_SIZE = 80
    
    def check_addition(selected, triple_sums, x):
        """Check if x can be added to selected maintaining triple-sum uniqueness.
        Returns True if valid, and updates triple_sums in place if valid."""
        # We need to check all triples involving x
        # Types: (x,x,x), (x,x,y), (x,y,y), (x,y,z) where y,z in selected
        # For efficiency, we check against existing triple_sums
        
        n = len(selected)
        # (x,x,x)
        s = 3 * x
        if s in triple_sums:
            return False
        new_sums = {s}
        
        # (x,x,y) for each y in selected
        for y in selected:
            s = 2 * x + y
            if s in triple_sums or s in new_sums:
                return False
            new_sums.add(s)
        
        # (x,y,y) for each y in selected
        for y in selected:
            s = x + 2 * y
            if s in triple_sums or s in new_sums:
                return False
            new_sums.add(s)
        
        # (x,y,z) for y <= z in selected
        for i in range(n):
            y = selected[i]
            for j in range(i, n):
                z = selected[j]
                s = x + y + z
                if s in triple_sums or s in new_sums:
                    return False
                new_sums.add(s)
        
        # All valid, add them
        triple_sums.update(new_sums)
        return True
    
    def build_greedy(start_from=1, max_size=MAX_SIZE, order=None):
        """Greedy construction with optional custom ordering."""
        selected = []
        triple_sums = set()
        
        if order is None:
            candidates = list(range(start_from, N + 1))
        else:
            candidates = order
        
        for x in candidates:
            if len(selected) >= max_size:
                break
            if time.time() > DEADLINE - 5:
                break
            if x > N:
                continue
            if x in selected:
                continue
            
            if check_addition(selected, triple_sums, x):
                selected.append(x)
        
        return selected
    
    def local_swap_search(selected, triple_sums, max_iterations=50000):
        """Try to improve the set by swapping elements."""
        selected = sorted(selected)
        # Rebuild triple_sums for safety
        triple_sums = set()
        n = len(selected)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    triple_sums.add(selected[i] + selected[j] + selected[k])
        
        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            if time.time() > DEADLINE - 5:
                break
            
            # Try removing one element and adding a better one
            for idx in range(len(selected)):
                if time.time() > DEADLINE - 5:
                    break
                old = selected[idx]
                # Try adding a new element that's larger (to pack more)
                # or just try random candidates
                # First try: can we replace old with something else?
                # Remove old
                trial_selected = selected[:idx] + selected[idx+1:]
                trial_sums = set()
                tn = len(trial_selected)
                for i in range(tn):
                    for j in range(i, tn):
                        for k in range(j, tn):
                            trial_sums.add(trial_selected[i] + trial_selected[j] + trial_selected[k])
                
                # Try to add a new element
                # Search for a candidate that fits
                found = False
                # Try a range of candidates
                search_range = range(1, N + 1)
                for new_x in search_range:
                    if new_x in trial_selected:
                        continue
                    if check_addition(trial_selected, trial_sums, new_x):
                        # Accept the swap
                        trial_selected.append(new_x)
                        selected = sorted(trial_selected)
                        # Rebuild triple_sums
                        triple_sums = set()
                        n2 = len(selected)
                        for i in range(n2):
                            for j in range(i, n2):
                                for k in range(j, n2):
                                    triple_sums.add(selected[i] + selected[j] + selected[k])
                        improved = True
                        found = True
                        break
                
                if found:
                    break
        
        return selected, triple_sums
    
    def random_reorder_greedy(start_from=1, max_size=MAX_SIZE, seed=0):
        """Greedy with randomized ordering."""
        rng = random.Random(seed)
        candidates = list(range(start_from, N + 1))
        # Weight toward smaller numbers but allow some randomness
        # Use a mix: mostly sequential with some random shuffling of windows
        selected = []
        triple_sums = set()
        
        # First pass: greedy from start
        for x in range(start_from, N + 1):
            if len(selected) >= max_size:
                break
            if time.time() > DEADLINE - 5:
                break
            if check_addition(selected, triple_sums, x):
                selected.append(x)
        
        return selected
    
    # Strategy 1: Standard greedy
    best_set = build_greedy(start_from=1, max_size=MAX_SIZE)
    best_len = len(best_set)
    
    if time.time() > DEADLINE - 10:
        return best_set
    
    # Strategy 2: Try different starting points
    for start in [1, 2, 3, 5, 10]:
        if time.time() > DEADLINE - 10:
            break
        s = build_greedy(start_from=start, max_size=MAX_SIZE)
        if len(s) > best_len:
            best_set = s
            best_len = len(s)
    
    # Strategy 3: Local swap search to improve
    if best_len < MAX_SIZE and time.time() < DEADLINE - 15:
        # Rebuild triple_sums for best_set
        triple_sums = set()
        n = len(best_set)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    triple_sums.add(best_set[i] + best_set[j] + best_set[k])
        
        improved_set, improved_sums = local_swap_search(best_set, triple_sums, max_iterations=10000)
        if len(improved_set) > best_len:
            best_set = improved_set
            best_len = len(best_set)
    
    # Strategy 4: Multiple random restarts with different orderings
    if best_len < MAX_SIZE:
        for seed in range(50):
            if time.time() > DEADLINE - 5:
                break
            rng = random.Random(seed)
            # Create a randomized priority: smaller numbers have higher priority but with noise
            candidates = list(range(1, N + 1))
            # Shuffle but keep small numbers somewhat preferred
            # Use exponential weighting
            weights = [1.0 / (x + 1) for x in candidates]
            # Partial shuffle: swap random pairs
            for _ in range(100):
                i, j = rng.sample(range(len(candidates)), 2)
                candidates[i], candidates[j] = candidates[j], candidates[i]
            
            s = build_greedy(start_from=1, max_size=MAX_SIZE, order=candidates)
            if len(s) > best_len:
                best_set = s
                best_len = len(s)
    
    return best_set
# EVOLVE-BLOCK-END
