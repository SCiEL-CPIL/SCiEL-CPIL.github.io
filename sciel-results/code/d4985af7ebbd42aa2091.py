import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def greedy_randomize(order_seed=0):
        """
        Greedy construction with randomized ordering.
        We shuffle the candidate pool or use a priority queue based on 'cost'.
        To speed up, we pre-compute nothing, but we use a fast check.
        """
        random.seed(order_seed)
        # Instead of 1..1000, we can try different orderings.
        # A good heuristic for B2 sets is to pick numbers that are 'far apart' 
        # or have low collision potential early on.
        
        # Simple strategy: Shuffle 1..1000 and greedily add if valid.
        # But pure shuffle is slow to converge to large sets.
        # Better: Use a priority queue where the key is the number of new collisions.
        # However, calculating collisions for all candidates is O(N^2) per step.
        
        # Let's try a hybrid: 
        # 1. Start with a small random seed set.
        # 2. Greedily add the next number from 1..1000 that minimizes new collisions,
        #    but with some randomization to escape local minima.
        
        chosen = []
        counts = defaultdict(int)
        
        # Pre-compute all possible sums? No, just track counts.
        
        # To make it faster, we can limit the search space or use a heuristic.
        # Let's try: Iterate through 1..1000 multiple times? No, that's just greedy.
        
        # Let's try a randomized greedy where at each step we pick from a window
        # of candidates with the lowest collision count, with some randomness.
        
        # Actually, the standard greedy (1 to N) is deterministic and yields ~sqrt(N) or similar.
        # For B2[2], the size is roughly N^(1/2) * constant? 
        # Asymptotic constant is at least 3/2? No, that's for Sidon sets (B2[1]).
        # For B2[2], the density is higher.
        
        # Let's implement a fast local search / hill climbing on the set.
        # 1. Start with a random valid set (e.g., greedy with random shuffle).
        # 2. Try to add a random number from 1..1000 not in the set.
        # 3. If invalid, try to swap or remove?
        
        # Simpler structural change: Multi-start greedy with different random seeds.
        # Each greedy run:
        #   - Shuffle 1..1000
        #   - Iterate through shuffled list
        #   - If adding x keeps max count <= 2, add it.
        
        # This is O(N^2) per run. N=1000, so 10^6 ops per run.
        # 145 seconds allows for many runs.
        
        best_size = 0
        best_set = []
        
        # Optimization: Use a list for counts? No, dict is fine.
        # To speed up the "check", we can maintain the counts dict.
        
        for seed in range(10000):
            if time.time() > deadline:
                break
            
            random.seed(seed)
            candidates = list(range(1, 1001))
            random.shuffle(candidates)
            
            chosen = []
            counts = defaultdict(int)
            
            for x in candidates:
                # Check if adding x is valid
                # New sums: x+y for y in chosen, and 2x
                # We need to check if any of these sums would exceed count 2.
                
                # Optimization: If x is large, it might create fewer collisions with small chosen numbers?
                # Actually, we just need to check the specific sums.
                
                # Fast check:
                # For each y in chosen:
                #   s = x + y
                #   if counts[s] == 2: invalid
                # Also s = 2x
                #   if counts[2x] == 2: invalid
                
                # This is O(|chosen|) per candidate.
                # Total O(N * |chosen|) ~ O(N^2/2).
                
                valid = True
                for y in chosen:
                    s = x + y
                    if counts[s] >= 2:
                        valid = False
                        break
                    s2 = 2 * x
                    if counts[s2] >= 2:
                        valid = False
                        break
                
                if valid:
                    # Add x
                    chosen.append(x)
                    for y in chosen[:-1]:
                        counts[x + y] += 1
                    counts[2 * x] += 1
            
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = chosen[:]
                
            # If we find a very large set, we might stop early?
            # Or we can use the best set as a starting point for local search?
            
        return best_set

    # The above is a multi-start randomized greedy.
    # Is there a better structural approach?
    # What if we use a greedy that prioritizes numbers with low "potential" for collisions?
    
    # Let's try a different heuristic:
    # Sort candidates by x % some_prime?
    # Or just use the random shuffle but with a twist:
    # Start with a random subset, then fill in.
    
    # Actually, the random shuffle greedy is a known simple baseline.
    # Let's enhance it: After the greedy phase, do a few steps of local search?
    # Local search:
    # 1. Pick a random element from the set, remove it.
    # 2. Try to add a random element from 1..1000 not in the set.
    # 3. If valid, keep the change.
    
    # This might escape local minima of the greedy.
    
    def local_search(initial_set, max_steps=1000):
        chosen = set(initial_set)
        counts = defaultdict(int)
        
        # Rebuild counts
        for i, x in enumerate(initial_set):
            for y in initial_set[i:]:
                counts[x+y] += 1
        
        # Pre-compute which numbers are not in the set
        available = [x for x in range(1, 1001) if x not in chosen]
        
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            
            # Try to add a random available number
            if not available:
                break
            
            idx = random.randint(0, len(available) - 1)
            x = available[idx]
            
            # Check if adding x is valid
            valid = True
            for y in chosen:
                s = x + y
                if counts[s] >= 2:
                    valid = False
                    break
            if counts[2*x] >= 2:
                valid = False
            
            if valid:
                # Add it
                chosen.add(x)
                for y in chosen:
                    if y != x:
                        counts[x+y] += 1
                counts[2*x] += 1
                available.pop(idx)
                
                # Maybe remove a random element to allow more?
                # No, we want to maximize size.
                # But sometimes removing a "bad" element allows adding two "good" ones.
                # Let's stick to simple addition first.
                
            else:
                # Try to swap?
                # Pick a random element from chosen, remove it, then try to add x.
                if chosen:
                    elem_to_remove = random.choice(list(chosen))
                    # Remove elem_to_remove
                    for y in chosen:
                        if y != elem_to_remove:
                            counts[elem_to_remove + y] -= 1
                    counts[2*elem_to_remove] -= 1
                    chosen.remove(elem_to_remove)
                    
                    # Now try to add x
                    valid_swap = True
                    for y in chosen:
                        s = x + y
                        if counts[s] >= 2:
                            valid_swap = False
                            break
                    if counts[2*x] >= 2:
                        valid_swap = False
                    
                    if valid_swap:
                        chosen.add(x)
                        for y in chosen:
                            if y != x:
                                counts[x+y] += 1
                        counts[2*x] += 1
                        available.append(elem_to_remove)
                        # Shuffle available? Or just leave it.
                        # We should remove x from available if it was there, but it wasn't.
                        # We added elem_to_remove to available.
                    else:
                        # Revert: Add back elem_to_remove
                        for y in chosen:
                            if y != elem_to_remove:
                                counts[elem_to_remove + y] += 1
                        counts[2*elem_to_remove] += 1
                        chosen.add(elem_to_remove)
                        # Remove elem_to_remove from available
                        available.remove(elem_to_remove)
        
        return list(chosen)

    # Strategy:
    # 1. Run several randomized greedy constructions.
    # 2. Take the best one.
    # 3. Run local search on it.
    
    best_set = []
    best_size = 0
    
    # Phase 1: Multi-start Greedy
    for seed in range(500): # Limit greedy runs to save time for local search
        if time.time() > deadline - 30: # Leave 30s for local search
            break
        
        random.seed(seed)
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        chosen = []
        counts = defaultdict(int)
        
        for x in candidates:
            valid = True
            for y in chosen:
                s = x + y
                if counts[s] >= 2:
                    valid = False
                    break
            if counts[2*x] >= 2:
                valid = False
            
            if valid:
                chosen.append(x)
                for y in chosen[:-1]:
                    counts[x + y] += 1
                counts[2 * x] += 1
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen[:]
            
    # Phase 2: Local Search on the best set
    if best_set:
        # Run local search multiple times with different random seeds
        for ls_seed in range(10):
            if time.time() > deadline - 5:
                break
            
            random.seed(1000 + ls_seed)
            improved_set = local_search(best_set, max_steps=5000)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set[:]
                
    return best_set
