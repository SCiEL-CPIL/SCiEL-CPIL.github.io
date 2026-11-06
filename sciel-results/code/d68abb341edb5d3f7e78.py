import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a B2[2] set in {1, ..., 1000} maximizing size.
    Uses a randomized greedy approach with multiple restarts and local search.
    """
    deadline = time.time() + 145  # Safety margin before 150s kill
    best_set = []
    best_size = 0

    def is_valid_addition(chosen, counts, x):
        """Check if adding x to chosen keeps the B2[2] property."""
        for y in chosen:
            if y < x:
                s = x + y
                if counts.get(s, 0) >= 2:
                    return False
        s2 = 2 * x
        if counts.get(s2, 0) >= 2:
            return False
        return True

    def add_element(chosen, counts, x):
        """Add x to chosen and update counts."""
        chosen.append(x)
        for y in chosen:
            if y < x:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
        s2 = 2 * x
        counts[s2] = counts.get(s2, 0) + 1

    def greedy_randomized(max_val, seed):
        """Randomized greedy construction."""
        random.seed(seed)
        chosen = []
        counts = {}
        
        # Randomly shuffle the range
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_addition(chosen, counts, x):
                add_element(chosen, counts, x)
        
        return chosen

    def greedy_forward(max_val, seed):
        """Forward greedy with some randomness in tie-breaking."""
        random.seed(seed)
        chosen = []
        counts = {}
        
        for x in range(1, max_val + 1):
            if is_valid_addition(chosen, counts, x):
                add_element(chosen, counts, x)
        
        return chosen

    def local_search(initial_set, max_val, iterations=100):
        """Try to improve the set via swaps."""
        chosen = sorted(initial_set)
        counts = {}
        for x in chosen:
            for y in chosen:
                if x <= y:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
        
        improved = True
        iters = 0
        while improved and iters < iterations:
            improved = False
            iters += 1
            for i, x in enumerate(chosen):
                # Try removing x and adding some other value
                for y in range(1, max_val + 1):
                    if y in chosen:
                        continue
                    # Temporarily remove x
                    temp_counts = dict(counts)
                    for z in chosen:
                        if z == x:
                            continue
                        s = x + z if z > x else z + x
                        if s in temp_counts:
                            temp_counts[s] -= 1
                            if temp_counts[s] == 0:
                                del temp_counts[s]
                        s2 = 2 * x
                        if s2 in temp_counts:
                            temp_counts[s2] -= 1
                            if temp_counts[s2] == 0:
                                del temp_counts[s2]
                    
                    # Check if we can add y
                    can_add = True
                    for z in chosen:
                        if z == x:
                            continue
                        s = y + z
                        if temp_counts.get(s, 0) >= 2:
                            can_add = False
                            break
                    if can_add and 2 * y in temp_counts and temp_counts[2 * y] >= 2:
                        can_add = False
                    
                    if can_add:
                        # Perform the swap
                        new_chosen = [z for z in chosen if z != x] + [y]
                        new_counts = {}
                        for a in new_chosen:
                            for b in new_chosen:
                                if a <= b:
                                    s = a + b
                                    new_counts[s] = new_counts.get(s, 0) + 1
                        
                        # Verify validity
                        valid = all(v <= 2 for v in new_counts.values())
                        if valid:
                            chosen = new_chosen
                            counts = new_counts
                            improved = True
                            break
        
        return chosen

    # Run multiple strategies
    # Strategy 1: Standard forward greedy (deterministic baseline)
    baseline = greedy_forward(1000, 0)
    best_size = len(baseline)
    best_set = baseline

    # Strategy 2: Multiple randomized greedy restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        candidate = greedy_randomized(1000, i)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best_set = candidate

    # Strategy 3: Local search on the best candidate
    if time.time() > deadline - 30:
        return best_set
    
    improved_set = local_search(best_set, 1000, iterations=20)
    if len(improved_set) > best_size:
        best_size = len(improved_set)
        best_set = improved_set

    # Strategy 4: More local search iterations if time permits
    if time.time() > deadline - 10:
        return best_set
    
    further_improved = local_search(best_set, 1000, iterations=50)
    if len(further_improved) > best_size:
        best_set = further_improved
        best_size = len(further_improved)

    return best_set

# EVOLVE-BLOCK-END
