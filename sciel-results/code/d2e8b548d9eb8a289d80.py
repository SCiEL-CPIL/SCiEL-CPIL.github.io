"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    def _check_validity(selected):
        """Check if no element divides the sum of any 3 distinct other selected elements."""
        n = len(selected)
        if n < 4:
            return True
        sel_set = set(selected)
        for i, a in enumerate(selected):
            others = [x for x in selected if x != a]
            # We need to check all triples from others
            # Optimization: if a is large, fewer triples will be divisible by it
            # But we still need to check all
            for t in combinations(others, 3):
                s = t[0] + t[1] + t[2]
                if s % a == 0:
                    return False
        return True

    def _check_validity_fast(selected, candidate):
        """Check if adding candidate to selected maintains validity.
        Only need to check:
        1. candidate does not divide sum of any 3 in selected
        2. no element in selected divides sum of candidate + 2 others from selected
        """
        sel_set = set(selected)
        
        # Check 1: candidate divides sum of 3 from selected
        for t in combinations(selected, 3):
            if (t[0] + t[1] + t[2]) % candidate == 0:
                return False
        
        # Check 2: for each a in selected, does a divide candidate + x + y for any x,y in selected, x!=y, x!=a, y!=a
        # We can precompute pairs sums or just iterate
        for a in selected:
            for i in range(len(selected)):
                x = selected[i]
                if x == a:
                    continue
                for j in range(i + 1, len(selected)):
                    y = selected[j]
                    if y == a:
                        continue
                    if x == y:
                        continue
                    s = candidate + x + y
                    if s % a == 0:
                        return False
        return True

    def _greedy_construct(order, deadline):
        """Greedy construction with fast incremental check."""
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if _check_validity_fast(selected, x):
                selected.append(x)
        return selected

    def _local_search(selected, deadline):
        """Improve set by trying to add elements or swap."""
        best = list(selected)
        best_set = set(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any element
            for x in range(192, 0, -1):
                if time.time() > deadline:
                    break
                if x in best_set:
                    continue
                if _check_validity_fast(best, x):
                    best.append(x)
                    best_set.add(x)
                    improved = True
                    break

            if time.time() > deadline:
                break

            # Try swapping: remove one element, try to add two?
            # Simple perturbation: remove a random element, then try to add greedily
            if len(best) > 0 and random.random() < 0.3:
                idx = random.randint(0, len(best) - 1)
                temp_best = [e for i, e in enumerate(best) if i != idx]
                temp_set = set(temp_best)
                
                # Try to add elements to temp_best
                for x in range(192, 0, -1):
                    if time.time() > deadline:
                        break
                    if x in temp_set:
                        continue
                    if _check_validity_fast(temp_best, x):
                        temp_best.append(x)
                        temp_set.add(x)
                
                if len(temp_best) > len(best):
                    best = temp_best
                    best_set = temp_set
                    improved = True
                    break

        return best

    # Initial greedy from high to low
    order = list(range(192, 0, -1))
    best = _greedy_construct(order, deadline)
    best_size = len(best)

    # Randomized restarts with different orderings
    random.seed(42)
    restart_count = 0
    while time.time() < deadline - 10.0:
        restart_count += 1
        order = list(range(192, 0, -1))
        random.shuffle(order)
        candidate = _greedy_construct(order, deadline)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
            # If we improved, do local search on this
            if time.time() < deadline - 5.0:
                improved = _local_search(best, deadline)
                if len(improved) > best_size:
                    best = improved
                    best_size = len(best)
    
    # Final local search on the best found
    if time.time() < deadline:
        improved = _local_search(best, deadline)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)

    if len(best) == 0:
        best = [192]

    return best
# EVOLVE-BLOCK-END
