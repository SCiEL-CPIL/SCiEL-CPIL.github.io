"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _check_subset_sum_avoidance(selected):
    """Check if no reciprocal equals sum of reciprocals of a subset of others."""
    n = len(selected)
    if n <= 1:
        return True
    recips = [1.0 / x for x in selected]
    for i in range(n):
        target = recips[i]
        others = [recips[j] for j in range(n) if j != i]
        m = len(others)
        # Use meet-in-the-middle for larger sets, brute force for small
        if m <= 20:
            # Brute force all subsets
            for mask in range(1, 1 << m):
                s = 0.0
                for bit in range(m):
                    if mask & (1 << bit):
                        s += others[bit]
                if abs(s - target) < 1e-9:
                    return False
        else:
            # Meet in the middle
            half = m // 2
            left = others[:half]
            right = others[half:]
            left_sums = set()
            for mask in range(1 << len(left)):
                s = 0.0
                for bit in range(len(left)):
                    if mask & (1 << bit):
                        s += left[bit]
                left_sums.add(round(s, 12))
            for mask in range(1 << len(right)):
                s = 0.0
                for bit in range(len(right)):
                    if mask & (1 << bit):
                        s += right[bit]
                needed = target - s
                if round(needed, 12) in left_sums:
                    # Check it's not the empty subset on both sides (sum=0)
                    if mask != 0 or needed != 0.0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # Start with seed
    best_set = list(range(13, 25))
    best_size = len(best_set)
    
    # Greedy construction with multiple random orderings
    elements = list(range(1, 25))
    random.seed(42)
    
    for trial in range(500):
        if time.time() > deadline - 10:
            break
        
        order = elements[:]
        random.shuffle(order)
        
        current = []
        for elem in order:
            candidate = current + [elem]
            if _check_subset_sum_avoidance(candidate):
                current = candidate
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Local search: try adding elements
    if time.time() < deadline - 5:
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            for elem in range(1, 25):
                if elem in best_set:
                    continue
                candidate = best_set + [elem]
                if _check_subset_sum_avoidance(candidate):
                    best_set = candidate
                    changed = True
    
    return best_set
# EVOLVE-BLOCK-END

