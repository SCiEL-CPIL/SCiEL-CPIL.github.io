"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sidon_subset(subset):
    """Check if a subset has all unordered pair sums (including diagonals) unique."""
    sums = set()
    for i, a in enumerate(subset):
        for j in range(i, len(subset)):
            s = a + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _largest_sidon_subset_size(ambient_set):
    """Find the largest subset of ambient_set that is Sidon (all pair sums including diagonals unique)."""
    n = len(ambient_set)
    # Try subsets in decreasing size order
    for size in range(n, 0, -1):
        for subset in combinations(ambient_set, size):
            if _is_sidon_subset(list(subset)):
                return size
    return 0

def _count_sum_collisions(ambient_set):
    """Count the number of duplicate pair sums in the ambient set."""
    sums = []
    for i, a in enumerate(ambient_set):
        for j in range(i, len(ambient_set)):
            sums.append(a + ambient_set[j])
    return len(sums) - len(set(sums))

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s to leave 5s safety margin
    
    best_set = None
    best_size = 12  # worst case: all 12 are Sidon
    
    # Try several random strategies to find a set with small largest Sidon subset
    random.seed(42)
    
    # Strategy 1: Random sampling
    for attempt in range(200):
        if time.time() > deadline:
            break
        candidate = random.sample(range(129), 12)
        size = _largest_sidon_subset_size(sorted(candidate))
        if size < best_size:
            best_size = size
            best_set = candidate
            if best_size <= 4:  # very good
                break
    
    # Strategy 2: Greedy construction - build set to maximize sum collisions
    if time.time() < deadline:
        for trial in range(50):
            if time.time() > deadline:
                break
            # Start with random pairs that have same sum
            candidate = []
            used_sums = set()
            for _ in range(6):
                # Pick a random sum s, then find pairs (a, b) with a+b=s
                s = random.randint(0, 256)
                found = False
                for a in range(0, 129):
                    b = s - a
                    if 0 <= b <= 128 and a < b and a not in candidate and b not in candidate:
                        candidate.extend([a, b])
                        found = True
                        break
                if not found:
                    # Add a random element
                    if len(candidate) < 12:
                        candidate.append(random.randint(0, 128))
            
            # Pad to 12 if needed
            while len(candidate) < 12:
                candidate.append(random.randint(0, 128))
            
            # Remove duplicates
            candidate = list(set(candidate))
            while len(candidate) < 12:
                candidate.append(random.randint(0, 128))
            
            candidate = sorted(candidate[:12])
            
            size = _largest_sidon_subset_size(candidate)
            if size < best_size:
                best_size = size
                best_set = candidate
                if best_size <= 4:
                    break
    
    # Strategy 3: Local search - perturb best set
    if best_set and time.time() < deadline:
        for iteration in range(100):
            if time.time() > deadline:
                break
            current = list(best_set)
            # Pick a random element to replace
            idx = random.randint(0, 11)
            old_val = current[idx]
            new_val = random.randint(0, 128)
            if new_val in current:
                continue
            current[idx] = new_val
            current = sorted(current)
            
            size = _largest_sidon_subset_size(current)
            if size <= best_size:
                if size < best_size:
                    best_size = size
                    best_set = current
                # Keep improving with local search
                if best_size <= 4:
                    break
    
    if best_set is None:
        return list(range(12))
    
    return best_set
# EVOLVE-BLOCK-END
