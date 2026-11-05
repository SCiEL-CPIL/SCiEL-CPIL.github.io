"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_sidon(subset):
    """Check if a subset has all unique pair sums (including diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _count_duplicate_sums(arr):
    """Count the number of duplicate pair sums in arr (including diagonals).
    Higher means more collisions, which correlates with smaller max Sidon subset."""
    sums = {}
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            sums[s] = sums.get(s, 0) + 1
    duplicates = 0
    for count in sums.values():
        if count > 1:
            duplicates += count - 1
    return duplicates

def _max_sidon_subset_size(arr):
    """Compute the largest subset of arr with all unordered pair sums unique (including diagonals)."""
    n = len(arr)
    # We want the largest k such that there exists a subset of size k that is Sidon.
    # Check k from n down to 1.
    for k in range(n, 0, -1):
        for subset in combinations(arr, k):
            if _is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s limit
    
    # Initial best: range(12)
    best_arr = list(range(12))
    best_size = _max_sidon_subset_size(best_arr)
    
    random.seed(42)
    
    # We will perform multiple restarts of a local search.
    # Local search minimizes _max_sidon_subset_size.
    
    for restart in range(500): # More restarts with tighter time guard
        if time.time() > deadline:
            break
            
        # Multiple initialization strategies
        strategy = restart % 5
        if strategy == 0:
            # Random set
            arr = sorted(random.sample(range(129), 12))
        elif strategy == 1:
            # Arithmetic progression with random start and step
            step = random.randint(1, 10)
            base = random.randint(0, max(0, 128 - 11 * step))
            arr = sorted([base + i * step for i in range(12)])
            if max(arr) > 128:
                continue
        elif strategy == 2:
            # Clustered: groups of 2-3 close elements
            arr = []
            for _ in range(6):
                center = random.randint(0, 120)
                size = random.randint(2, 3)
                for d in range(size):
                    val = center + d
                    if 0 <= val <= 128 and val not in arr:
                        arr.append(val)
            while len(arr) < 12:
                val = random.randint(0, 128)
                if val not in arr:
                    arr.append(val)
            arr = sorted(arr[:12])
        elif strategy == 3:
            # Geometric-like: powers or near-powers
            arr = [0]
            current = 1
            while len(arr) < 12 and current <= 128:
                arr.append(current)
                current = current * random.randint(2, 3) + random.randint(0, 5)
            while len(arr) < 12:
                val = random.randint(0, 128)
                if val not in arr:
                    arr.append(val)
            arr = sorted(arr[:12])
        else:
            # Hybrid: some clustered, some spread
            arr = []
            # 4 clusters of 3
            for _ in range(4):
                center = random.randint(0, 100)
                for d in range(3):
                    val = center + d
                    if 0 <= val <= 128 and val not in arr:
                        arr.append(val)
            while len(arr) < 12:
                val = random.randint(0, 128)
                if val not in arr:
                    arr.append(val)
            arr = sorted(arr[:12])
        
        if len(arr) != 12 or len(set(arr)) != 12:
            continue
            
        current_size = _max_sidon_subset_size(arr)
        
        if current_size < best_size:
            best_arr = arr
            best_size = current_size
            if best_size <= 4:
                return best_arr

        # Local Search with multiple move types
        no_improve_count = 0
        for step in range(2000):
            if time.time() > deadline:
                break
                
            move_type = random.randint(0, 3)
            
            if move_type == 0:
                # Replace one element
                i = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val in arr:
                    continue
                new_arr = arr.copy()
                new_arr[i] = new_val
            elif move_type == 1:
                # Swap two elements
                i, j = random.sample(range(12), 2)
                new_arr = arr.copy()
                new_arr[i], new_arr[j] = new_arr[j], new_arr[i]
            elif move_type == 2:
                # Replace two elements
                i, j = random.sample(range(12), 2)
                new_val1 = random.randint(0, 128)
                new_val2 = random.randint(0, 128)
                if new_val1 in arr or new_val2 in arr or new_val1 == new_val2:
                    continue
                new_arr = arr.copy()
                new_arr[i] = new_val1
                new_arr[j] = new_val2
            else:
                # Shift: add random delta to one element
                i = random.randint(0, 11)
                delta = random.randint(-10, 10)
                new_val = arr[i] + delta
                if new_val < 0 or new_val > 128 or new_val in arr:
                    continue
                new_arr = arr.copy()
                new_arr[i] = new_val
            
            new_arr = sorted(new_arr)
            if len(set(new_arr)) != 12:
                continue
            
            new_size = _max_sidon_subset_size(new_arr)
            
            if new_size < current_size:
                arr = new_arr
                current_size = new_size
                no_improve_count = 0
                if current_size < best_size:
                    best_arr = arr
                    best_size = current_size
                    if best_size <= 4:
                        return best_arr
            else:
                no_improve_count += 1
                if no_improve_count > 100:
                    break
        
    return best_arr
# EVOLVE-BLOCK-END
