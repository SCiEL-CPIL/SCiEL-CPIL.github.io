"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _count_collisions(arr):
    """Count number of unordered pair sum collisions (including diagonals)."""
    sums = {}
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            sums[s] = sums.get(s, 0) + 1
    return sum(c * (c - 1) // 2 for c in sums.values())

def _max_sidon_size_fast(arr):
    """Optimized: try subsets greedily, then verify."""
    n = len(arr)
    # First check if the full set is Sidon
    sums = set()
    valid = True
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    if valid:
        return n
    
    # Try size n-1
    for skip in range(n):
        subset = [arr[i] for i in range(n) if i != skip]
        sums = set()
        valid = True
        for i in range(n - 1):
            for j in range(i, n - 1):
                s = subset[i] + subset[j]
                if s in sums:
                    valid = False
                    break
                sums.add(s)
            if not valid:
                break
        if valid:
            return n - 1
    
    # Try size n-2
    for skip1 in range(n):
        for skip2 in range(skip1 + 1, n):
            subset = [arr[i] for i in range(n) if i != skip1 and i != skip2]
            sums = set()
            valid = True
            for i in range(n - 2):
                for j in range(i, n - 2):
                    s = subset[i] + subset[j]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
            if valid:
                return n - 2
    
    # Fallback: exhaustive from size n-3 down
    for size in range(n - 3, 1, -1):
        for subset in combinations(range(n), size):
            sums = set()
            valid = True
            for i in range(size):
                for j in range(i, size):
                    s = arr[subset[i]] + arr[subset[j]]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                return size
    return 1

def _collision_density_score(arr):
    """
    Compute a surrogate score based on collision graph density.
    For each pair (i,j) with i<=j, compute sum s = arr[i]+arr[j].
    Count how many other pairs produce the same sum.
    Higher total collision count = more 'hard' to extract large Sidon subsets.
    This is a smooth-ish surrogate that correlates with low max Sidon size.
    """
    sums_count = {}
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            sums_count[s] = sums_count.get(s, 0) + 1
    # Total number of collision pairs (excess over 1 per sum)
    total_collisions = sum(c * (c - 1) // 2 for c in sums_count.values())
    return total_collisions

def _suggested_replacements(arr, k=10):
    """
    Suggest replacement values that maximize local collision density.
    For each position i, find values v not in arr that, when replacing arr[i],
    maximize the collision density score (as a proxy for minimizing max Sidon size).
    Returns a list of (index, new_value) pairs sorted by expected improvement.
    """
    n = len(arr)
    current_score = _collision_density_score(arr)
    suggestions = []
    
    # Precompute sum counts for current arrangement
    sums_count = {}
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            sums_count[s] = sums_count.get(s, 0) + 1
    
    for i in range(n):
        old_val = arr[i]
        # For each candidate new value, compute how it affects collisions
        best_local = -1
        best_local_val = None
        # Sample a subset of candidate values to keep this fast
        candidates = set()
        # Values that create many collisions with existing elements
        for v in range(0, 129):
            if v in arr:
                continue
            # Quick estimate: count how many existing elements x have (v+x) 
            # colliding with existing sums
            local_coll = 0
            for j in range(n):
                if j == i:
                    continue
                s = v + arr[j]
                # How many times does this sum appear in current arrangement?
                # (excluding pairs involving element i, which we're replacing)
                if s in sums_count:
                    # Subtract pairs that included old_val
                    if s - old_val == arr[j] and j != i:
                        # This sum appeared via old_val + arr[j]
                        pass
                    local_coll += sums_count.get(s, 0)
            # Also count diagonal: v + v
            s_diag = 2 * v
            local_coll += sums_count.get(s_diag, 0)
            
            if local_coll > best_local:
                best_local = local_coll
                best_local_val = v
        
        if best_local_val is not None:
            suggestions.append((best_local, i, best_local_val))
    
    # Sort by collision count descending
    suggestions.sort(key=lambda x: -x[0])
    return suggestions[:k]

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit
    
    random.seed(42)
    
    # Start with a structured initial set that has many collisions
    # Use arithmetic progression-like structure to create sum collisions
    best_arr = list(range(0, 128, 128 // 12))[:12]
    # Ensure distinct
    seen = set()
    unique_best = []
    for x in best_arr:
        if x not in seen:
            seen.add(x)
            unique_best.append(x)
    while len(unique_best) < 12:
        unique_best.append(random.randint(0, 128))
        seen.add(unique_best[-1])
    best_arr = unique_best[:12]
    
    best_score = _max_sidon_size_fast(best_arr)
    
    # Local search with collision-density-guided moves
    no_improve_count = 0
    max_no_improve = 800
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Strategy 1: Use collision-density suggestions for targeted replacements
        if no_improve_count % 50 == 0 or no_improve_count == 0:
            # Get suggestions from collision graph
            suggestions = _suggested_replacements(best_arr, k=8)
            for _, idx, new_val in suggestions:
                if time.time() >= deadline:
                    break
                candidate = best_arr[:]
                candidate[idx] = new_val
                if len(set(candidate)) == 12:
                    score = _max_sidon_size_fast(candidate)
                    if score < best_score:
                        best_arr = candidate
                        best_score = score
                        no_improve_count = 0
                        break
                    else:
                        no_improve_count += 1
        else:
            # Strategy 2: Random replacement
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            candidate = best_arr[:]
            candidate[i] = new_val
            if len(set(candidate)) == 12:
                score = _max_sidon_size_fast(candidate)
                if score < best_score:
                    best_arr = candidate
                    best_score = score
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            # Strategy 3: Random two-element replacement
            i, j = random.sample(range(12), 2)
            candidate = best_arr[:]
            candidate[i] = random.randint(0, 128)
            candidate[j] = random.randint(0, 128)
            if len(set(candidate)) == 12:
                score = _max_sidon_size_fast(candidate)
                if score < best_score:
                    best_arr = candidate
                    best_score = score
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        
        # Strategy 4: Pure swap of two elements (sometimes helpful)
        if random.random() < 0.1:
            i, j = random.sample(range(12), 2)
            candidate = best_arr[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            score = _max_sidon_size_fast(candidate)
            if score < best_score:
                best_arr = candidate
                best_score = score
                no_improve_count = 0
            else:
                no_improve_count += 1
    
    return best_arr
# EVOLVE-BLOCK-END
