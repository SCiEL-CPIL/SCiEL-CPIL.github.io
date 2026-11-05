"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

def _set_sizes(A):
    """Compute |A+A| and |A*A| for set A (with equal elements allowed)."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return len(sums), len(prods)

def _objective(A):
    """Return max(|A+A|, |A*A|)."""
    s, p = _set_sizes(A)
    return max(s, p)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin under 60s
    
    random.seed(42)
    
    # Initial candidate: small consecutive integers
    best_set = list(range(1, 21))
    best_obj = _objective(best_set)
    
    def try_candidate(A):
        """Validate and evaluate candidate, update best if improved."""
        nonlocal best_set, best_obj
        if len(A) != 20:
            return
        if len(set(A)) != 20:
            return
        if any(x < 1 or x > 10000 for x in A):
            return
        obj = _objective(A)
        if obj < best_obj:
            best_obj = obj
            best_set = list(A)
    
    # Generate diverse initial candidates
    # 1. Small consecutive
    try_candidate(list(range(1, 21)))
    
    # 2. Powers of 2 (first 20)
    geo2 = [2**i for i in range(20)]
    if all(x <= 10000 for x in geo2):
        try_candidate(sorted(geo2))
    
    # 3. Powers of 3
    geo3 = [3**i for i in range(20)]
    if all(x <= 10000 for x in geo3):
        try_candidate(sorted(geo3))
    
    # 4. Mixed structured sets
    # Small numbers with some gaps
    for gap in [1, 2, 3]:
        cand = list(range(1, 1 + 20*gap, gap))
        if len(cand) == 20 and all(x <= 10000 for x in cand):
            try_candidate(cand)
    
    # 5. Random sets with small values
    for _ in range(30):
        cand = sorted(random.sample(range(1, 100), 20))
        try_candidate(cand)
    
    # 6. Random sets with mixed range
    for _ in range(30):
        cand = sorted(random.sample(range(1, 500), 20))
        try_candidate(cand)
    
    # 7. Structured: small base with powers
    for base in [2, 3, 4]:
        cand = []
        val = 1
        while len(cand) < 20 and val <= 10000:
            cand.append(val)
            val *= base
        if len(cand) >= 20:
            try_candidate(sorted(cand[:20]))
    
    # 8. Hybrid: some small, some large
    for _ in range(20):
        small = random.sample(range(1, 20), 10)
        large = random.sample(range(100, 1000), 10)
        cand = sorted(small + large)
        if len(set(cand)) == 20:
            try_candidate(cand)
    
    # Local search: perturb one element at a time
    improved = True
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        iteration += 1
        
        # Try perturbing each element
        for i in range(20):
            if time.time() > deadline:
                break
            orig_val = best_set[i]
            
            # Try nearby values with adaptive range
            search_range = max(10, int(0.1 * orig_val))
            for delta in range(-search_range, search_range + 1):
                if delta == 0:
                    continue
                new_val = orig_val + delta
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in best_set:
                    continue
                
                new_set = best_set[:]
                new_set[i] = new_val
                new_set.sort()
                try_candidate(new_set)
            
            # Try larger random perturbations
            for _ in range(15):
                new_val = random.randint(1, 10000)
                if new_val == orig_val or new_val in best_set:
                    continue
                new_set = best_set[:]
                new_set[i] = new_val
                new_set.sort()
                try_candidate(new_set)
        
        # Try swapping two elements
        if time.time() > deadline:
            break
        for _ in range(150):
            i, j = random.sample(range(20), 2)
            new_i = random.randint(1, 10000)
            new_j = random.randint(1, 10000)
            if new_i == new_j:
                continue
            if new_i in best_set or new_j in best_set:
                continue
            
            new_set = best_set[:]
            new_set[i] = new_i
            new_set[j] = new_j
            new_set.sort()
            try_candidate(new_set)
        
        # Try replacing multiple elements at once
        if time.time() > deadline:
            break
        for _ in range(50):
            k = random.randint(2, 5)
            indices = random.sample(range(20), k)
            new_vals = random.sample(range(1, 10000), k)
            
            # Check for duplicates
            if len(set(new_vals)) != k:
                continue
            if any(v in best_set for v in new_vals):
                continue
            
            new_set = best_set[:]
            for idx, val in zip(indices, new_vals):
                new_set[idx] = val
            new_set.sort()
            try_candidate(new_set)
    
    return best_set
# EVOLVE-BLOCK-END
