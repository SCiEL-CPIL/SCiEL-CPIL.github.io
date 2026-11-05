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
    deadline = start_time + 50.0  # safety margin under 60s
    
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
    
    # 2. Geometric progression (powers of 2, scaled to stay under 10000)
    try_candidate([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    # Need 20 elements, so use smaller base or mixed
    geo = []
    val = 1
    while len(geo) < 20 and val <= 10000:
        geo.append(val)
        val *= 2
    if len(geo) >= 20:
        try_candidate(sorted(geo[:20]))
    
    # 3. Random sets with small values
    for _ in range(20):
        cand = sorted(random.sample(range(1, 100), 20))
        try_candidate(cand)
    
    # 4. Random sets with mixed range
    for _ in range(20):
        cand = sorted(random.sample(range(1, 500), 20))
        try_candidate(cand)
    
    # 5. Structured: powers and small multiples
    for base in [2, 3]:
        cand = []
        val = 1
        while len(cand) < 20 and val <= 10000:
            cand.append(val)
            val *= base
        if len(cand) >= 20:
            try_candidate(sorted(cand[:20]))
    
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
            
            # Try nearby values
            for delta in range(-50, 51):
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
            for _ in range(10):
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
        for _ in range(100):
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
    
    return best_set
# EVOLVE-BLOCK-END
