"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

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
    # Safety margin: The evaluator kills at 90s, but we should aim for < 60s.
    # Using 55s to be safe.
    deadline = start_time + 55.0
    
    # Fixed seed for reproducibility in evaluation if needed, 
    # though randomization is key for search.
    random.seed(42)
    
    # The problem is scale-invariant in structure. 
    # Optimal sets often have small elements to maximize collisions in sums and products.
    # We restrict the search space to a small range to make local search effective.
    # If the best set in [1, SEARCH_MAX] has max element < SEARCH_MAX, 
    # we can try scaling it up, but usually the "shape" is what matters.
    # Let's search in [1, 200]. This is dense enough for 20 elements.
    SEARCH_MAX = 200
    
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
        # We primarily search in small numbers, but allow up to 10000 if found
        obj = _objective(A)
        if obj < best_obj:
            best_obj = obj
            best_set = list(A)
            
    # --- Diverse Initial Candidates in Small Range ---
    
    # 1. Consecutive
    try_candidate(list(range(1, 21)))
    
    # 2. Arithmetic Progressions
    for start in range(1, 5):
        for step in range(1, 15):
            cand = [start + i * step for i in range(20)]
            if max(cand) <= SEARCH_MAX:
                try_candidate(sorted(cand))
                
    # 3. Geometric Progressions (Powers of 2, 3, etc.)
    for base in [2, 3]:
        cand = []
        val = 1
        while len(cand) < 20 and val <= SEARCH_MAX:
            cand.append(val)
            val *= base
        if len(cand) == 20:
            try_candidate(sorted(cand))
            
    # 4. Random sets in small range
    for _ in range(200):
        cand = sorted(random.sample(range(1, SEARCH_MAX + 1), 20))
        try_candidate(cand)
        
    # 5. Structured "Mixed": Small numbers + some larger to break symmetry?
    # Actually, keeping them small is usually better for collisions.
    # Try sets that are dense in [1, 50] and sparse elsewhere.
    for _ in range(50):
        smalls = random.sample(range(1, 50), 15)
        larges = random.sample(range(50, SEARCH_MAX + 1), 5)
        cand = sorted(smalls + larges)
        if len(set(cand)) == 20:
            try_candidate(cand)

    # --- Simulated Annealing Refinement ---
    current_set = best_set[:]
    current_obj = best_obj
    
    # Initial temperature
    T = 10.0
    cooling = 0.995
    min_T = 0.01
    
    # Track stagnation
    no_improve_count = 0
    
    while time.time() < deadline:
        if T < min_T:
            # Restart
            current_set = best_set[:]
            current_obj = best_obj
            T = 10.0
            no_improve_count = 0
            continue
            
        new_set = current_set[:]
        move_type = random.random()
        
        # Define moves
        if move_type < 0.5:
            # Nudge one element by small amount
            i = random.randint(0, 19)
            # Bias towards small deltas
            delta = random.choice([-3, -2, -1, 1, 2, 3, 5, -5, 10, -10])
            new_val = current_set[i] + delta
            if 1 <= new_val <= SEARCH_MAX and new_val not in current_set:
                new_set[i] = new_val
            else:
                continue
                
        elif move_type < 0.7:
            # Replace one element with random value in range
            i = random.randint(0, 19)
            new_val = random.randint(1, SEARCH_MAX)
            if new_val not in current_set:
                new_set[i] = new_val
            else:
                continue
                
        elif move_type < 0.9:
            # Swap two elements with random values
            i, j = random.sample(range(20), 2)
            new_i = random.randint(1, SEARCH_MAX)
            new_j = random.randint(1, SEARCH_MAX)
            if new_i != new_j and new_i not in current_set and new_j not in current_set:
                new_set[i] = new_i
                new_set[j] = new_j
            else:
                continue
        else:
            # Replace 2-3 elements with random values
            k = random.randint(2, 3)
            indices = random.sample(range(20), k)
            existing = set(current_set)
            new_vals = set()
            for _ in range(k * 10):
                v = random.randint(1, SEARCH_MAX)
                if v not in existing and v not in new_vals:
                    new_vals.add(v)
                    if len(new_vals) == k:
                        break
            if len(new_vals) != k:
                continue
            for idx, val in zip(indices, list(new_vals)):
                new_set[idx] = val
                
        new_set.sort()
        
        if len(new_set) != 20 or len(set(new_set)) != 20:
            continue
            
        new_obj = _objective(new_set)
        
        # Acceptance criterion
        if new_obj <= current_obj:
            current_set = new_set
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_set = new_set[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            delta_obj = new_obj - current_obj
            # Avoid overflow in exp
            if delta_obj < 100:
                accept_prob = math.exp(-delta_obj / max(T, 0.01))
                if random.random() < accept_prob:
                    current_set = new_set
                    current_obj = new_obj
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        T *= cooling
        
    return best_set
# EVOLVE-BLOCK-END
