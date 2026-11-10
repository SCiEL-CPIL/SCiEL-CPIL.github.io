import time
import random
import math

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    n = len(A)
    sumset = set()
    prodset = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sumset.add(ai + aj)
            prodset.add(ai * aj)
    return max(len(sumset), len(prodset))

def generate_lattice_points(max_val=10000):
    """
    Generate all numbers of the form 2^x * 3^y <= max_val.
    Returns a list of (value, x, y).
    """
    points = []
    x = 0
    while (2 ** x) <= max_val:
        y = 0
        val_base = 2 ** x
        while val_base * (3 ** y) <= max_val:
            val = val_base * (3 ** y)
            points.append((val, x, y))
            y += 1
        x += 1
    return points

def simulated_annealing(A_init, lattice_points, exp_map, max_val=10000, time_limit=10.0, seed=42):
    """
    Perform simulated annealing on a set of lattice points.
    """
    rng = random.Random(seed)
    A = list(A_init)
    A.sort()
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    
    start_time = time.time()
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    
    # Precompute a list of all candidate values for swapping
    all_vals = [p[0] for p in lattice_points]
    
    while (time.time() - start_time) < time_limit:
        if T < T_min:
            break
            
        # Pick a random index to swap
        i = rng.randint(0, len(A) - 1)
        old_val = A[i]
        
        # Pick a random new value from the lattice
        # Bias towards values with similar magnitude to old_val to keep sumset tight?
        # Or just random from lattice.
        new_val = rng.choice(all_vals)
        
        if new_val == old_val or new_val in A:
            continue
            
        new_A = A.copy()
        new_A[i] = new_val
        new_score = calc_score(new_A)
        
        delta = new_score - current_score
        
        if delta <= 0:
            # Accept
            A = new_A
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = list(A)
        else:
            # Metropolis criterion
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                A = new_A
                current_score = new_score
                
        T *= cooling_rate
        
    return best_A, best_score

def local_search_hill_climbing(A_init, lattice_points, exp_map, max_val=10000, time_limit=5.0, seed=42):
    """
    Aggressive hill climbing: only accept improvements.
    """
    rng = random.Random(seed)
    A = list(A_init)
    A.sort()
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    
    start_time = time.time()
    all_vals = [p[0] for p in lattice_points]
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        A_set = set(A)
        
        # Shuffle the order of indices to avoid bias
        indices = list(range(len(A)))
        rng.shuffle(indices)
        
        for i in indices:
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
                
            old_val = A[i]
            
            # Try a subset of candidates to save time, or all?
            # With ~100 candidates, trying all is fast.
            # Let's try all candidates in a random order
            candidates = [v for v in all_vals if v != old_val and v not in A_set]
            rng.shuffle(candidates)
            
            best_swap_val = None
            best_swap_score = current_score
            
            for new_val in candidates:
                new_A = A.copy()
                new_A[i] = new_val
                score = calc_score(new_A)
                
                if score < best_swap_score:
                    best_swap_score = score
                    best_swap_val = new_val
            
            if best_swap_val is not None:
                A[i] = best_swap_val
                A_set.remove(old_val)
                A_set.add(best_swap_val)
                current_score = best_swap_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_A = list(A)
                    
    return best_A, best_score

def construct():
    start_time = time.time()
    # Safety margin: 85 seconds
    deadline = start_time + 85.0
    
    # Generate lattice points
    lattice_points = generate_lattice_points(10000)
    exp_map = {val: (x, y) for val, x, y in lattice_points}
    all_vals = [val for val, _, _ in lattice_points]
    
    best_A = None
    best_score = float('inf')
    
    initial_sets = []
    
    # Strategy 1: Clusters in exponent space
    # We want exponents (x,y) to be close to minimize product set spread.
    # We also want values to be small to minimize sum set spread.
    # Center (x,y) around (2,2) -> val ~ 4*9=36
    # Center (x,y) around (4,1) -> val ~ 16*3=48
    # Center (x,y) around (1,3) -> val ~ 2*27=54
    
    centers = [
        (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (3, 0), (2, 1), (1, 2),
        (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (5, 0), (4, 1), (3, 2)
    ]
    
    for cx, cy in centers:
        # Sort lattice points by distance to center
        def exp_dist(p):
            val, x, y = p
            return abs(x - cx) + abs(y - cy)
        
        sorted_points = sorted(lattice_points, key=exp_dist)
        vals = []
        seen = set()
        for val, x, y in sorted_points:
            if val not in seen:
                vals.append(val)
                seen.add(val)
            if len(vals) == 20:
                break
        if len(vals) == 20:
            initial_sets.append(sorted(vals))
            
    # Strategy 2: Smallest values
    initial_sets.append(sorted(all_vals[:20]))
    
    # Strategy 3: Random samples from small values
    small_vals = [v for v in all_vals if v < 200]
    rng = random.Random(123)
    for _ in range(5):
        if len(small_vals) >= 20:
            sample = rng.sample(small_vals, 20)
            initial_sets.append(sorted(sample))
            
    # Deduplicate
    seen_sets = set()
    unique_initial = []
    for s in initial_sets:
        key = tuple(sorted(s))
        if key not in seen_sets:
            seen_sets.add(key)
            unique_initial.append(s)
            
    initial_sets = unique_initial
    
    # Run search on each initial set
    # Allocate time based on remaining time
    num_candidates = len(initial_sets)
    
    for idx, A in enumerate(initial_sets):
        if time.time() > deadline:
            break
            
        # Validate
        if len(set(A)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in A):
            continue
            
        remaining_time = deadline - time.time()
        remaining_count = num_candidates - idx
        alloc_time = max(1.0, remaining_time / remaining_count)
        
        # Alternate between SA and Hill Climbing
        if idx % 2 == 0:
            improved_A, score = simulated_annealing(A, lattice_points, exp_map, time_limit=alloc_time, seed=42+idx)
        else:
            improved_A, score = local_search_hill_climbing(A, lattice_points, exp_map, time_limit=alloc_time, seed=42+idx)
            
        if score < best_score:
            best_score = score
            best_A = improved_A
            
    # Final Polish: Run a quick SA on the best result found so far
    if best_A is not None and time.time() < deadline - 5.0:
        # Use remaining time for one more SA pass
        final_time = deadline - 5.0 - time.time()
        final_A, final_score = simulated_annealing(best_A, lattice_points, exp_map, time_limit=final_time, seed=999)
        if final_score < best_score:
            best_score = final_score
            best_A = final_A
            
    if best_A is None:
        # Fallback: Arithmetic Progression
        best_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        best_score = calc_score(best_A)
        
    return best_A

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Result: {sorted(res)}")
    print(f"Score: {calc_score(res)}")
    print(f"Time: {end - start:.2f}s")
