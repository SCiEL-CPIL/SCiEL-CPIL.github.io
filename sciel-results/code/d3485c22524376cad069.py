import time
import random
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # safety margin before 90s kill

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A_arr = np.array(A, dtype=np.int64)
    sums = A_arr[:, None] + A_arr[None, :]
    num_sums = len(np.unique(sums))
    
    products = A_arr[:, None] * A_arr[None, :]
    num_products = len(np.unique(products))
    
    return max(num_sums, num_products)

def _is_valid(A):
    """Check if A has 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > MAX_VAL:
            return False
    return True

def _build_geometric(a, p, q):
    """Build a geometric-like sequence starting at a with ratio p/q."""
    A = []
    current = a
    for i in range(20):
        if current > MAX_VAL or current < 1:
            return None
        if current in A:
            return None
        A.append(current)
        if i < 19:
            # Use integer arithmetic: next = current * p // q
            # This may truncate, but produces valid integers
            current = (current * p) // q
    if len(set(A)) != 20:
        return None
    return A

def _rational_geometric_search(deadline):
    """
    Search for sets of the form { a * (p/q)^i } for i=0..19.
    Prioritize exact GPs where a is a multiple of q^19.
    Returns list of (A, obj) pairs sorted by objective.
    """
    results = []
    rng = random.Random(42)
    
    max_pq = 50
    
    for p in range(2, max_pq + 1):
        if time.time() > deadline:
            break
        for q in range(1, p):
            if time.time() > deadline:
                break
            
            ratio = p / q
            if ratio <= 1.0:
                continue
            
            # a * ratio^19 <= MAX_VAL
            max_a = int(MAX_VAL / (ratio ** 19))
            if max_a < 1:
                continue
            
            candidates_a = set()
            
            # Priority 1: Exact GPs (a = k * q^19)
            base = q ** 19
            max_k = max_a // base
            if max_k > 0:
                for k in range(1, min(max_k, 80) + 1):
                    candidates_a.add(k * base)
            
            # Priority 2: Boundary and small values
            candidates_a.add(max_a)
            for k in range(1, min(30, max_a) + 1):
                candidates_a.add(k)
            
            # Priority 3: Logarithmically spaced values
            if max_a > 10:
                import math
                log_min = math.log10(max(1, max_a // 100))
                log_max = math.log10(max_a)
                for frac in np.linspace(0, 1, 20):
                    val = int(10 ** (log_min + frac * (log_max - log_min)))
                    if 1 <= val <= max_a:
                        candidates_a.add(val)
            
            # Priority 4: Random samples
            for _ in range(20):
                candidates_a.add(rng.randint(1, max_a))
            
            # Evaluate candidates
            for a in candidates_a:
                if a < 1:
                    continue
                
                A = _build_geometric(a, p, q)
                if A is None:
                    continue
                
                obj = _objective(A)
                results.append((A, obj))
                
                if time.time() > deadline:
                    break
        
        if time.time() > deadline:
            break
    
    # Sort by objective
    results.sort(key=lambda x: x[1])
    return results

def _near_geometric_perturb(A, rng, max_perturb=3):
    """
    Perturb a set while preserving near-geometric structure.
    Try to maintain multiplicative ratios by adjusting values slightly.
    """
    A = list(A)
    n = len(A)
    
    # Compute current ratios
    ratios = [A[i+1] / A[i] for i in range(n-1)]
    
    # Try perturbations that preserve ratio structure
    for _ in range(max_perturb * 5):
        idx = rng.randint(0, n-1)
        old_val = A[idx]
        
        # Generate candidates near old_val that might preserve structure
        candidates = set()
        
        # Small perturbations
        for d in range(-10, 11):
            if d != 0:
                candidates.add(old_val + d)
        
        # Multiplicative perturbations based on neighboring ratios
        if idx > 0:
            r_prev = A[idx] / A[idx-1]
            for factor in [0.9, 0.95, 1.05, 1.1]:
                candidates.add(int(A[idx-1] * r_prev * factor))
        
        if idx < n-1:
            r_next = A[idx+1] / A[idx]
            for factor in [0.9, 0.95, 1.05, 1.1]:
                candidates.add(int(A[idx+1] / r_next * factor))
        
        for new_val in candidates:
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val in A:
                continue
            
            A[idx] = new_val
            return A, True
    
    return A, False

def _local_search(A, time_limit, rng, temperature=50.0):
    """Local search with simulated annealing, enhanced with near-geometric perturbations."""
    A = list(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    cooling_rate = 0.995
    min_temperature = 0.01
    
    while temperature > min_temperature:
        if time.time() - start_time > time_limit:
            break
        
        # Mix of strategies: 60% local perturbation, 40% near-geometric
        if rng.random() < 0.6:
            idx = rng.randint(0, 19)
            old_val = A[idx]
            
            for _ in range(25):
                if time.time() - start_time > time_limit:
                    break
                
                r = rng.random()
                if r < 0.3:
                    new_val = old_val + rng.randint(-15, 15)
                elif r < 0.6:
                    new_val = old_val + rng.randint(-50, 50)
                elif r < 0.85:
                    new_val = old_val + rng.randint(-150, 150)
                else:
                    new_val = old_val + rng.randint(-400, 400)
                
                if new_val < 1 or new_val > MAX_VAL:
                    continue
                if new_val in A:
                    continue
                
                A[idx] = new_val
                new_obj = _objective(A)
                A[idx] = old_val
                
                delta = new_obj - current_obj
                if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                    A[idx] = new_val
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    break
        else:
            # Near-geometric perturbation
            new_A, success = _near_geometric_perturb(A, rng, max_perturb=3)
            if success:
                new_obj = _objective(new_A)
                delta = new_obj - current_obj
                if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                    A = new_A
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _polish_pass(A, time_limit, rng):
    """Final polish with very small steps and targeted product collision search."""
    A = list(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        for idx in range(20):
            if time.time() - start_time > time_limit:
                break
            
            old_val = A[idx]
            
            # Standard small perturbations
            for delta in range(-10, 11):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 1 or new_val > MAX_VAL:
                    continue
                if new_val in A:
                    continue
                
                A[idx] = new_val
                new_obj = _objective(A)
                A[idx] = old_val
                
                if new_obj < current_obj:
                    A[idx] = new_val
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    improved = True
                    break
            
            # Targeted: try values that create product collisions
            # For each other element, try to make A[idx] * A[j] collide with existing products
            for j in range(20):
                if j == idx:
                    continue
                target = A[j] * A[idx]
                # Try to find a new_val such that new_val * A[j] equals an existing product
                # This is complex, so just try a few candidates near old_val
                for d in [-3, -2, -1, 1, 2, 3]:
                    new_val = old_val + d
                    if new_val < 1 or new_val > MAX_VAL:
                        continue
                    if new_val in A:
                        continue
                    
                    A[idx] = new_val
                    new_obj = _objective(A)
                    A[idx] = old_val
                    
                    if new_obj < current_obj:
                        A[idx] = new_val
                        current_obj = new_obj
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_A = list(A)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 3  # leave margin
    deadline = start_time + time_budget
    
    best_A = None
    best_obj = float('inf')
    
    # Phase 1: Deterministic Rational Geometric Search
    try:
        results = _rational_geometric_search(deadline)
        if results:
            # Take top candidates for multi-start refinement
            top_candidates = results[:min(8, len(results))]
            best_A = top_candidates[0][0]
            best_obj = top_candidates[0][1]
            
            # Phase 2: Refine top candidates
            if time.time() < deadline:
                remaining = deadline - time.time()
                per_candidate = remaining / max(1, len(top_candidates))
                
                for i, (cand_A, cand_obj) in enumerate(top_candidates):
                    if time.time() > deadline - 2:
                        break
                    
                    # Local search
                    if per_candidate > 1.5:
                        try:
                            refined_A, refined_obj = _local_search(
                                cand_A, per_candidate * 0.7, random.Random(100 + i)
                            )
                            if refined_obj < best_obj:
                                best_A = refined_A
                                best_obj = refined_obj
                        except Exception:
                            pass
                    
                    # Polish
                    if time.time() > deadline - 1:
                        break
                    remaining_after_ls = deadline - time.time()
                    if remaining_after_ls > 0.5:
                        try:
                            polished_A, polished_obj = _polish_pass(
                                best_A, remaining_after_ls, random.Random(200 + i)
                            )
                            if polished_obj < best_obj:
                                best_A = polished_A
                                best_obj = polished_obj
                        except Exception:
                            pass
                
                # Final polish on best
                if time.time() < deadline - 0.5:
                    try:
                        polished_A, polished_obj = _polish_pass(
                            best_A, deadline - time.time() - 0.5, random.Random(999)
                        )
                        if polished_obj < best_obj:
                            best_A = polished_A
                            best_obj = polished_obj
                    except Exception:
                        pass
    except Exception:
        pass
    
    # Fallback
    if best_A is None:
        best_A = list(range(1, 21))
    
    # Ensure validity
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return sorted(best_A)
