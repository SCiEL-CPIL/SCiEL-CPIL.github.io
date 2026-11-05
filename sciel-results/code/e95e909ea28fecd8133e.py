"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _compute_cost(A):
    """
    Compute max(|A+A|, |A*A|) for a list A of distinct integers.
    """
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def _compute_cost_pair(A):
    """Return (|A+A|, |A*A|) for diagnostics."""
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return (len(sums), len(prods))

def _generate_ap(start=1, step=1, n=20, max_val=10000):
    A = [start + i * step for i in range(n)]
    if max(A) <= max_val:
        return sorted(A)
    return None

def _generate_gp(n=20, max_val=10000):
    A = set()
    val = 1
    while len(A) < n and val <= max_val:
        A.add(val)
        val *= 2
    if len(A) >= n:
        return sorted(list(A)[:n])
    candidate = 1
    while len(A) < n:
        if candidate not in A:
            A.add(candidate)
        candidate += 1
    return sorted(A)

def _generate_random(n=20, max_val=10000):
    return sorted(random.sample(range(1, max_val + 1), n))

def _generate_mixed(n=20, max_val=10000):
    """Mix of small AP and medium spaced values."""
    A = set()
    # Small consecutive block
    for i in range(1, 11):
        A.add(i)
    # Add spaced values
    base = 50
    i = 1
    while len(A) < n:
        v = base + i * 30
        if v <= max_val and v not in A:
            A.add(v)
        i += 1
    if len(A) >= n:
        return sorted(list(A)[:n])
    candidate = 1
    while len(A) < n:
        if candidate not in A:
            A.add(candidate)
        candidate += 1
    return sorted(A)

def _generate_power_mix(n=20, max_val=10000):
    """Mix of small numbers and powers."""
    A = set()
    # Small numbers 1-8
    for i in range(1, 9):
        A.add(i)
    # Powers of 2 from 16 onwards
    val = 16
    while len(A) < n and val <= max_val:
        A.add(val)
        val *= 2
    # Fill remaining
    candidate = 9
    while len(A) < n:
        if candidate not in A:
            A.add(candidate)
        candidate += 1
    return sorted(list(A)[:n])

def _generate_log_spaced(n=20, max_val=10000):
    """Logarithmically spaced integers."""
    A = set()
    for i in range(n):
        val = int(max_val ** (i / (n - 1))) + 1
        if 1 <= val <= max_val:
            A.add(val)
    # Fill remaining
    candidate = 1
    while len(A) < n:
        if candidate not in A:
            A.add(candidate)
        candidate += 1
    return sorted(list(A)[:n])

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    n = 20
    max_val = 10000
    
    # Generate diverse initial candidates
    candidates = []
    
    # APs with various parameters
    candidates.append(_generate_ap(1, 1, n, max_val))
    candidates.append(_generate_ap(1, 2, n, max_val))
    candidates.append(_generate_ap(1, 3, n, max_val))
    candidates.append(_generate_ap(5, 1, n, max_val))
    candidates.append(_generate_ap(10, 1, n, max_val))
    candidates.append(_generate_ap(1, 5, n, max_val))
    
    # GP
    candidates.append(_generate_gp(n, max_val))
    
    # Mixed structures
    candidates.append(_generate_mixed(n, max_val))
    candidates.append(_generate_power_mix(n, max_val))
    candidates.append(_generate_log_spaced(n, max_val))
    
    # Random initializations
    for _ in range(20):
        candidates.append(_generate_random(n, max_val))
    
    # Filter out None
    candidates = [c for c in candidates if c is not None]
    
    # Evaluate all initial candidates
    best_A = candidates[0]
    best_cost = _compute_cost(best_A)
    
    for init_A in candidates[1:]:
        if time.time() > deadline:
            break
        c = _compute_cost(init_A)
        if c < best_cost:
            best_cost = c
            best_A = init_A[:]
    
    # Simulated annealing / local search from multiple starting points
    # Use the best few initializations as starting points for SA
    
    # Sort candidates by cost
    scored_candidates = []
    for init_A in candidates:
        c = _compute_cost(init_A)
        scored_candidates.append((c, init_A))
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 5 as SA starting points
    sa_starts = [sc[1] for sc in scored_candidates[:5]]
    
    for start_idx, init_A in enumerate(sa_starts):
        if time.time() > deadline:
            break
        
        current_A = init_A[:]
        current_cost = _compute_cost(current_A)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_A = current_A[:]
        
        # Simulated annealing parameters
        T = 50.0  # Initial temperature
        T_min = 0.1
        cooling = 0.995
        max_iter = 200000
        iter_count = 0
        
        while time.time() < deadline and T > T_min and iter_count < max_iter:
            iter_count += 1
            
            # Choose a random element to modify
            idx = random.randint(0, n - 1)
            old_val = current_A[idx]
            
            # Generate new value
            if random.random() < 0.3:
                # Large jump
                new_val = random.randint(1, max_val)
            elif random.random() < 0.6:
                # Medium perturbation
                delta = random.randint(-200, 200)
                new_val = old_val + delta
            else:
                # Small perturbation
                delta = random.randint(-10, 10)
                new_val = old_val + delta
            
            new_val = max(1, min(max_val, new_val))
            
            if new_val == old_val:
                continue
            
            # Check distinctness
            is_distinct = True
            for i, v in enumerate(current_A):
                if i != idx and v == new_val:
                    is_distinct = False
                    break
            
            if not is_distinct:
                continue
            
            # Apply and evaluate
            current_A[idx] = new_val
            new_cost = _compute_cost(current_A)
            
            delta_cost = new_cost - current_cost
            
            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                # Accept
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = current_A[:]
            else:
                # Revert
                current_A[idx] = old_val
            
            T *= cooling
        
        # After SA, do a final local search pass
        no_improve = 0
        while time.time() < deadline and no_improve < 5000:
            idx = random.randint(0, n - 1)
            old_val = current_A[idx]
            
            if random.random() < 0.5:
                new_val = random.randint(1, max_val)
            else:
                delta = random.randint(-50, 50)
                new_val = old_val + delta
                new_val = max(1, min(max_val, new_val))
            
            if new_val == old_val:
                no_improve += 1
                continue
            
            is_distinct = True
            for i, v in enumerate(current_A):
                if i != idx and v == new_val:
                    is_distinct = False
                    break
            
            if not is_distinct:
                no_improve += 1
                continue
            
            current_A[idx] = new_val
            new_cost = _compute_cost(current_A)
            
            if new_cost < current_cost:
                current_cost = new_cost
                no_improve = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = current_A[:]
            else:
                current_A[idx] = old_val
                no_improve += 1
    
    return best_A

# EVOLVE-BLOCK-END
