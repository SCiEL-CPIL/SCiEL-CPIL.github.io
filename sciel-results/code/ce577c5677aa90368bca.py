import time
import random
from math import log
import numpy as np

def fast_score(U):
    """
    Fast scoring using numpy.
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U)
    
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences (i < j)
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    num_pos_diffs = len(np.unique(diffs))
    num_diffs = 2 * num_pos_diffs + 1
    
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def geometric_set(base, m):
    """
    Construct a set U = {0} U { floor(base^k) for k=1..m }
    """
    U = [0]
    val = 1.0
    for _ in range(m):
        val *= base
        iv = int(val)
        if iv <= 0:
            break
        if iv not in U:
            U.append(iv)
    return sorted(set(U))

def ternary_search_geometric(deadline):
    """
    Search over base b and length m for geometric-like sets.
    """
    best_score = -1
    best_set = [0, 1]
    
    b_low, b_high = 1.05, 8.0
    m_low, m_high = 3, 40
    
    for m in range(m_low, m_high + 1):
        if time.time() > deadline - 15:
            break
        
        b_lo, b_hi = b_low, b_high
        # Ternary search for optimal base for this m
        for _ in range(12):
            if time.time() > deadline - 10:
                break
            b1 = b_lo + (b_hi - b_lo) / 3
            b2 = b_hi - (b_hi - b_lo) / 3
            
            U1 = geometric_set(b1, m)
            U2 = geometric_set(b2, m)
            
            s1 = fast_score(U1)
            s2 = fast_score(U2)
            
            if s1 > s2:
                b_hi = b2
            else:
                b_lo = b1
                
            if s1 > best_score:
                best_score = s1
                best_set = U1[:]
            if s2 > best_score:
                best_score = s2
                best_set = U2[:]
                
    return best_set, best_score

def simulated_annealing(U, deadline, max_iter=3000):
    """
    Simulated annealing to improve the set U.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score = current_score
    
    T = 0.15
    cooling = 0.997
    
    start_time = time.time()
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(current)
        r = random.random()
        if r < 0.35:
            move_type = 'perturb'
        elif r < 0.55:
            move_type = 'add'
        elif r < 0.7:
            move_type = 'remove'
        elif r < 0.85:
            move_type = 'swap'
        else:
            move_type = 'scale'
        
        new_set = current[:]
        
        if move_type == 'perturb' and n > 1:
            i = random.randint(1, n-1)
            old_val = current[i]
            max_val = max(current)
            scale_choice = random.random()
            if scale_choice < 0.4:
                scale = max(1, int(max_val * 0.03))
            elif scale_choice < 0.7:
                scale = max(1, int(max_val * 0.1))
            else:
                scale = max(1, int(max_val * 0.25))
            new_val = old_val + random.randint(-scale, scale)
            if new_val <= 0:
                new_val = old_val + random.randint(1, scale)
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
        
        elif move_type == 'add' and n < 30:
            max_val = max(current)
            if random.random() < 0.3:
                new_val = max_val + random.randint(1, max(2, int(max_val * 0.3)))
            elif random.random() < 0.6:
                new_val = random.randint(1, max_val)
            else:
                new_val = random.randint(max_val + 1, max_val * 4)
            new_set.append(new_val)
            new_set = sorted(set(new_set))
            if len(new_set) == n:
                continue
        
        elif move_type == 'remove' and n > 4:
            i = random.randint(1, n-1)
            new_set.pop(i)
        
        elif move_type == 'swap' and n > 2:
            i, j = random.sample(range(1, n), 2)
            new_set[i], new_set[j] = new_set[j], new_set[i]
            new_set = sorted(new_set)
        
        elif move_type == 'scale' and n > 1:
            factor = random.choice([0.7, 0.85, 1.15, 1.3, 1.6])
            new_set = [0] + [max(1, int(v * factor)) for v in current[1:]]
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
        
        else:
            continue
            
        new_score = fast_score(new_set)
        
        if new_score < 0:
            if random.random() < T:
                current = new_set
                current_score = new_score
        else:
            if new_score > current_score:
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current[:]
            else:
                delta = current_score - new_score
                if delta < 0.8 and random.random() < np.exp(-delta / max(T, 0.001)):
                    current = new_set
                    current_score = new_score
        
        T *= cooling
        if T < 0.001:
            T = 0.001
            
    return best, best_score

def high_precision_polish(U, deadline):
    """
    Final high-precision polish: try perturbations with wider range and multiple passes.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score = current_score
    
    # Multiple passes with decreasing step sizes
    for pass_num in range(3):
        n = len(current)
        if n < 2:
            break
        max_val = max(current)
        
        # Step sizes: start wide, narrow down
        if pass_num == 0:
            step_range = max(1, int(max_val * 0.05))
        elif pass_num == 1:
            step_range = max(1, int(max_val * 0.02))
        else:
            step_range = max(1, int(max_val * 0.005))
        
        for i in range(1, n):
            if time.time() > deadline:
                break
            old_val = current[i]
            # Try a range of offsets
            for offset in range(-step_range, step_range + 1):
                if offset == 0:
                    continue
                new_val = old_val + offset
                if new_val <= 0:
                    continue
                test_set = current[:]
                test_set[i] = new_val
                test_set = sorted(set(test_set))
                if len(test_set) != n:
                    continue
                s = fast_score(test_set)
                if s > best_score:
                    best_score = s
                    best = test_set[:]
                    current = test_set[:]
                    old_val = new_val
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: evaluator kills at 150s
    
    best_set = [0, 1, 4, 9, 11]
    best_score = fast_score(best_set)
    
    # 1. Geometric Search
    geo_set, geo_score = ternary_search_geometric(deadline)
    if geo_score > best_score:
        best_score = geo_score
        best_set = geo_set[:]
        
    # 2. Add known good candidates
    candidates = [
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 41],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67],
        geo_set[:]
    ]
    
    # Generate random sparse sets
    random.seed(42)
    for _ in range(20):
        n_elem = random.randint(4, 16)
        max_val = random.randint(30, 500)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
        
    # Geometric sets with specific ratios
    for ratio in [1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]:
        U = [0]
        val = 1
        for _ in range(15):
            U.append(val)
            val = int(val * ratio)
            if val > 10000:
                break
        candidates.append(U)
    
    # Evaluate candidates
    scored_candidates = []
    for U in candidates:
        U_clean = sorted(set(U))
        if 0 not in U_clean or len(U_clean) < 2:
            continue
        s = fast_score(U_clean)
        if s > 0:
            scored_candidates.append((s, U_clean))
            
    scored_candidates.sort(reverse=True)
    
    # 3. Run SA on top candidates
    sa_budget = deadline - 40  # Leave 40s for polish
    for idx, (score, U) in enumerate(scored_candidates[:15]):
        if time.time() > sa_budget:
            break
        remaining = sa_budget - time.time()
        alloc = max(5, min(remaining, 12))
        refined, refined_score = simulated_annealing(U, time.time() + alloc)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined[:]
            
    # 4. Final high-precision polish on the best set
    if time.time() < deadline - 5:
        refined, refined_score = high_precision_polish(best_set, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined[:]
    
    return best_set

# Test
if __name__ == "__main__":
    print(f"Best set: {construct()}")
