"""
Combined approach: Structural insight (dynamic candidate generation from arithmetic neighborhoods)
+ Refinement machinery (efficient scoring, ternary/geometric search, and local search with SA).
+ Integration of Singer Difference Sets for structural prior.
+ Linear recurrence-based set generation with continuous parameter tuning.
"""
import time
import random
import numpy as np
from math import log

# EVOLVE-BLOCK-START

def score_set(U):
    """
    Efficiently score a set U based on the sumset/difference-set exponent.
    Uses Python sets for small N and NumPy for larger N to balance overhead and speed.
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    
    # For small sets, Python sets are faster due to lower overhead
    if n < 15:
        sums = set()
        diffs = set()
        for i in range(n):
            ui = U[i]
            for j in range(i, n):
                sums.add(ui + U[j])
            for j in range(i + 1, n):
                d = U[j] - ui
                diffs.add(d)
                diffs.add(-d)
        diffs.add(0)
        
        sum_count = len(sums)
        diff_count = len(diffs)
    else:
        # For larger sets, NumPy vectorization is faster
        U_arr = np.array(U, dtype=np.int64)
        
        # Compute all sums (i <= j)
        sums_arr = U_arr[:, None] + U_arr[None, :]
        sum_vals = np.unique(sums_arr)
        sum_count = len(sum_vals)
        
        # Compute all differences
        diffs_arr = U_arr[:, None] - U_arr[None, :]
        diff_vals = np.unique(diffs_arr)
        diff_count = len(diff_vals)

    if sum_count == 0:
        return -1.0
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1.0
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
        
    return 1 + log(ratio) / log(D)

def generate_recurrence_set(a1, a2, K):
    """
    Generate a set from a linear recurrence relation:
    u_0 = 0, u_1 = a1, u_2 = a2, u_n = p*u_{n-1} + q*u_{n-2}
    where p and q are derived from a1, a2 to ensure growth.
    
    The characteristic equation r^2 - p*r - q = 0 should have roots with |r| > 1
    to ensure super-polynomial growth, which helps separate sumset and difference set.
    """
    if K < 2:
        return [0]
    if K == 2:
        return [0, a1]
    
    # Choose p and q such that the recurrence grows quickly
    # For simplicity, use p = round(a2/a1) if a1 > 0, else p = 2
    if a1 > 0 and a2 > a1:
        p = max(2, round(a2 / a1))
    else:
        p = 2
    
    # q should be small to keep the recurrence stable
    q = 1 if p > 2 else 2
    
    U = [0, a1]
    if K > 2:
        U.append(a2)
    
    for i in range(3, K):
        next_val = p * U[-1] + q * U[-2]
        if next_val <= 0 or next_val == U[-1]:
            next_val = U[-1] + 1
        U.append(next_val)
    
    # Ensure all values are distinct and sorted
    U = sorted(set(U))
    return U

def generate_parametric_recurrence_candidates():
    """
    Generate candidates from linear recurrences with tuned parameters.
    """
    candidates = []
    
    # Vary the initial conditions and recurrence coefficients
    for a1 in range(1, 10):
        for a2 in range(a1 + 1, 20):
            for K in range(4, 15):
                U = generate_recurrence_set(a1, a2, K)
                if len(U) >= 3 and 0 in U:
                    candidates.append(U)
    
    # Also try with different recurrence structures
    for p in [2, 3, 4, 5]:
        for q in [1, 2, 3]:
            for a1 in range(1, 5):
                U = [0, a1]
                if p > 1:
                    U.append(p * a1)
                for i in range(3, 12):
                    next_val = p * U[-1] + q * U[-2]
                    if next_val > 0 and next_val != U[-1]:
                        U.append(next_val)
                U = sorted(set(U))
                if len(U) >= 3:
                    candidates.append(U)
    
    return candidates

def generate_structured_candidates():
    """
    Generate high-quality structured sets:
    1. Geometric Progressions
    2. Singer Difference Sets (Perfect Difference Sets)
    3. Fibonacci-like
    4. Linear Recurrence Sets
    """
    candidates = []
    
    # 1. Geometric Progressions
    for B in range(2, 15):
        for K in range(3, 12):
            U = [0]
            val = 1
            for i in range(1, K):
                U.append(val)
                val *= B
            candidates.append(U)
            
    # 2. Singer Difference Sets (known perfect difference sets)
    known_singer = {
        2: [0, 1, 3],
        3: [0, 1, 3, 4, 9, 10, 12],
        4: [0, 1, 4, 10, 12, 13, 15, 16, 19, 21, 25],
        5: [0, 1, 4, 14, 16, 17, 22, 28, 29, 34, 35, 38, 39, 41, 44],
        7: [0, 1, 3, 7, 12, 18, 21, 29, 32, 34, 37, 38, 46, 48, 49, 51, 53, 57, 58, 60, 63],
    }
    
    for q, U in known_singer.items():
        candidates.append(U)
        
    # 3. Fibonacci-like
    for K in range(3, 12):
        U = [0, 1]
        for i in range(2, K):
            U.append(U[-1] + U[-2])
        candidates.append(U)
        
    # 4. Powers of 2 (Sidon-like)
    for K in range(3, 15):
        U = [0]
        val = 1
        for i in range(1, K):
            U.append(val)
            val *= 2
        candidates.append(U)
    
    # 5. Linear Recurrence Sets
    recurrence_candidates = generate_parametric_recurrence_candidates()
    candidates.extend(recurrence_candidates)
    
    return candidates

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    """
    U_set = set(U)
    candidates = set()
    
    # Sums and differences of existing elements
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
    
    # Add some random small integers to ensure diversity
    for _ in range(20):
        r = random.randint(1, min(50, limit_val))
        if r not in U_set:
            candidates.add(r)
            
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=2000, max_n_limit=30):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 15
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.15:
            current_max = max(U)
            new_limit = max(max_val_limit, int(current_max * 1.5))
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.4:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                random.shuffle(candidates)
                for c in candidates[:8]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, int(max(U)*1.5)))
                        break
        elif move_type < 0.65:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max_val_limit)
        else:
            # Try to REPLACE an element
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                random.shuffle(candidates)
                for new_val in candidates[:8]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, int(max(U)*1.5)))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    best_score = -1.0
    best_set = [0, 1, 4, 9, 11]
    
    # Initial best
    s = score_set(best_set)
    if s > best_score:
        best_score = s
        best_set = list(best_set)

    # 1. Try structured candidates (including Singer sets and recurrence sets)
    structured_candidates = generate_structured_candidates()
    
    # Add a few more specific geometric ones if not covered
    for B in [1.5, 2.5, 3.5]:
        U = [0]
        val = 1.0
        for i in range(1, 10):
            U.append(int(val))
            val *= B
        U = sorted(set(U))
        structured_candidates.append(U)

    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
            
    # 2. Local Search with Dynamic Candidate Generation
    # We will run multiple restarts from different structured seeds.
    max_val_limit = 2000 
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        # Choose a starting set from the structured candidates
        # Cycle through them to ensure diversity
        base = structured_candidates[restarts % len(structured_candidates)]
        U = list(base)
        
        # Run local search
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    # Final polish: Try very small moves on the best set found
    if time.time() < deadline - 2:
        U = list(best_set)
        U.sort()
        final_score, final_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
            
    return best_set

# EVOLVE-BLOCK-END
