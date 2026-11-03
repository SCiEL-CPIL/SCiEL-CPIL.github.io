"""Improved program using Model-and-Optimize with Local Search."""
import time
import random
import itertools
from math import log
import numpy as np

# EVOLVE-BLOCK-START

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    sums = set()
    diffs = set()
    # Optimization: Use lists for iteration
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def generate_parametric_candidates(deadline):
    candidates = []
    
    # 1. Geometric Progressions
    for B in range(2, 15):
        for K in range(3, 12):
            U = [0]
            val = 1
            for i in range(1, K):
                U.append(val)
                val *= B
            if time.time() > deadline - 10: return candidates
            candidates.append(U)
            
    # 2. "Powers of B" with offset or scaling
    for B in range(2, 10):
        for K in range(3, 10):
            for offset in range(0, 3):
                U = [offset]
                val = offset + 1
                for i in range(1, K):
                    U.append(val)
                    val = val * B + offset
                if 0 not in U:
                    U = sorted(set(U + [0]))
                candidates.append(U)
                
    # 3. Fibonacci-like
    for K in range(3, 10):
        U = [0, 1]
        for i in range(2, K):
            U.append(U[-1] + U[-2])
        candidates.append(U)
        
    # 4. Randomized geometric
    random.seed(42)
    for _ in range(50):
        B = random.uniform(1.5, 5.0)
        K = random.randint(4, 12)
        U = [0]
        val = 1
        for i in range(1, K):
            U.append(int(val))
            val *= B
        U = sorted(set(U))
        candidates.append(U)
        
    return candidates

def score_geometric(b, m):
    """Score a geometric progression U = {floor(b^k) for k=0..m} with 0."""
    if b <= 1.0 or m < 1:
        return -1
    U = [0]
    val = 1.0
    for k in range(1, m + 1):
        U.append(int(val))
        val *= b
    U = sorted(set(U))
    if len(U) < 2:
        return -1
    return score_set(U)

def ternary_search_geometric(deadline, b_min=1.5, b_max=5.0, m_range=range(3, 15)):
    """Ternary search over base b for each m, then combine results."""
    best_score = -1
    best_params = None
    
    for m in m_range:
        if time.time() > deadline - 10:
            break
        # Ternary search over b for fixed m
        lo = b_min
        hi = b_max
        for _ in range(30):  # 30 iterations of ternary search
            if time.time() > deadline - 10:
                break
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            s1 = score_geometric(m1, m)
            s2 = score_geometric(m2, m)
            if s1 < s2:
                lo = m1
            else:
                hi = m2
        
        # Evaluate at the final point
        b_opt = (lo + hi) / 2.0
        s_opt = score_geometric(b_opt, m)
        if s_opt > best_score:
            best_score = s_opt
            best_params = (b_opt, m)
    
    return best_params, best_score

def continuous_geometric_search(deadline):
    """Perform continuous parameter sweep over geometric progressions."""
    # Phase 1: Coarse grid search to identify promising regions
    best_score = -1
    best_params = None
    
    # Coarse grid over b and m
    b_grid = np.linspace(1.5, 5.0, 20)
    m_grid = range(3, 15)
    
    for m in m_grid:
        if time.time() > deadline - 10:
            break
        for b in b_grid:
            s = score_geometric(b, m)
            if s > best_score:
                best_score = s
                best_params = (b, m)
    
    if best_params is None:
        return None, -1
    
    # Phase 2: Ternary search around the best coarse point
    b0, m0 = best_params
    b_lo = max(1.5, b0 - 0.5)
    b_hi = min(5.0, b0 + 0.5)
    
    lo = b_lo
    hi = b_hi
    for _ in range(40):  # 40 iterations for fine-grained search
        if time.time() > deadline - 10:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        s1 = score_geometric(m1, m0)
        s2 = score_geometric(m2, m0)
        if s1 < s2:
            lo = m1
        else:
            hi = m2
    
    b_opt = (lo + hi) / 2.0
    s_opt = score_geometric(b_opt, m0)
    
    # Also try adjacent m values
    for m_adj in [m0 - 1, m0 + 1]:
        if m_adj < 2 or time.time() > deadline - 10:
            continue
        # Quick ternary search for this m
        lo2 = max(1.5, b_opt - 0.3)
        hi2 = min(5.0, b_opt + 0.3)
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            m1 = lo2 + (hi2 - lo2) / 3.0
            m2 = hi2 - (hi2 - lo2) / 3.0
            s1 = score_geometric(m1, m_adj)
            s2 = score_geometric(m2, m_adj)
            if s1 < s2:
                lo2 = m1
            else:
                hi2 = m2
        b_adj = (lo2 + hi2) / 2.0
        s_adj = score_geometric(b_adj, m_adj)
        if s_adj > s_opt:
            s_opt = s_adj
            b_opt = b_adj
            m0 = m_adj
    
    # Construct the final set
    U = [0]
    val = 1.0
    for k in range(1, m0 + 1):
        U.append(int(val))
        val *= b_opt
    U = sorted(set(U))
    
    return U, s_opt

def local_search(start_set, deadline, max_iter=2000):
    current = sorted(start_set)
    current_score = score_set(current)
    if current_score < 0:
        return current, -1
    
    best_set = current[:]
    best_score = current_score
    
    # Define moves
    def try_move(new_set, allow_worsening=False, tolerance=0.0):
        nonlocal current, current_score, best_set, best_score
        new_set = sorted(set(new_set))
        if 0 not in new_set:
            return False
        if len(new_set) < 2:
            return False
        new_score = score_set(new_set)
        if allow_worsening:
            if new_score > current_score - tolerance:
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = current[:]
                return True
        else:
            if new_score > current_score:
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = current[:]
                return True
        return False

    iterations = 0
    while iterations < max_iter and time.time() < deadline - 5:
        iterations += 1
        moved = False
        
        # 1. Move one element
        if len(current) > 1:
            i = random.randint(1, len(current) - 1)
            old_val = current[i]
            # Try small shifts
            for delta in random.sample([-10, -5, -3, -2, -1, 1, 2, 3, 5, 10], 3):
                new_val = old_val + delta
                if new_val > 0:
                    new_set = current[:]
                    new_set[i] = new_val
                    if try_move(new_set):
                        moved = True
                        break
        if moved: continue
        
        # 2. Add an element
        if len(current) < 20:
            max_val = current[-1]
            # Try adding near existing elements or extending
            candidates = []
            for x in current:
                candidates.extend([x+1, x-1, x*2, x+random.randint(1,5)])
            candidates.append(max_val + random.randint(1, 10))
            candidates.append(random.randint(1, max_val))
            
            for new_val in random.sample(candidates, min(5, len(candidates))):
                if new_val > 0 and new_val not in current:
                    new_set = current + [new_val]
                    if try_move(new_set):
                        moved = True
                        break
        if moved: continue
        
        # 3. Remove an element (if set is too large)
        if len(current) > 4:
            i = random.randint(1, len(current) - 1)
            new_set = [x for j, x in enumerate(current) if j != i]
            if try_move(new_set):
                moved = True
                continue

        # 4. Simulated Annealing step (allow small worsening)
        if random.random() < 0.1:
             i = random.randint(1, len(current) - 1)
             new_val = current[i] + random.randint(-3, 3)
             if new_val > 0:
                 new_set = current[:]
                 new_set[i] = new_val
                 new_set = sorted(set(new_set))
                 if 0 in new_set and len(new_set) > 1:
                     # Allow worsening with tolerance
                     if try_move(new_set, allow_worsening=True, tolerance=0.01):
                         moved = True

    return best_set, best_score

def construct():
    deadline = time.time() + 110  # Safety margin
    
    # 1. Generate and evaluate parametric candidates
    candidates = generate_parametric_candidates(deadline)
    
    best_set = [0, 1, 4, 9, 11] # Fallback
    best_score = score_set(best_set)
    
    # 2. Continuous geometric search
    if time.time() < deadline - 20:
        geo_set, geo_score = continuous_geometric_search(deadline)
        if geo_set is not None and geo_score > best_score:
            best_score = geo_score
            best_set = geo_set[:]
            candidates.append(geo_set)
    
    # Sort candidates by initial score to prioritize good starts
    scored_candidates = []
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
        if s > 0:
            scored_candidates.append((s, c))
            
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Run local search on top candidates
    # We can run multiple searches if time permits
    # Let's pick top 5 unique structures
    seen_structures = set()
    top_starts = []
    for s, c in scored_candidates:
        key = tuple(c)
        if key not in seen_structures:
            seen_structures.add(key)
            top_starts.append(c)
        if len(top_starts) >= 5:
            break
            
    # Also include the current best if not in top starts
    if tuple(best_set) not in [tuple(x) for x in top_starts]:
        top_starts.append(best_set)
        
    for start in top_starts:
        if time.time() > deadline - 10:
            break
        local_best, local_score = local_search(start, deadline, max_iter=500)
        if local_score > best_score:
            best_score = local_score
            best_set = local_best[:]
            
    return best_set

# EVOLVE-BLOCK-END
