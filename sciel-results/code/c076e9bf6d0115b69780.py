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
    
    # Use numpy for faster computation when set is large enough
    if n >= 10:
        U_arr = np.array(U, dtype=np.int64)
        # Compute all pairwise sums
        sums_arr = U_arr[:, None] + U_arr[None, :]
        # Only consider i <= j to avoid duplicates
        mask = np.triu_indices(n)
        sums_unique = np.unique(sums_arr[mask[0], mask[1]])
        
        # Compute all pairwise differences
        diffs_arr = U_arr[:, None] - U_arr[None, :]
        diffs_unique = np.unique(diffs_arr)
        
        ratio = len(diffs_unique) / len(sums_unique)
        if ratio <= 1:
            return -1
        max_u = U[-1]
        D = 2 * max_u + 1
        if D <= 1:
            return -1
        return 1 + log(ratio) / log(D)
    else:
        # Original implementation for small sets
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
    no_improve_count = 0
    max_no_improve = 100
    
    # Define moves
    def try_move(new_set):
        nonlocal current, current_score, best_set, best_score, no_improve_count
        new_set = sorted(set(new_set))
        if 0 not in new_set:
            return False
        if len(new_set) < 2:
            return False
        new_score = score_set(new_set)
        if new_score > current_score + 1e-10:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current[:]
                no_improve_count = 0
            return True
        else:
            no_improve_count += 1
            return False

    # Temperature for simulated annealing
    temperature = 0.05
    cooling_rate = 0.995
    
    iterations = 0
    while iterations < max_iter and time.time() < deadline - 5:
        iterations += 1
        
        # If too many non-improving moves, cool down and reset
        if no_improve_count > max_no_improve:
            temperature *= cooling_rate
            no_improve_count = 0
            if temperature < 0.001:
                break
        
        moved = False
        
        # 1. Move one element with larger search radius
        if len(current) > 1:
            i = random.randint(1, len(current) - 1)
            old_val = current[i]
            # Adaptive delta based on local density
            local_range = max(1, int(old_val * 0.1))
            deltas = random.sample(range(-local_range, local_range + 1), min(5, 2 * local_range + 1))
            for delta in deltas:
                new_val = old_val + delta
                if new_val > 0:
                    new_set = current[:]
                    new_set[i] = new_val
                    if try_move(new_set):
                        moved = True
                        break
        if moved: continue
        
        # 2. Add an element with better candidate generation
        if len(current) < 25:
            max_val = current[-1]
            # Generate candidates based on gaps and structure
            candidates = set()
            
            # Near existing elements
            for x in current:
                candidates.update([x + 1, x - 1, x * 2, x + random.randint(1, 3)])
            
            # Fill gaps
            if len(current) > 2:
                for i in range(len(current) - 1):
                    gap = current[i + 1] - current[i]
                    if gap > 1:
                        candidates.add(current[i] + gap // 2)
                        candidates.add(current[i] + gap - 1)
            
            # Extend the set
            candidates.add(max_val + random.randint(1, max(2, max_val // 10)))
            candidates.add(max_val * 2)
            
            # Random candidates in range
            for _ in range(3):
                candidates.add(random.randint(1, max_val))
            
            for new_val in list(candidates)[:7]:
                if new_val > 0 and new_val not in current:
                    new_set = current + [new_val]
                    if try_move(new_set):
                        moved = True
                        break
        if moved: continue
        
        # 3. Remove an element with strategic selection
        if len(current) > 5:
            # Try removing elements that might be causing sum collisions
            i = random.randint(1, len(current) - 1)
            new_set = [x for j, x in enumerate(current) if j != i]
            if try_move(new_set):
                moved = True
                continue
        
        # 4. Simulated Annealing step (allow worsening moves)
        if random.random() < 0.15:
            i = random.randint(1, len(current) - 1)
            # Larger perturbation for SA
            perturbation = random.randint(-5, 5)
            new_val = current[i] + perturbation
            if new_val > 0:
                new_set = current[:]
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                if 0 in new_set and len(new_set) > 1:
                    new_score = score_set(new_set)
                    delta_score = new_score - current_score
                    # Accept with probability based on temperature
                    if delta_score > 0 or random.random() < np.exp(delta_score / temperature):
                        if try_move(new_set):
                            moved = True

    return best_set, best_score

def construct():
    deadline = time.time() + 105  # Safety margin
    
    # 1. Generate and evaluate parametric candidates
    candidates = generate_parametric_candidates(deadline)
    
    # Add more diverse initial candidates
    additional_candidates = [
        [0, 1, 3, 7, 15, 31, 63],  # Mersenne-like
        [0, 1, 2, 4, 8, 16, 32, 64],  # Powers of 2
        [0, 1, 5, 25, 125, 625],  # Powers of 5
        [0, 2, 6, 18, 54, 162],  # 2 * 3^k
        [0, 1, 4, 16, 64, 256],  # Powers of 4
        [0, 1, 3, 9, 27, 81],  # Powers of 3
        [0, 2, 5, 12, 29, 70],  # Custom growth
    ]
    candidates.extend(additional_candidates)
    
    best_set = [0, 1, 4, 9, 11] # Fallback
    best_score = score_set(best_set)
    
    # 2. Continuous geometric search
    if time.time() < deadline - 25:
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
    
    # 3. Run local search on top candidates with more iterations
    # We can run multiple searches if time permits
    # Let's pick top 8 unique structures
    seen_structures = set()
    top_starts = []
    for s, c in scored_candidates:
        key = tuple(c)
        if key not in seen_structures:
            seen_structures.add(key)
            top_starts.append(c)
        if len(top_starts) >= 8:
            break
            
    # Also include the current best if not in top starts
    if tuple(best_set) not in [tuple(x) for x in top_starts]:
        top_starts.append(best_set)
        
    # Allocate time budget for local searches
    time_per_search = max(5, (deadline - 10 - time.time()) / max(1, len(top_starts)))
    
    for start in top_starts:
        if time.time() > deadline - 10:
            break
        search_deadline = min(deadline - 5, time.time() + time_per_search)
        local_best, local_score = local_search(start, search_deadline, max_iter=800)
        if local_score > best_score:
            best_score = local_score
            best_set = local_best[:]
            
    # Final polish: one more local search from the best found
    if time.time() < deadline - 15:
        final_best, final_score = local_search(best_set, deadline - 5, max_iter=1000)
        if final_score > best_score:
            best_score = final_score
            best_set = final_best[:]
            
    return best_set

# EVOLVE-BLOCK-END
