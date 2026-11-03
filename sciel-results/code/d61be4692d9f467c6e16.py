import time
import random
from math import log, log2
import numpy as np

def fast_score(U):
    """
    Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences (i < j) -> positive diffs
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

def golomb_ruler_scores(n):
    """
    Score known optimal Golomb rulers for sizes 4..12.
    """
    rulers = {
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 25],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 41],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
        11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58],
        12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67],
    }
    results = []
    for n, ruler in rulers.items():
        s = fast_score(ruler)
        if s > 0:
            results.append((s, ruler[:]))
    return results

def geometric_set(base, m):
    U = [0]
    val = 1
    for _ in range(m):
        val = int(val * base)
        if val <= 0:
            break
        if val not in U:
            U.append(val)
    return sorted(set(U))

def structured_candidates():
    """
    Generate diverse structured candidate sets.
    """
    cands = []
    # Golomb rulers
    cands.extend([r for _, r in golomb_ruler_scores(12)])
    
    # Geometric sequences with various bases
    for base in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 5.0]:
        for m in range(4, 12):
            U = geometric_set(base, m)
            if len(U) >= 4:
                cands.append(U)
    
    # Mixed base
    for b1, b2 in [(2,3),(3,5),(2,5),(3,7),(4,7)]:
        U = [0]
        for k in range(1, 10):
            if k % 2 == 1:
                U.append(b1 ** k)
            else:
                U.append(b2 ** (k // 2))
        cands.append(sorted(set(U)))
    
    # Arithmetic progressions with gaps
    for a in [1,2,3]:
        for d in [3,4,5,7,8,11]:
            U = [0]
            for k in range(1, 8):
                U.append(a + k * d)
            cands.append(sorted(set(U)))
    
    # Handcrafted
    handcrafted = [
        [0, 1, 3, 7],
        [0, 1, 5, 8],
        [0, 2, 5, 9],
        [0, 1, 4, 8, 13],
        [0, 1, 5, 9, 14],
        [0, 2, 7, 12, 18],
        [0, 1, 4, 10, 16],
        [0, 3, 7, 12, 18],
        [0, 1, 6, 11, 17],
        [0, 2, 8, 14, 20],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 9, 27],
        [0, 2, 6, 18],
        [0, 1, 4, 9, 16, 25],
    ]
    cands.extend(handcrafted)
    
    # Deduplicate
    seen = set()
    unique = []
    for c in cands:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique

def simulated_annealing(U, deadline, max_iter=5000):
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
        max_val = max(current)
        r = random.random()
        if r < 0.30:
            move_type = 'perturb'
        elif r < 0.45:
            move_type = 'add'
        elif r < 0.55:
            move_type = 'remove'
        elif r < 0.70:
            move_type = 'swap'
        elif r < 0.80:
            move_type = 'scale'
        else:
            move_type = 'replace'
        
        new_set = current[:]
        
        if move_type == 'perturb' and n > 1:
            i = random.randint(1, n-1)
            old_val = current[i]
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
        
        elif move_type == 'add' and n < 14:
            max_val = max(current)
            if random.random() < 0.3:
                new_val = max_val + random.randint(1, max(2, int(max_val * 0.15)))
            elif random.random() < 0.6:
                new_val = random.randint(1, max_val)
            else:
                new_val = random.randint(max_val + 1, max_val * 2)
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
            factor = random.choice([0.85, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3])
            new_set = [0] + [max(1, int(v * factor)) for v in current[1:]]
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
        
        elif move_type == 'replace' and n > 1:
            i = random.randint(1, n-1)
            candidates_val = set()
            for f in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
                candidates_val.add(max(1, int(max_val * f)))
            for v in current[1:]:
                for d in [-3, -2, -1, 1, 2, 3, 5, 7]:
                    candidates_val.add(max(1, v + d))
            for p in [2, 3, 5]:
                val = 1
                for _ in range(min(15, max_val + 1)):
                    if val > max_val * 1.5:
                        break
                    candidates_val.add(val)
                    val *= p
            candidates_val.discard(0)
            candidates_val.discard(current[i])
            if not candidates_val:
                continue
            new_val = random.choice(list(candidates_val))
            new_set[i] = new_val
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
                if delta < 0.5 and random.random() < np.exp(-delta / max(T, 0.001)):
                    current = new_set
                    current_score = new_score
        
        T *= cooling
        if T < 0.001:
            T = 0.001
            
    return best, best_score

def high_precision_polish(U, deadline):
    """
    Final high-precision polish: exhaustive single-element perturbations,
    pair swaps, and element replacement with wider ranges.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score = current_score
    
    n = len(current)
    max_val = max(current)
    
    # Phase 1: Single-element perturbations with adaptive range
    step_range = max(5, min(20, int(max_val * 0.03)))
    for i in range(1, n):
        if time.time() > deadline:
            break
        old_val = current[i]
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
    
    # Phase 2: Try replacing each element with values at various fractions
    if time.time() < deadline - 5:
        for i in range(1, n):
            if time.time() > deadline:
                break
            old_val = current[i]
            for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.3, 1.5, 2.0]:
                new_val = max(1, int(max_val * frac))
                if new_val == old_val:
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
    
    # Phase 3: Pair swaps
    if time.time() < deadline - 2 and n > 2:
        for i in range(1, n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                test_set = current[:]
                test_set[i], test_set[j] = test_set[j], test_set[i]
                test_set = sorted(test_set)
                s = fast_score(test_set)
                if s > best_score:
                    best_score = s
                    best = test_set[:]
                    current = test_set[:]
    
    # Phase 4: Add/remove one element
    if time.time() < deadline - 1 and n > 3:
        # Try removing each element
        for i in range(1, n):
            if time.time() > deadline:
                break
            test_set = current[:]
            test_set.pop(i)
            s = fast_score(test_set)
            if s > best_score:
                best_score = s
                best = test_set[:]
                current = test_set[:]
    
    if time.time() < deadline - 1 and n < 13:
        # Try adding elements
        max_val = max(current)
        for new_val in range(1, max_val + 10):
            if time.time() > deadline:
                break
            if new_val in current:
                continue
            test_set = current[:] + [new_val]
            test_set = sorted(set(test_set))
            s = fast_score(test_set)
            if s > best_score:
                best_score = s
                best = test_set[:]
                current = test_set[:]
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    random.seed(42)
    
    # 1. Evaluate all structured candidates
    candidates = structured_candidates()
    
    # Add random sparse sets
    for _ in range(30):
        n_elem = random.randint(4, 10)
        max_val = random.randint(20, 300)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
    
    # Geometric sets with specific ratios
    for ratio in [1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0]:
        U = [0]
        val = 1
        for _ in range(12):
            U.append(val)
            val = int(val * ratio)
            if val > 5000:
                break
        candidates.append(U)
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # Score all candidates
    scored_candidates = []
    for U in unique_candidates:
        U_clean = sorted(set(U))
        if 0 not in U_clean or len(U_clean) < 2:
            continue
        s = fast_score(U_clean)
        if s > 0:
            scored_candidates.append((s, U_clean))
    
    scored_candidates.sort(reverse=True)
    
    best_set = scored_candidates[0][1][:] if scored_candidates else [0, 1, 4, 9, 11]
    best_score = scored_candidates[0][0] if scored_candidates else fast_score(best_set)
    
    # 2. Run SA on top candidates
    sa_budget = deadline - 45  # Leave 45s for polish
    top_k = min(10, len(scored_candidates))
    for idx in range(top_k):
        if time.time() > sa_budget:
            break
        score, U = scored_candidates[idx]
        remaining = sa_budget - time.time()
        if idx < 3:
            alloc = max(8, min(remaining, 25))
        else:
            alloc = max(5, min(remaining, 15))
        refined, refined_score = simulated_annealing(U, time.time() + alloc)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined[:]
    
    # 3. Run SA on the global best set itself
    if time.time() < deadline - 40:
        remaining = deadline - 40 - time.time()
        if remaining > 5:
            refined, refined_score = simulated_annealing(best_set, time.time() + min(remaining, 20))
            if refined_score > best_score:
                best_score = refined_score
                best_set = refined[:]
    
    # 4. Final high-precision polish
    if time.time() < deadline - 10:
        refined, refined_score = high_precision_polish(best_set, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined[:]
    
    # 5. One more SA round on the polished result if time allows
    if time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        if remaining > 3:
            refined, refined_score = simulated_annealing(best_set, time.time() + min(remaining, 10))
            if refined_score > best_score:
                best_score = refined_score
                best_set = refined[:]
    
    # 6. Final polish
    if time.time() < deadline - 2:
        refined, refined_score = high_precision_polish(best_set, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined[:]
    
    return best_set

if __name__ == "__main__":
    print(f"Best set: {construct()}")
