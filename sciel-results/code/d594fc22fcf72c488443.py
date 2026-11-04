import time
from math import log, floor, log1p
import random
import numpy as np

def score_set(U):
    """Compute the exact score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums (i <= j)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(int(arr[i] + arr[j]))
    
    # Compute all pairwise differences (i < j), add both signs
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = int(arr[j] - arr[i])
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def compute_sum_diff_counts(U):
    """Return (sum_count, diff_count, max_u) for a sorted set U containing 0."""
    n = len(U)
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
    return len(sums), len(diffs), max(U)

def fast_score_estimate(U, candidate):
    """
    Fast estimate of score change when adding 'candidate' to U.
    Returns estimated new score without full recomputation.
    """
    U = sorted(U)
    n = len(U)
    candidate = int(candidate)
    
    if candidate in U:
        return None
    
    # Count existing sums
    existing_sums = set()
    for i in range(n):
        for j in range(i, n):
            existing_sums.add(U[i] + U[j])
    
    # Count new sums involving candidate
    new_sums_count = 0
    for i in range(n):
        s = U[i] + candidate
        if s not in existing_sums:
            new_sums_count += 1
    if candidate + candidate not in existing_sums:
        new_sums_count += 1
    
    sum_count = len(existing_sums) + new_sums_count
    
    # Count existing diffs
    existing_diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = U[j] - U[i]
            existing_diffs.add(d)
            existing_diffs.add(-d)
    existing_diffs.add(0)
    
    # Count new diffs involving candidate
    new_diffs_count = 0
    for i in range(n):
        d = abs(candidate - U[i])
        if d > 0:
            if d not in existing_diffs:
                new_diffs_count += 2
    
    diff_count = len(existing_diffs) + new_diffs_count
    
    if sum_count == 0 or diff_count <= sum_count:
        return None
    
    max_u = max(max(U), candidate)
    D = 2 * max_u + 1
    if D <= 1:
        return None
    
    ratio = diff_count / sum_count
    return 1 + log(ratio) / log(D)

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included and all are nonnegative distinct integers."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    """
    U_set = set(U)
    candidates = set()
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            
    # Also add some random candidates for diversity
    max_val = max(U) if U else 0
    for _ in range(min(20, max_val // 2)):
        r = random.randint(1, min(limit_val, max_val + 50))
        if r not in U_set:
            candidates.add(r)
            
    return list(candidates)

def compute_marginal_gain(U, candidate):
    """
    Compute the marginal effect of adding 'candidate' to U on the score components.
    Returns (delta_sum_count, delta_diff_count, new_max_u).
    """
    U_sorted = sorted(U + [candidate])
    n = len(U_sorted)
    
    existing_sums = set()
    for i in range(n):
        for j in range(i, n):
            existing_sums.add(U_sorted[i] + U_sorted[j])
    
    existing_diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = U_sorted[j] - U_sorted[i]
            existing_diffs.add(d)
            existing_diffs.add(-d)
    existing_diffs.add(0)
    
    orig_sums, orig_diffs, orig_max = compute_sum_diff_counts(U)
    
    delta_sums = len(existing_sums) - orig_sums
    delta_diffs = len(existing_diffs) - orig_diffs
    new_max = max(orig_max, candidate)
    
    return delta_sums, delta_diffs, new_max

def polynomial_support_analysis(U):
    """
    Analyze the polynomial P(x) = sum x^{u} for u in U.
    Support of P(x)^2 gives sumset, support of P(x)P(x^{-1}) gives difference set.
    Return (sum_support, diff_support) as sets of integers.
    """
    U_set = set(U)
    max_u = max(U)
    min_u = min(U)
    
    # Sum support: all u1 + u2 for u1, u2 in U
    sum_support = set()
    for u1 in U:
        for u2 in U:
            sum_support.add(u1 + u2)
    
    # Difference support: all u1 - u2 for u1, u2 in U
    diff_support = set()
    for u1 in U:
        for u2 in U:
            diff_support.add(u1 - u2)
    
    return sum_support, diff_support

def polynomial_guided_search(U, deadline, max_val_limit=800, max_n_limit=35):
    """
    Search guided by polynomial support analysis.
    Uses the algebraic structure to prefer elements that maximize
    the difference set support relative to the sum set support.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial polynomial analysis
    sum_support, diff_support = polynomial_support_analysis(U)
    
    no_improvement_count = 0
    max_no_improvement = 40
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Generate candidates based on polynomial structure
        candidates = set()
        max_val = max(U)
        
        # Candidates from sum support that are not in U
        for s in sum_support:
            if 0 < s <= max_val_limit and s not in U:
                candidates.add(s)
        
        # Candidates from difference support (absolute values) that are not in U
        for d in diff_support:
            ad = abs(d)
            if 0 < ad <= max_val_limit and ad not in U:
                candidates.add(ad)
        
        # Geometric expansion candidates
        for base in [1.5, 2, 2.5, 3]:
            for k in range(1, 10):
                val = floor(max_val * (base ** k)) if max_val > 0 else k
                if 0 < val <= max_val_limit:
                    candidates.add(val)
        
        # Random candidates
        for _ in range(30):
            val = random.randint(1, min(max_val_limit, max_val * 2 + 20))
            candidates.add(val)
        
        candidates = [c for c in candidates if c not in U]
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                # Score candidates by polynomial-guided heuristic
                scored_candidates = []
                for c in candidates[:25]:
                    if c in U:
                        continue
                    try:
                        ds, dd, nm = compute_marginal_gain(U, c)
                        # Heuristic: prefer large diff growth relative to sum growth
                        # Penalize large max values (affects D in denominator)
                        benefit = log1p(max(0, dd)) - 0.6 * log1p(max(0, ds))
                        if nm > max_val * 1.8:
                            benefit *= 0.75
                        scored_candidates.append((benefit, c))
                    except:
                        pass
                
                scored_candidates.sort(key=lambda x: -x[0])
                for _, c in scored_candidates[:6]:
                    new_U = sorted(U + [c])
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        sum_support, diff_support = polynomial_support_analysis(U)
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 3:
                removable = [x for x in U if x != 0]
                if removable:
                    best_remove = None
                    best_remove_score = current_score
                    for c in random.sample(removable, min(6, len(removable))):
                        new_U = [x for x in U if x != c]
                        new_score = score_set(new_U)
                        if new_score > best_remove_score:
                            best_remove_score = new_score
                            best_remove = c
                    if best_remove is not None and best_remove_score > current_score:
                        U = [x for x in U if x != best_remove]
                        current_score = best_remove_score
                        improved = True
                        sum_support, diff_support = polynomial_support_analysis(U)
        else:
            # Try to REPLACE an element
            if len(U) > 2 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                random.shuffle(candidates)
                for new_val in candidates[:12]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U = sorted(set(new_U))
                    if len(new_U) != len(U):
                        continue
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        sum_support, diff_support = polynomial_support_analysis(U)
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def modular_greedy_construct(p, n_elements, deadline):
    """
    Greedy construction restricted to multiples of prime p.
    """
    random.seed(42)
    
    V = [0, 1]
    best_score = score_set([0, p])
    
    for _ in range(n_elements - 2):
        if time.time() > deadline - 10:
            break
            
        max_existing = max(V) if V else 0
        
        candidates = set()
        for base in [1.5, 2, 2.5, 3]:
            for k in range(1, 8):
                val = floor(max_existing * (base ** k)) if max_existing > 0 else k
                if val > 0:
                    candidates.add(val)
        
        for _ in range(50):
            val = random.randint(1, max_existing * 3 + 10)
            if val > 0:
                candidates.add(val)
        
        for val in range(1, min(50, max_existing + 1)):
            candidates.add(val)
        
        candidates = [c for c in candidates if c not in V]
        
        best_candidate = None
        best_local_score = best_score
        
        random.shuffle(candidates)
        for c in candidates[:30]:
            if time.time() > deadline - 10:
                break
            new_V = V + [c]
            new_V.sort()
            U = [v * p for v in new_V]
            s = score_set(U)
            if s > best_local_score:
                best_local_score = s
                best_candidate = c
        
        if best_candidate is not None:
            V.append(best_candidate)
            V.sort()
            best_score = best_local_score
        else:
            if random.random() < 0.5:
                V.append(1 if 1 not in V else 2)
                V.sort()
            break
    
    U = [v * p for v in V]
    return U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    random.seed(42)

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Structured Candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        [0, 1, 2, 4, 8, 16, 32, 64, 128],
        [0, 1, 3, 7, 15, 31, 63, 127],
        [0, 1, 4, 10, 20, 35, 56, 84],
        [0, 1, 3, 4, 9, 10, 12, 13],
        [0, 1, 4, 10, 18, 28, 40, 54],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        [0, 1, 3, 9, 27, 81, 243],
        [0, 1, 5, 14, 30, 55, 91],
        [0, 1, 4, 10, 20, 35, 56, 84, 120],
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121],
        [0, 1, 2, 4, 7, 11, 16, 22, 29, 37],
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Geometric Search
    b_candidates = [b_int / 10.0 for b_int in range(12, 80)]
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 45:
            break
        for m in range(2, 20):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                if s > best_score:
                    best_score = s
                    best_set = U

    # 3. Modular Greedy Construction
    for p in [2, 3, 5, 7, 11]:
        if time.time() > deadline - 35:
            break
        mod_set, mod_score = modular_greedy_construct(p, 15, deadline)
        if mod_score > best_score:
            best_score = mod_score
            best_set = mod_set

    # 4. Polynomial-Guided Search with multiple restarts
    seeds = [best_set]
    for cand in structured_candidates[:5]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 600
    max_n_limit = 32
    
    restarts = 0
    while time.time() < deadline - 30:
        restarts += 1
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        local_score, local_set = polynomial_guided_search(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            if len(seeds) < 8:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 5. Final High-Precision Polish
    if time.time() < deadline - 3:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        polish_deadline = deadline - 1
        
        while improved and time.time() < polish_deadline and iteration < 1500:
            improved = False
            iteration += 1
            n = len(current)
            
            # Phase 1: Fine-grained replacement (delta -25 to 25)
            for i in range(1, n):
                if time.time() > polish_deadline:
                    break
                old_val = current[i]
                for delta in range(-25, 26):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Phase 2: Add element near max or in gaps
            if not improved and n < 30 and time.time() < polish_deadline:
                max_val = max(current)
                try_vals = set()
                for v in range(max(1, max_val - 40), max_val + 41):
                    try_vals.add(v)
                for v in range(1, max_val, max(1, max_val // 30)):
                    try_vals.add(v)
                
                for new_val in sorted(try_vals):
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            # Phase 3: Remove element with least impact
            if not improved and n > 3 and time.time() < polish_deadline:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            # Phase 4: Pairwise swap
            if not improved and n >= 4 and time.time() < polish_deadline:
                for i in range(1, n):
                    if time.time() > polish_deadline:
                        break
                    for j in range(i + 1, n):
                        if time.time() > polish_deadline:
                            break
                        old_i = current[i]
                        old_j = current[j]
                        for di in range(-8, 9):
                            for dj in range(-8, 9):
                                if di == 0 and dj == 0:
                                    continue
                                new_i = old_i + di
                                new_j = old_j + dj
                                if new_i <= 0 or new_j <= 0:
                                    continue
                                if new_i == new_j:
                                    continue
                                new_set = list(current)
                                new_set[i] = new_i
                                new_set[j] = new_j
                                new_set = sorted(set(new_set))
                                if len(new_set) != n:
                                    continue
                                new_score = score_set(new_set)
                                if new_score > current_score:
                                    current = new_set
                                    current_score = new_score
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
