import time
from math import log, floor
import random
import numpy as np
from itertools import combinations

def score_set(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
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

def digit_base_set(base, digits_per_pos, max_positions):
    """
    Generate a set using a base-b expansion with variable digit sets.
    U = { sum_{k=0}^{max_positions-1} d_k * base^k } for all combinations of digits.
    We ensure 0 is included (all digits 0).
    To keep size manageable, we use a small number of positions and digits.
    """
    U = set()
    U.add(0)
    
    # Precompute powers
    powers = [1]
    for k in range(1, max_positions):
        powers.append(powers[-1] * base)
    
    # Generate all combinations of digits
    # This is product of digit sets
    # We use recursion or iterative construction
    current = [0]
    for k in range(max_positions):
        new_current = []
        for val in current:
            for d in digits_per_pos[k]:
                new_val = val + d * powers[k]
                new_current.append(new_val)
        current = new_current
    
    for val in current:
        if val >= 0:
            U.add(val)
    
    return sorted(list(U))

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
            # Also try 2*a-b style moves
            c = 2*a - b
            if 0 < c <= limit_val and c not in U_set:
                candidates.add(c)
            # Try a+b-c for another element
            for c_val in U:
                if c_val == a or c_val == b:
                    continue
                t = a + b - c_val
                if 0 < t <= limit_val and t not in U_set:
                    candidates.add(t)
            
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
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
    max_no_improvement = 30
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.1:
            # Expand limit if we are growing
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                # Try a few random candidates
                random.shuffle(candidates)
                for c in candidates[:10]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.7:
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
                # Pick a random candidate
                random.shuffle(candidates)
                for new_val in candidates[:10]:
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
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
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
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Structured Candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Digit Base Search (New Strategy)
    # Explore sets generated by base-b expansions with varying digit sets
    if time.time() < deadline - 60:
        bases = [2, 3, 4, 5]
        max_pos = [3, 4, 5]
        
        for b in bases:
            if time.time() > deadline - 60:
                break
            for pos in max_pos:
                if time.time() > deadline - 60:
                    break
                # Try different digit sets for each position
                # Simple heuristic: digits 0..b-1 for all positions gives full range
                # But we can restrict digits to keep size smaller or structure specific
                # Let's try full digit set first (this is just 0..b^pos - 1, which is a contiguous range, likely bad)
                # Better: sparse digit sets
                # Example: digits {0, 1} for all positions -> {0, 1, b, b+1, ...}
                # This creates a "digital" set
                
                # Let's try a few specific digit patterns
                patterns = [
                    [[0, 1]] * pos,  # Binary-like
                    [[0, 1, 2]] * pos, # Ternary-like
                    [[0, b-1]] * pos, # Extremes
                ]
                
                for pat in patterns:
                    U = digit_base_set(b, pat, pos)
                    if len(U) < 2 or len(U) > 100: # Keep size manageable
                        continue
                    s = score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = list(U)

    # 3. Exhaustive Search for Small n
    def exhaustive_search(n, max_val):
        """Find best set of size n (including 0) with elements <= max_val."""
        best_s = -1
        best_c = None
        range_vals = range(1, max_val + 1)
        for comb in combinations(range_vals, n - 1):
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_s:
                best_s = s
                best_c = U
        return best_s, best_c

    if time.time() < deadline - 50:
        s, c = exhaustive_search(3, 50)
        if s > best_score:
            best_score = s
            best_set = c
        
        s, c = exhaustive_search(4, 40)
        if s > best_score:
            best_score = s
            best_set = c
            
        s, c = exhaustive_search(5, 30)
        if s > best_score:
            best_score = s
            best_set = c
            
        if time.time() < deadline - 30:
            s, c = exhaustive_search(6, 20)
            if s > best_score:
                best_score = s
                best_set = c

    # 4. Geometric Search
    b_candidates = [b_int / 10.0 for b_int in range(12, 101)]
    best_b = None
    best_m = None
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 40:
            break
        for m in range(2, 21):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                if s > best_score:
                    best_score = s
                    best_set = U

    # 5. Dynamic Local Search
    seeds = [best_set]
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 500
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            if len(seeds) < 5:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 6. Final Polish
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        max_iterations = 1000
        while improved and time.time() < deadline - 1 and iteration < max_iterations:
            improved = False
            iteration += 1
            n = len(current)
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                for delta in range(-20, 21):
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
            
            if not improved and n < 30 and time.time() < deadline - 1:
                max_val = max(current)
                for new_val in range(max(1, max_val - 20), max_val + 21):
                    if new_val <= 0:
                        continue
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            if not improved and n > 3 and time.time() < deadline - 1:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
