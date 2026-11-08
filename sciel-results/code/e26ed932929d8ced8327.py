import time
from math import log, floor
import random
import numpy as np

def score_set(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
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

def score_set_fast(U):
    """Faster scoring using numpy for larger sets."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    if n <= 50:
        return score_set(U)
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all sums U+U
    sums = np.add.outer(U_arr, U_arr)
    unique_sums = np.unique(sums)
    sum_count = len(unique_sums)
    
    # Compute all differences U-U
    diffs = np.subtract.outer(U_arr, U_arr)
    unique_diffs = np.unique(diffs)
    diff_count = len(unique_diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = U_arr[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def generate_structured_sets():
    """Generate various structured candidate sets."""
    candidates = []
    
    # Powers of 2
    for m in range(2, 20):
        U = [2**k for k in range(m)]
        if 0 not in U:
            U.insert(0, 0)
        candidates.append(U)
    
    # Powers of 3
    for m in range(2, 15):
        U = [3**k for k in range(m)]
        if 0 not in U:
            U.insert(0, 0)
        candidates.append(U)
    
    # Powers of 4
    for m in range(2, 12):
        U = [4**k for k in range(m)]
        if 0 not in U:
            U.insert(0, 0)
        candidates.append(U)
    
    # Geometric with base 2.5, 3.5, etc.
    for base in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        for m in range(2, 15):
            U = [floor(base**k) for k in range(m)]
            U = [x for x in U if x >= 0]
            if 0 not in U:
                U.insert(0, 0)
            U = sorted(set(U))
            if len(U) >= 2:
                candidates.append(U)
    
    # Mixed bases: alternate between two bases
    for b1 in [2, 3, 4]:
        for b2 in [3, 4, 5, 6]:
            if b1 == b2:
                continue
            for m in range(2, 12):
                U = []
                for k in range(m):
                    if k % 2 == 0:
                        U.append(b1 ** (k // 2))
                    else:
                        U.append(b2 ** ((k - 1) // 2))
                if 0 not in U:
                    U.insert(0, 0)
                U = sorted(set(U))
                if len(U) >= 2:
                    candidates.append(U)
    
    # Digit-based constructions: U = {sum of digits * place values}
    # For base b with digits {0, 1}, this is just powers of b
    # For base b with digits {0, 1, 2}, we get more structure
    for b in [3, 4, 5]:
        for num_digits in range(2, 6):
            # Generate all numbers with digits 0 or 1 in base b
            U = []
            for mask in range(2 ** num_digits):
                val = 0
                for d in range(num_digits):
                    if mask & (1 << d):
                        val += b ** d
                U.append(val)
            U = sorted(set(U))
            if 0 not in U:
                U.insert(0, 0)
            if len(U) >= 2:
                candidates.append(U)
    
    # Triangular numbers
    for m in range(2, 30):
        U = [k * (k + 1) // 2 for k in range(m)]
        if 0 not in U:
            U.insert(0, 0)
        candidates.append(U)
    
    # Squares
    for m in range(2, 25):
        U = [k * k for k in range(m)]
        if 0 not in U:
            U.insert(0, 0)
        candidates.append(U)
    
    # Cubes
    for m in range(2, 15):
        U = [k * k * k for k in range(m)]
        if 0 not in U:
            U.insert(0, 0)
        candidates.append(U)
    
    return candidates

def local_search(U, deadline, max_val_limit=2000, max_n_limit=50, use_fast_scoring=True):
    """
    Local search with add/remove/replace moves.
    """
    U = sorted(set(U))
    score_func = score_set_fast if use_fast_scoring else score_set
    current_score = score_func(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to ADD an element
            if len(U) < max_n_limit:
                # Generate candidates from existing structure
                U_set = set(U)
                candidates = set()
                
                for a in U:
                    for b in U:
                        s = a + b
                        d = abs(a - b)
                        if 0 < s <= max_val_limit and s not in U_set:
                            candidates.add(s)
                        if 0 < d <= max_val_limit and d not in U_set:
                            candidates.add(d)
                
                # Also try random values near existing elements
                for a in U:
                    for delta in random.sample(range(-10, 11), min(10, 21)):
                        if delta == 0:
                            continue
                        nv = a + delta
                        if 0 < nv <= max_val_limit and nv not in U_set:
                            candidates.add(nv)
                
                # Also try some larger jumps
                max_u = max(U)
                for _ in range(5):
                    nv = random.randint(1, max(max_val_limit, max_u * 2))
                    if nv not in U_set:
                        candidates.add(nv)
                
                if candidates:
                    cand_list = list(candidates)
                    random.shuffle(cand_list)
                    for c in cand_list[:15]:
                        new_U = U + [c]
                        new_U.sort()
                        new_score = score_func(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
                            break
                            
        elif move_type < 0.6:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_func(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        
        else:
            # Try to REPLACE an element
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                U_set = set(U)
                
                # Try nearby values
                tried = 0
                for delta in random.sample(range(-30, 31), min(30, 61)):
                    if delta == 0 or tried >= 15:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0 or new_val in U_set:
                        continue
                    new_set = set(U)
                    new_set.discard(old_val)
                    new_set.add(new_val)
                    new_U = sorted(new_set)
                    new_score = score_func(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
                    tried += 1
                
                if not improved:
                    # Try a random new value
                    for _ in range(5):
                        new_val = random.randint(1, max(max_val_limit, max(U) * 2))
                        if new_val in U_set:
                            continue
                        new_set = set(U)
                        new_set.discard(old_val)
                        new_set.add(new_val)
                        new_U = sorted(new_set)
                        new_score = score_func(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
                            break
                
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
            
    return best_local_score, best_local_set

def final_polish(U, deadline):
    """Final high-precision polish pass."""
    U = sorted(set(U))
    current_score = score_set_fast(U)
    improved = True
    iteration = 0
    
    while improved and time.time() < deadline - 1 and iteration < 200:
        improved = False
        iteration += 1
        n = len(U)
        
        # Try replacing each element with nearby values
        for i in range(1, n):
            if time.time() > deadline - 1:
                break
            old_val = U[i]
            for delta in range(-5, 6):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                new_set = [U[j] if j != i else new_val for j in range(n)]
                new_set = sorted(set(new_set))
                if len(new_set) != n:
                    continue
                new_score = score_set_fast(new_set)
                if new_score > current_score:
                    U = new_set
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        # Try adding an element
        if not improved and n < 30 and time.time() < deadline - 1:
            max_val = max(U)
            for new_val in range(max(1, max_val - 10), max_val + 11):
                if new_val in U:
                    continue
                new_set = sorted(U + [new_val])
                new_score = score_set_fast(new_set)
                if new_score > current_score:
                    U = new_set
                    current_score = new_score
                    improved = True
                    break
        
        # Try removing an element
        if not improved and n > 3 and time.time() < deadline - 1:
            for i in range(1, n):
                new_set = [U[j] for j in range(n) if j != i]
                new_score = score_set_fast(new_set)
                if new_score > current_score:
                    U = new_set
                    current_score = new_score
                    improved = True
                    break
                    
    return U, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    best_set = [0, 1, 2]
    best_score = score_set(best_set)
    
    # 1. Evaluate structured candidates
    structured = generate_structured_sets()
    for cand in structured:
        if time.time() > deadline - 60:
            break
        s = score_set_fast(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # 2. Local search from best structured set and several other seeds
    seeds = [best_set]
    # Add a few diverse seeds
    for cand in structured[:5]:
        if cand != best_set and cand not in seeds:
            seeds.append(cand)
    
    max_val_limit = 3000
    max_n_limit = 40
    
    restarts = 0
    while time.time() < deadline - 15:
        restarts += 1
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        local_score, local_set = local_search(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            if len(seeds) < 8:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set
    
    # 3. Final polish
    if time.time() < deadline - 2:
        best_set, best_score = final_polish(best_set, deadline)
    
    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
