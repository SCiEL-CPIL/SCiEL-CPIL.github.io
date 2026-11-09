"""Rewritten program with stronger algorithm for sumset/difference-set exponent optimization."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from math import log

def _compute_score(U):
    """Compute the score for a given set U using NumPy for speed."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    if len(U) == 0:
        return 0.0
    if 0 not in U:
        return 0.0
    n = len(U)
    if n == 0:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    
    # Compute U+U and U-U using broadcasting
    sumset = np.unique(U[:, None] + U[None, :])
    diffset = np.unique(U[:, None] - U[None, :])
    
    len_sum = len(sumset)
    len_diff = len(diffset)
    
    if len_sum == 0:
        return 0.0
        
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
    
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
    
    score = 1 + log(ratio) / denominator
    return score

def _compute_score_fast(U_list):
    """Faster score computation for local search - assumes U_list is sorted and contains 0."""
    if not U_list or 0 not in U_list:
        return 0.0
    n = len(U_list)
    if n == 0:
        return 0.0
    max_u = U_list[-1]
    if max_u == 0:
        return 1.0
    
    U = np.array(U_list, dtype=np.int64)
    
    # Compute U+U and U-U
    sumset = np.unique(U[:, None] + U[None, :])
    diffset = np.unique(U[:, None] - U[None, :])
    
    len_sum = len(sumset)
    len_diff = len(diffset)
    
    if len_sum == 0:
        return 0.0
        
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
    
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
    
    score = 1 + log(ratio) / denominator
    return score

def _local_search(U, max_iterations=100000, timeout=10.0):
    """Improved local search with better candidate generation."""
    start_time = time.time()
    current = sorted(set(U))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    best = list(current)
    best_score = current_score
    
    for i in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        U_set = set(current)
        max_u = max(U_set) if U_set else 0
        n = len(U_set)
        
        if n == 0 or max_u == 0:
            break
        
        # Generate candidates for addition
        candidates = set()
        
        # 1. Small perturbations around existing elements
        for x in list(U_set):
            for offset in [-3, -2, -1, 1, 2, 3, 4, 5, 7, 8, 10]:
                c = x + offset
                if c >= 0:
                    candidates.add(c)
        
        # 2. Structure-preserving: sums and diffs of pairs (limited)
        elems = list(U_set)
        if n > 30:
            # Sample from the set
            elems = random.sample(elems, 30)
        for idx_a in range(len(elems)):
            for idx_b in range(len(elems)):
                s = elems[idx_a] + elems[idx_b]
                if s <= 3 * max_u + 200:
                    candidates.add(s)
                d = abs(elems[idx_a] - elems[idx_b])
                if 0 < d <= 3 * max_u + 200:
                    candidates.add(d)
        
        # 3. Powers of 2 and 3
        p = 1
        while p <= 3 * max_u + 200:
            candidates.add(p)
            p *= 2
        p = 1
        while p <= 3 * max_u + 200:
            candidates.add(p)
            p *= 3
        p = 1
        while p <= 3 * max_u + 200:
            candidates.add(p)
            p *= 5
            
        # 4. Random candidates in range
        for _ in range(30):
            candidates.add(random.randint(0, 3 * max_u + 200))
            
        # 5. Specific "gap" candidates
        if max_u > 0:
            for k in range(1, 6):
                candidates.add(max_u + k)
                candidates.add(max_u * 2 + k)
                candidates.add(max_u * 3 + k)
            candidates.add(max_u + max_u // 2)
            candidates.add(max_u + max_u // 3)
            
        # 6. Multiples
        if max_u > 0:
            for k in [2, 3, 4, 5]:
                candidates.add(max_u * k)
                candidates.add(max_u * k + 1)
                candidates.add(max_u * k - 1)
        
        candidate_list = list(candidates)
        random.shuffle(candidate_list)
        
        improved = False
        for cand in candidate_list:
            if time.time() - start_time > timeout:
                break
            if cand in U_set:
                continue
            new_set = U_set | {cand}
            new_score = _compute_score(new_set)
            if new_score > current_score + 1e-9:
                current = sorted(new_set)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
                improved = True
                break
        
        if not improved:
            # Try removing an element
            removable = [x for x in current if x != 0]
            random.shuffle(removable)
            for cand in removable[:40]:
                if time.time() - start_time > timeout:
                    break
                new_set = U_set - {cand}
                if len(new_set) < 2:
                    continue
                new_score = _compute_score(new_set)
                if new_score > current_score + 1e-9:
                    current = sorted(new_set)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = list(current)
                    improved = True
                    break
        
        if not improved and i > 100:
            # Try a swap: remove one, add another
            if random.random() < 0.7:
                removable = [x for x in current if x != 0]
                if removable:
                    rem = random.choice(removable)
                    # Try adding a random element in a wider range
                    new_elem = random.randint(0, 3 * max_u + 300)
                    new_set = (U_set - {rem}) | {new_elem}
                    if len(new_set) >= 2 and 0 in new_set:
                        new_score = _compute_score(new_set)
                        if new_score > current_score + 1e-9:
                            current = sorted(new_set)
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best = list(current)
        
        # Occasionally try multiple additions
        if not improved and i > 200 and random.random() < 0.3:
            # Try adding 2 elements at once
            for _ in range(5):
                if time.time() - start_time > timeout:
                    break
                add1 = random.randint(0, 3 * max_u + 300)
                add2 = random.randint(0, 3 * max_u + 300)
                if add1 not in U_set and add2 not in U_set and add1 != add2:
                    new_set = U_set | {add1, add2}
                    if 0 in new_set and len(new_set) >= 2:
                        new_score = _compute_score(new_set)
                        if new_score > current_score + 1e-9:
                            current = sorted(new_set)
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best = list(current)
                            improved = True
                            break
    
    return best, best_score

def _simulated_annealing(U, max_iterations=50000, timeout=10.0):
    """Simulated annealing for global search."""
    start_time = time.time()
    current = sorted(set(U))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    best = list(current)
    best_score = current_score
    
    T = 0.1  # Initial temperature
    T_min = 0.001
    cooling = 0.995
    
    for i in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        U_set = set(current)
        max_u = max(U_set) if U_set else 0
        n = len(U_set)
        
        if n == 0 or max_u == 0:
            break
        
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add an element
            new_elem = random.randint(0, 3 * max_u + 300)
            if new_elem not in U_set:
                new_set = U_set | {new_elem}
            else:
                continue
        elif move_type < 0.7:
            # Remove an element
            removable = [x for x in current if x != 0]
            if not removable:
                continue
            rem = random.choice(removable)
            new_set = U_set - {rem}
            if len(new_set) < 2:
                continue
        else:
            # Swap
            removable = [x for x in current if x != 0]
            if not removable:
                continue
            rem = random.choice(removable)
            new_elem = random.randint(0, 3 * max_u + 300)
            new_set = (U_set - {rem}) | {new_elem}
            if len(new_set) < 2 or 0 not in new_set:
                continue
        
        new_score = _compute_score(new_set)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current = sorted(new_set)
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = list(current)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Start with known good small sets
    best_result = [0, 1, 2, 4]
    best_score = _compute_score(best_result)
    
    # Generate diverse initial sets
    initial_sets = []
    
    # 1. Powers of 2
    p2 = [0]
    v = 1
    while v <= 10000:
        p2.append(v)
        v *= 2
    initial_sets.append(p2)
    
    # 2. Powers of 3
    p3 = [0]
    v = 1
    while v <= 10000:
        p3.append(v)
        v *= 3
    initial_sets.append(p3)
    
    # 3. Mixed powers (2 and 3)
    mixed = [0, 1]
    v = 3
    while v <= 10000:
        mixed.append(v)
        v *= 2
        if v <= 10000:
            mixed.append(v)
            v *= 3
    initial_sets.append(sorted(set(mixed)))
    
    # 4. Quadratic residues modulo a prime
    for p in [101, 103, 107, 109, 113, 127, 131]:
        qr = [0]
        for i in range(1, p):
            val = (i * i) % p
            if val not in qr:
                qr.append(val)
        initial_sets.append(sorted(set(qr)))
        
    # 5. Random sparse sets
    for _ in range(8):
        n = random.randint(10, 60)
        max_val = random.randint(200, 2000)
        rand_set = {0}
        while len(rand_set) < n:
            rand_set.add(random.randint(0, max_val))
        initial_sets.append(sorted(rand_set))
        
    # 6. Dense low, sparse high
    dense_sparse = list(range(15))
    v = 30
    while v <= 2000:
        dense_sparse.append(v)
        v *= 2
    initial_sets.append(sorted(set(dense_sparse)))
    
    # 7. Fibonacci
    fib = [0, 1]
    while fib[-1] <= 10000:
        fib.append(fib[-1] + fib[-2])
    initial_sets.append(fib)
    
    # 8. Arithmetic progression with gaps
    ap = [0]
    v = 1
    while v <= 10000:
        ap.append(v)
        v += random.choice([2, 3, 4, 5])
    initial_sets.append(sorted(set(ap)))
    
    # 9. Logarithmic spacing
    log_space = [0]
    v = 1
    while v <= 10000:
        log_space.append(v)
        v = int(v * 1.5) + 1
    initial_sets.append(sorted(set(log_space)))
    
    # 10. Prime-based
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    prime_set = [0] + primes[:20]
    initial_sets.append(sorted(set(prime_set)))
    
    # Sort initial sets by score to try best ones first
    scored_sets = [(s, _compute_score(s)) for s in initial_sets]
    scored_sets.sort(key=lambda x: x[1], reverse=True)
    
    # Run local search on each initial set
    for init_set, init_score in scored_sets:
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - time.time() - 3
        if remaining < 3:
            break
            
        # Allocate time based on potential
        timeout = min(20.0, remaining * 0.6)
        improved, score = _local_search(init_set, max_iterations=50000, timeout=timeout)
        
        if score > best_score:
            best_score = score
            best_result = improved
    
    # Run simulated annealing on the best result
    if time.time() < deadline - 10:
        sa_timeout = min(15.0, deadline - time.time() - 2)
        if sa_timeout > 2:
            sa_result, sa_score = _simulated_annealing(best_result, max_iterations=30000, timeout=sa_timeout)
            if sa_score > best_score:
                best_score = sa_score
                best_result = sa_result
    
    # Final polish on the best result
    if time.time() < deadline - 5:
        polish_timeout = min(10.0, deadline - time.time() - 1)
        if polish_timeout > 1:
            polished, polished_score = _local_search(best_result, max_iterations=20000, timeout=polish_timeout)
            if polished_score > best_score:
                best_score = polished_score
                best_result = polished
    
    # Final check
    final_score = _compute_score(best_result)
    if final_score < 1.0:
        best_result = [0, 1, 2, 4]
        
    return best_result
# EVOLVE-BLOCK-END
