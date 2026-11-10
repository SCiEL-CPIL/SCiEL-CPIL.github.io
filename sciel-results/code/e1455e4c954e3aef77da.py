import time
import random
import math

def _calc_score(A):
    """Compute max(|A+A|, |A*A|) for a set of distinct positive integers."""
    n = len(A)
    sumset = set()
    prodset = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sumset.add(ai + aj)
            prodset.add(ai * aj)
    return max(len(sumset), len(prodset))

def _generate_smooth_candidates(max_val, primes):
    """Generate smooth numbers with given prime factors <= max_val."""
    vals = set()
    def gen(idx, current_val):
        if current_val > max_val:
            return
        vals.add(current_val)
        if idx < len(primes):
            p = primes[idx]
            v = current_val
            while v * p <= max_val:
                v *= p
                gen(idx, v)
            gen(idx + 1, current_val)
    gen(0, 1)
    return sorted(list(vals))

def _generate_geometric_perturbed(base, exponents, perturbations, max_val=10000):
    """Generate set {base^e_i * k_i} with small coprime perturbations."""
    result = set()
    for e, k in zip(exponents, perturbations):
        val = int(round(base ** e * k))
        if 1 <= val <= max_val:
            result.add(val)
    return sorted(result)

def _local_search_swap(A, pool, timeout=10.0):
    """Local search by swapping elements in A with elements in pool."""
    A = sorted(A)
    A_set = set(A)
    current_score = _calc_score(A)
    best_A = list(A)
    best_score = current_score
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        improved = False
        for i in range(len(A)):
            old_val = A[i]
            best_new_val = old_val
            best_new_score = current_score
            
            for new_val in pool:
                if new_val in A_set or new_val == old_val:
                    continue
                
                temp_A = A[:i] + [new_val] + A[i+1:]
                new_score = _calc_score(temp_A)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_val = new_val
            
            if best_new_val != old_val:
                A[i] = best_new_val
                A.sort()
                A_set = set(A)
                current_score = best_new_score
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_A = list(A)
        
        if not improved:
            break
            
    return best_A, best_score

def _simulated_annealing(A, pool, timeout=10.0):
    """SA with moves restricted to pool or small perturbations."""
    A = sorted(A)
    A_set = set(A)
    current_score = _calc_score(A)
    best_A = list(A)
    best_score = current_score
    start_time = time.time()
    n = len(A)
    
    start_temp = 50.0
    end_temp = 1.0
    max_iters = 5000
    
    for i in range(max_iters):
        if time.time() - start_time > timeout:
            break
            
        T = start_temp * (end_temp / start_temp) ** (i / max_iters)
        
        idx = random.randint(0, n - 1)
        old_val = A[idx]
        
        r = random.random()
        if r < 0.6:
            valid_pool = [x for x in pool if x not in A_set]
            if not valid_pool:
                continue
            new_val = random.choice(valid_pool)
        elif r < 0.8:
            delta = random.randint(-10, 10)
            new_val = old_val + delta
            if new_val < 1 or new_val > 10000 or new_val in A_set:
                continue
        else:
            delta = random.randint(-100, 100)
            new_val = old_val + delta
            if new_val < 1 or new_val > 10000 or new_val in A_set:
                continue
            
        temp_A = A[:idx] + [new_val] + A[idx+1:]
        new_score = _calc_score(temp_A)
        delta_e = new_score - current_score
        
        if delta_e < 0 or (T > 0 and random.random() < math.exp(-delta_e / T)):
            A[idx] = new_val
            A.sort()
            A_set = set(A)
            current_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_A = list(A)
                
    return best_A, best_score

def _search_geometric_structures(timeout=20.0):
    """Search for geometric progressions with multiplicative perturbations."""
    start_time = time.time()
    best_A = None
    best_score = float('inf')
    
    # Try various bases and exponent ranges
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]:
        if time.time() - start_time > timeout:
            break
        
        # Find max exponent such that base^e <= 10000
        max_e = 0
        while base ** (max_e + 1) <= 10000:
            max_e += 1
        
        if max_e < 19:
            continue
        
        # Try different exponent spacings
        for start_e in range(0, min(5, max_e - 19)):
            for step in [1, 2, 3]:
                exponents = [start_e + i * step for i in range(20)]
                if max(exponents) > max_e:
                    continue
                
                # Try different perturbation sets
                # Small coprime integers that create collisions
                perturbations_list = [
                    [1]*20,  # Pure geometric
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,3],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,4,5],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,4,5,6,7,8],
                    [1,1,1,1,1,1,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                ]
                
                for perms in perturbations_list:
                    if time.time() - start_time > timeout:
                        break
                    
                    A = _generate_geometric_perturbed(base, exponents, perms)
                    if len(A) < 20:
                        continue
                    
                    # Ensure distinctness and validity
                    A = sorted(set(x for x in A if 1 <= x <= 10000))
                    if len(A) < 20:
                        continue
                    
                    score = _calc_score(A)
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # 1. Generate Candidate Pools
    pool_2 = _generate_smooth_candidates(10000, [2, 3])
    pool_3 = _generate_smooth_candidates(10000, [2, 3, 5])
    pool_4 = _generate_smooth_candidates(10000, [2, 3, 5, 7])
    
    random.seed(42)
    random_ints = random.sample(range(1, 200), 50)
    
    candidate_pool = list(set(pool_2 + pool_3 + pool_4 + random_ints))
    candidate_pool.sort()
    
    # 2. Initial Candidates
    initial_sets = []
    
    if len(pool_2) >= 20: initial_sets.append(pool_2[:20])
    if len(pool_3) >= 20: initial_sets.append(pool_3[:20])
    if len(pool_4) >= 20: initial_sets.append(pool_4[:20])
    
    initial_sets.append(list(range(1, 21)))
    
    p2 = [2**i for i in range(20)]
    if max(p2) <= 10000: initial_sets.append(p2)
    
    geo = [1]
    for i in range(19):
        geo.append(int(round(geo[-1] * 1.5)))
    geo = sorted(set(x for x in geo if x <= 10000))
    if len(geo) >= 20: initial_sets.append(geo[:20])
    
    if len(candidate_pool) >= 20:
        initial_sets.append(sorted(random.sample(candidate_pool, 20)))
        
    best_A = None
    best_score = float('inf')
    
    for init_set in initial_sets:
        if time.time() > deadline - 25.0:
            break
        init_set = sorted(set(x for x in init_set if 1 <= x <= 10000))
        if len(init_set) < 20:
            continue
        score = _calc_score(init_set)
        if score < best_score:
            best_score = score
            best_A = list(init_set)
            
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = _calc_score(best_A)
        
    # 3. Search geometric structures (new strategy)
    time_budget_geo = min(20.0, deadline - time.time() - 25.0)
    if time_budget_geo > 2.0:
        geo_A, geo_score = _search_geometric_structures(timeout=time_budget_geo)
        if geo_A is not None and geo_score < best_score:
            best_score = geo_score
            best_A = geo_A
    
    # 4. Optimization Phase 1: Greedy Swap Local Search
    time_budget_1 = min(15.0, deadline - time.time() - 15.0)
    if time_budget_1 > 1.0:
        best_A, best_score = _local_search_swap(best_A, candidate_pool, timeout=time_budget_1)
        
    # 5. Optimization Phase 2: Simulated Annealing
    time_budget_2 = min(15.0, deadline - time.time() - 10.0)
    if time_budget_2 > 1.0:
        best_A, best_score = _simulated_annealing(best_A, candidate_pool, timeout=time_budget_2)
        
    # 6. Optimization Phase 3: Fine Polish
    time_budget_3 = deadline - time.time() - 2.0
    if time_budget_3 > 1.0:
        local_pool = set()
        for x in best_A:
            for d in range(-5, 6):
                nx = x + d
                if 1 <= nx <= 10000:
                    local_pool.add(nx)
        local_pool = list(local_pool)
        best_A, best_score = _local_search_swap(best_A, local_pool, timeout=time_budget_3)
        
    return best_A
