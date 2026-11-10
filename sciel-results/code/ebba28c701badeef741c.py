import time
import random
from math import gcd

def calc_score(A):
    """
    Calculate the max of the size of the sumset and product set of A.
    A is a list of distinct positive integers.
    """
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _generate_smooth(primes, max_val=10000):
    """Generate smooth numbers with given prime bases."""
    result = set()
    def rec(idx, current):
        if current > max_val:
            return
        result.add(current)
        if idx < len(primes):
            val = current
            while val * primes[idx] <= max_val:
                val *= primes[idx]
                rec(idx + 1, val)
    rec(0, 1)
    return sorted(result)

def _gen_geo_perturbed(base, exponents, perturbations, max_val=10000):
    """
    Generate a set A = {base^e_i * k_i} where k_i are small coprime perturbations.
    This exploits alignment of additive and multiplicative structures.
    """
    A = set()
    for e, k in zip(exponents, perturbations):
        val = (base ** e) * k
        if 1 <= val <= max_val:
            A.add(val)
    if len(A) >= 20:
        return sorted(A)[:20]
    return None

def local_search(A, time_limit, rng, candidate_pool=None):
    """
    Perform local search to minimize the score.
    Tries to replace elements in A with other integers to improve the score.
    """
    A = list(A)
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    if candidate_pool is None:
        # Build a rich candidate pool
        candidate_pool = set()
        # Small integers
        candidate_pool.update(range(1, 101))
        # 2-smooth
        for i in range(14):
            for j in range(14):
                val = (2 ** i) * (3 ** j)
                if val <= 10000:
                    candidate_pool.add(val)
        # 2,3,5-smooth
        for p in _generate_smooth([2, 3, 5]):
            if p <= 10000:
                candidate_pool.add(p)
        # Geometric progression values with perturbations
        for base in [2, 3, 4]:
            for e in range(12):
                for k in range(1, 8):
                    val = (base ** e) * k
                    if val <= 10000:
                        candidate_pool.add(val)
        # Powers of small integers
        for b in range(2, 10):
            val = b
            while val <= 10000:
                candidate_pool.add(val)
                val *= b
        
        candidate_pool = sorted(candidate_pool)
        rng.shuffle(candidate_pool)
    
    # Ensure A is sorted
    A = sorted(A)
    best_A = sorted(best_A)
    
    # Use a set for O(1) membership testing
    A_set = set(A)
    
    improved = True
    stall_count = 0
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        stall_count += 1
        # After several full passes without improvement, reduce pool or stop
        if stall_count > 5:
            break
            
        for i in range(len(A)):
            if (time.time() - start_time) > time_limit:
                return best_A, best_score
            
            old_val = A[i]
            
            for new_val in candidate_pool:
                if new_val == old_val or new_val in A_set:
                    continue
                
                # Create new set
                new_A = A.copy()
                new_A[i] = new_val
                new_A.sort()
                new_A_set = set(new_A)
                
                # Calculate score
                score = calc_score(new_A)
                
                if score < current_score:
                    current_score = score
                    A = new_A
                    A_set = new_A_set
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    stall_count = 0
                    break
        
    return best_A, best_score

def _polish(A, time_limit, rng):
    """
    Final polish: try small perturbations of each element.
    """
    A = sorted(A)
    current_score = calc_score(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    A_set = set(A)
    
    # Try +/- small deltas for each element
    deltas = list(range(1, 6)) + list(range(-5, 0))
    
    while (time.time() - start_time) < time_limit:
        improved = False
        for i in range(len(A)):
            if (time.time() - start_time) > time_limit:
                break
            old_val = A[i]
            for d in deltas:
                new_val = old_val + d
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in A_set:
                    continue
                
                new_A = A.copy()
                new_A[i] = new_val
                new_A.sort()
                new_A_set = set(new_A)
                
                score = calc_score(new_A)
                if score < current_score:
                    current_score = score
                    A = new_A
                    A_set = new_A_set
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_A = list(A)
                    break
        if not improved:
            break
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin
    
    rng = random.Random(42)
    
    candidates = []
    
    # Candidate 1: Arithmetic Progression 1..20
    candidates.append(list(range(1, 21)))
    
    # Candidate 2: Geometric Progression 1, 2, 4, ...
    gp = [2**i for i in range(14)]
    for x in range(1, 100):
        if x not in gp and len(gp) < 20:
            gp.append(x)
    candidates.append(gp)
    
    # Candidate 3: 2,3-smooth numbers
    smooth = []
    for a in range(14):
        for b in range(14):
            val = (2**a) * (3**b)
            if val <= 10000:
                smooth.append(val)
    smooth = sorted(set(smooth))
    candidates.append(smooth[:20])
    
    # Candidate 4: Random set of small integers
    rand_set = random.sample(range(1, 101), 20)
    candidates.append(rand_set)
    
    # Candidate 5: Mixed structure - low AP + high sparse
    mixed = list(range(1, 11)) + [12, 14, 16, 20, 25, 30, 40, 50, 60, 80]
    candidates.append(mixed)
    
    # Candidate 6: 2,3,5-smooth numbers
    smooth_235 = _generate_smooth([2, 3, 5])
    candidates.append(smooth_235[:20])
    
    # Candidate 7: 2,3,5,7-smooth numbers
    smooth_2357 = _generate_smooth([2, 3, 5, 7])
    candidates.append(smooth_2357[:20])
    
    # Candidate 8: Geometric progression with ratio ~1.3
    geo = []
    val = 1
    for i in range(20):
        val = int(round(val * 1.3))
        if val > 10000:
            break
        if val not in geo:
            geo.append(val)
    if len(geo) >= 20:
        candidates.append(geo[:20])
    
    # Candidate 9: Geometric progression with ratio ~1.5
    geo2 = []
    val = 1
    for i in range(20):
        val = int(round(val * 1.5))
        if val > 10000:
            break
        if val not in geo2:
            geo2.append(val)
    if len(geo2) >= 20:
        candidates.append(geo2[:20])
    
    # Candidate 10: Mix: small numbers + powers of 2
    mix = list(range(1, 11)) + [2**i for i in range(4, 14)]
    mix = sorted(set(x for x in mix if x <= 10000))
    if len(mix) >= 20:
        candidates.append(mix[:20])
    
    # Candidate 11: Random sample from smooth pool
    if len(smooth_235) >= 20:
        candidates.append(sorted(random.sample(smooth_235, 20)))
    
    # NEW: Geometric progression with multiplicative perturbations
    # Base=2, exponents 0..19, perturbations 1..7 (cyclic)
    exponents = list(range(20))
    perturbations = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6]
    gp_pert = _gen_geo_perturbed(2, exponents, perturbations)
    if gp_pert:
        candidates.append(gp_pert)
    
    # Base=3, exponents 0..13 (3^13=1594329 > 10000, so up to 3^8=6561)
    exponents3 = list(range(9))  # 0..8
    perturbations3 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Need 20 elements, so use more exponents with perturbations
    exponents3b = [0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8]
    perturbations3b = [1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 1]
    gp3_pert = _gen_geo_perturbed(3, exponents3b, perturbations3b)
    if gp3_pert:
        candidates.append(gp3_pert)
    
    # Base=4, exponents 0..9 (4^9=262144 > 10000, so up to 4^6=4096)
    exponents4 = [0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6]
    perturbations4 = [1, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5]
    gp4_pert = _gen_geo_perturbed(4, exponents4, perturbations4)
    if gp4_pert:
        candidates.append(gp4_pert)
    
    # Candidate: 2-smooth with varied exponents
    smooth2 = []
    for a in range(14):
        for b in range(14):
            val = (2**a) * (3**b)
            if val <= 10000:
                smooth2.append(val)
    smooth2 = sorted(set(smooth2))
    # Pick a spread subset
    if len(smooth2) >= 20:
        step = max(1, len(smooth2) // 20)
        candidates.append([smooth2[i*step] for i in range(20)])
    
    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for c in candidates:
        if len(set(c)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in c):
            continue
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    best_A = None
    best_score = float('inf')
    
    n_cands = len(unique_candidates)
    # Allocate time: more time for early candidates (they're often better seeds)
    for idx, A in enumerate(unique_candidates):
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 2:
            break
        
        # Allocate proportionally: first few get more time
        remaining_cands = n_cands - idx
        # Give each candidate at least 3s, cap at 15s
        alloc_time = min(15.0, max(3.0, time_remaining / max(1, remaining_cands) * 0.8))
        
        improved_A, score = local_search(A, alloc_time, rng)
        
        if score < best_score:
            best_score = score
            best_A = improved_A
    
    # Final polish on the best result
    if best_A is not None and time.time() < deadline - 3:
        polish_time = min(10.0, deadline - time.time() - 2)
        if polish_time > 1:
            polished_A, polished_score = _polish(best_A, polish_time, rng)
            if polished_score < best_score:
                best_score = polished_score
                best_A = polished_A
    
    if best_A is None:
        best_A = list(range(1, 21))
    
    return best_A
