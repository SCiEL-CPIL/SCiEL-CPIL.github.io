import math
import time
import random

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n == 0:
        return -1
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    if len(sumset) == 0:
        return -1
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return -1
    max_u = max(U)
    if max_u == 0:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1 + math.log(ratio) / denom

def _score_fast(U):
    """Fast scoring using sorted list approach."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n == 0:
        return -1
    
    # Compute sumset and diffset
    sumset = set()
    diffset = set()
    for i in range(n):
        for j in range(n):
            sumset.add(U[i] + U[j])
            diffset.add(U[i] - U[j])
    
    if len(sumset) == 0:
        return -1
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return -1
    max_u = max(U)
    if max_u == 0:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1 + math.log(ratio) / denom

def _gap_multiplication_construct(primes, max_elements=20):
    """
    Construct set using gap-multiplication: cumulative sum of products of distinct primes.
    The idea: if gaps are products of distinct primes, differences have unique factorization,
    maximizing |U-U| while keeping max(U) moderate.
    """
    # Start with 0
    U = [0]
    current = 0
    used_primes = set()
    
    # Greedy: pick next prime product that maximizes instantaneous ratio gain
    # For each candidate gap, check if it creates new differences vs sums
    for step in range(max_elements):
        best_gap = None
        best_gain = -1
        best_new_U = None
        
        # Try single primes not yet used
        for p in primes:
            if p in used_primes:
                continue
            gap = p
            new_elem = current + gap
            new_U = U + [new_elem]
            if len(set(new_U)) != len(new_U):
                continue
            s = _score_fast(new_U)
            if s > best_gain:
                best_gain = s
                best_gap = gap
                best_new_U = new_U
        
        # Try products of two unused primes
        for i in range(len(primes)):
            for j in range(i + 1, len(primes)):
                p1, p2 = primes[i], primes[j]
                if p1 in used_primes or p2 in used_primes:
                    continue
                gap = p1 * p2
                new_elem = current + gap
                new_U = U + [new_elem]
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_fast(new_U)
                if s > best_gain:
                    best_gain = s
                    best_gap = gap
                    best_new_U = new_U
        
        if best_gap is None or best_new_U is None:
            break
        
        U = best_new_U
        current = U[-1]
        # Mark primes as used (we don't track exactly which, but this limits reuse)
        # Actually let's not restrict too much - allow reuse for more options
    
    return sorted(U)

def _local_search(U, deadline, max_iters=5000):
    """Improve set U via local search with swap, add, and remove moves."""
    U = sorted(set(U))
    best_score = _score_fast(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 80
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        
        # Try swapping one element
        for i in range(len(best_U)):
            old_val = best_U[i]
            # Try a range of new values
            for new_val in range(0, upper_bound + 1):
                if new_val == old_val:
                    continue
                new_U = best_U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                if 0 not in new_U:
                    continue
                s = _score_fast(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding an element
            if len(best_U) < 25:
                for new_val in range(0, upper_bound + 1):
                    if new_val in best_U:
                        continue
                    new_U = best_U + [new_val]
                    s = _score_fast(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        improved = True
                        break
        
        if not improved:
            # Try removing an element (if size > 3)
            if len(best_U) > 3:
                for i in range(len(best_U)):
                    if best_U[i] == 0:
                        continue
                    new_U = best_U[:i] + best_U[i+1:]
                    s = _score_fast(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        improved = True
                        break
        
        if not improved:
            break
    
    return sorted(set(best_U)), best_score

def _generate_golomb_like(n):
    """Generate Golomb ruler-like sets for various n."""
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 76],
        15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 76, 84],
    }
    if n in known:
        return known[n]
    # Greedy Golomb ruler
    ruler = [0]
    for i in range(1, n):
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    candidates = []
    
    # 1. Golomb rulers
    for n in range(3, 16):
        if time.time() > deadline:
            break
        gr = _generate_golomb_like(n)
        candidates.append(gr)
    
    # 2. Gap-multiplication constructions
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for max_elems in [5, 7, 9, 11, 13]:
        if time.time() > deadline:
            break
        gm = _gap_multiplication_construct(small_primes, max_elements=max_elems)
        if len(gm) >= 2:
            candidates.append(gm)
    
    # 3. Quadratic residues mod p
    for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]:
        if time.time() > deadline:
            break
        qr = set()
        for x in range(p):
            qr.add((x * x) % p)
        qr = sorted(qr)
        if 0 in qr:
            candidates.append(qr)
        else:
            candidates.append([0] + qr)
    
    # 4. Random sets
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        n = random.randint(3, 18)
        max_val = random.randint(10, 200)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # 5. Structured sets: geometric progressions with perturbations
    for base in [2, 3]:
        for pert in range(0, 5):
            if time.time() > deadline:
                break
            U = [0]
            val = 1
            for i in range(10):
                U.append(val)
                val = val * base + pert
            candidates.append(U[:])
    
    # Evaluate all candidates
    for U in candidates:
        if time.time() > deadline:
            break
        s = _score_fast(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Local search on the best candidate
    if time.time() < deadline:
        best_set, best_score = _local_search(best_set, deadline, max_iters=3000)
    
    # Try local search on other promising candidates
    for U in candidates:
        if time.time() > deadline:
            break
        if U == best_set:
            continue
        s = _score_fast(U)
        if s > best_score - 0.03:
            improved_U, improved_score = _local_search(U, deadline, max_iters=1500)
            if improved_score > best_score:
                best_score = improved_score
                best_set = improved_U
    
    # Final polish
    if time.time() < deadline:
        best_set, best_score = _local_search(best_set, deadline, max_iters=2000)
    
    return sorted(set(best_set))
