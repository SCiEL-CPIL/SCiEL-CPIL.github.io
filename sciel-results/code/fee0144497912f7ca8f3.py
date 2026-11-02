"""Combined approach: sparse constructions + prime-based + local search optimization."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set
    sumset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
    
    diffset = set()
    for a in U_set:
        for b in U_set:
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set_fast(U):
    """Faster scoring using numpy for small-to-moderate sets."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    arr = np.array(sorted(U_set), dtype=np.int64)
    
    if n <= 250:
        sums = arr[:, None] + arr[None, :]
        sumset_size = len(np.unique(sums))
        diffs = arr[:, None] - arr[None, :]
        diffset_size = len(np.unique(diffs))
    else:
        sumset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
        sumset_size = len(sumset)
        
        diffset = set()
        for a in U_set:
            for b in U_set:
                diffset.add(a - b)
        diffset_size = len(diffset)
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _is_prime(n):
    """Check if n is prime."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _generate_primes(max_val):
    """Generate all primes up to max_val."""
    if max_val < 2:
        return []
    sieve = [True] * (max_val + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(max_val)) + 1):
        if sieve[i]:
            for j in range(i*i, max_val + 1, i):
                sieve[j] = False
    return [i for i in range(2, max_val + 1) if sieve[i]]

def _construct_golomb(n):
    """Construct a Golomb ruler of order n (all pairwise differences distinct)."""
    if n <= 1:
        return [0]
    
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
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 84],
        15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 84, 96],
        16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 84, 96, 108],
    }
    if n in known:
        return known[n]
    
    # Greedy construction
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

def _construct_sidon(n, max_val=None):
    """Construct a Sidon (B_2) sequence of length n."""
    if max_val is None:
        max_val = n * n * 6
    
    seq = [0]
    used_sums = {0}
    
    for i in range(1, n):
        pos = 1
        while pos <= max_val:
            valid = True
            for a in seq:
                s = a + pos
                if s in used_sums:
                    valid = False
                    break
            if valid:
                all_vals = seq + [pos]
                sums = set()
                for x in range(len(all_vals)):
                    for y in range(x, len(all_vals)):
                        s = all_vals[x] + all_vals[y]
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                if valid:
                    seq.append(pos)
                    break
            pos += 1
        if len(seq) < i + 1:
            break
    
    return seq

def _prime_gap_construction(n, primes=None, max_val=1000):
    """Construct U = {0, p1, p1+p2, p1+p2+p3, ...} using distinct primes."""
    if n <= 1:
        return [0]
    
    if primes is None:
        primes = _generate_primes(max_val)
    
    U = [0]
    current = 0
    for i in range(1, n):
        if i - 1 < len(primes):
            current += primes[i - 1]
            U.append(current)
        else:
            break
    
    return U

def _sparse_arithmetic(n, base, step, deadline=None):
    """Sparse arithmetic progression with prime-like steps."""
    U = [0]
    current = 0
    primes = _generate_primes(step * n * 2)
    
    for i in range(1, n):
        if i - 1 < len(primes):
            current += primes[i - 1] * base
            U.append(current)
        else:
            break
    
    return U

def _greedy_prime_build(n, max_val=500, deadline=None):
    """Greedy selection preferring primes to minimize sumset collisions."""
    U = [0]
    primes = _generate_primes(max_val)
    
    for _ in range(n - 1):
        if deadline and time.time() > deadline:
            break
        
        best_score = _score_set_fast(U)
        best_candidate = None
        best_s = best_score
        
        # Try primes first
        for p in primes:
            if p in U:
                continue
            new_U = sorted(U + [p])
            s = _score_set_fast(new_U)
            if s > best_s + 1e-12:
                best_s = s
                best_candidate = p
        
        if best_candidate is None:
            # Try non-primes
            for v in range(1, max_val + 1):
                if v in U or _is_prime(v):
                    continue
                new_U = sorted(U + [v])
                s = _score_set_fast(new_U)
                if s > best_s + 1e-12:
                    best_s = s
                    best_candidate = v
        
        if best_candidate is not None:
            U.append(best_candidate)
            U = sorted(U)
        else:
            break
    
    return U

def _local_search(U, max_iters=5000, deadline=None, max_size=40):
    """Local search with add/remove/modify operations."""
    U = sorted(set(U))
    best_score = _score_set_fast(U)
    best_U = U[:]
    max_u = max(U) if U else 0
    upper_bound = max_u + 80
    
    scorer = _score_set_fast
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        U_work = U[:]
        random.shuffle(U_work)
        
        # Try modifying one element
        for i in range(len(U_work)):
            old_val = U_work[i]
            candidates = []
            for delta in [-5, -3, -2, -1, 1, 2, 3, 5, 10, 20, 30]:
                nv = old_val + delta
                if 0 <= nv <= upper_bound and nv != old_val:
                    candidates.append(nv)
            for _ in range(3):
                nv = random.randint(0, upper_bound)
                if nv != old_val:
                    candidates.append(nv)
            
            for new_val in candidates:
                new_U = U_work[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                new_U_sorted = sorted(new_U_set)
                if new_U_sorted[0] != 0:
                    continue
                s = scorer(new_U_sorted)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U_sorted
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding an element
            if len(U) < max_size:
                # Try primes first
                primes = _generate_primes(upper_bound)
                for new_val in primes:
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = scorer(new_U)
                    if s > best_score + 1e-10:
                        best_score = s
                        best_U = sorted(new_U)
                        improved = True
                        break
                
                if not improved:
                    for new_val in range(1, upper_bound + 1):
                        if new_val in U:
                            continue
                        new_U = U + [new_val]
                        s = scorer(new_U)
                        if s > best_score + 1e-10:
                            best_score = s
                            best_U = sorted(new_U)
                            improved = True
                            break
        
        if not improved:
            # Try removing an element
            if len(U) > 2:
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    new_U = U[:i] + U[i+1:]
                    s = scorer(new_U)
                    if s > best_score + 1e-10:
                        best_score = s
                        best_U = new_U
                        improved = True
                        break
        
        if not improved:
            # Random mutation
            if random.random() < 0.15:
                mutated = U[:]
                for _ in range(random.randint(1, 2)):
                    nv = random.randint(1, upper_bound)
                    if nv not in mutated:
                        mutated.append(nv)
                removable = [x for x in mutated if x != 0]
                for _ in range(random.randint(0, min(1, len(removable)))):
                    if removable:
                        idx = random.randint(0, len(removable) - 1)
                        val = removable.pop(idx)
                        mutated.remove(val)
                if 0 in mutated:
                    mutated = sorted(set(mutated))
                    s = scorer(mutated)
                    if s > best_score + 1e-10:
                        best_score = s
                        best_U = mutated
                        improved = True
        
        if not improved and it > 50:
            upper_bound += 30
    
    return sorted(set(best_U)), best_score

def _polish(U, deadline, max_iters=2000):
    """Final high-precision polish with tighter tolerance."""
    U = sorted(set(U))
    best_score = _score_set_fast(U)
    best_U = U[:]
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        U_work = U[:]
        random.shuffle(U_work)
        
        for i in range(len(U_work)):
            old_val = U_work[i]
            # Fine-grained search around current value
            for delta in range(-3, 4):
                nv = old_val + delta
                if nv < 0 or nv == old_val:
                    continue
                new_U = U_work[:]
                new_U[i] = nv
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                new_U_sorted = sorted(new_U_set)
                if new_U_sorted[0] != 0:
                    continue
                s = _score_set_fast(new_U_sorted)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U_sorted
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try add/remove
            max_u = max(best_U) if best_U else 0
            if len(best_U) < 35:
                for new_val in range(1, max_u + 20):
                    if new_val in best_U:
                        continue
                    new_U = best_U + [new_val]
                    s = _score_set_fast(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = sorted(new_U)
                        improved = True
                        break
            
            if not improved and len(best_U) > 3:
                for i in range(len(best_U)):
                    if best_U[i] == 0:
                        continue
                    new_U = best_U[:i] + best_U[i+1:]
                    s = _score_set_fast(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U
                        improved = True
                        break
        
        if not improved and it > 20:
            break
    
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)
    
    # Phase 1: Generate diverse candidates from multiple construction strategies
    candidates = []
    
    # Golomb rulers
    for n in range(4, 18):
        if time.time() > deadline - 10:
            break
        gr = _construct_golomb(n)
        candidates.append(gr)
    
    # Sidon sequences
    for n in range(4, 15):
        if time.time() > deadline - 10:
            break
        ss = _construct_sidon(n, max_val=n * n * 8)
        if len(ss) >= 4:
            candidates.append(ss)
    
    # Prime gap constructions
    primes_500 = _generate_primes(500)
    for n in range(4, 15):
        if time.time() > deadline - 10:
            break
        pg = _prime_gap_construction(n, primes=primes_500)
        candidates.append(pg)
    
    # Sparse arithmetic with prime steps
    for base in [1, 2, 3]:
        for n in range(4, 12):
            if time.time() > deadline - 10:
                break
            sa = _sparse_arithmetic(n, base, 1)
            candidates.append(sa)
    
    # Greedy prime builds
    for n in [5, 8, 10, 12, 15]:
        if time.time() > deadline - 10:
            break
        gp = _greedy_prime_build(n, max_val=300, deadline=deadline)
        if len(gp) >= 4:
            candidates.append(gp)
    
    # Random sets
    random.seed(42)
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        n = random.randint(4, 15)
        max_val = random.randint(30, 200)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        candidates.append(sorted(U))
    
    # Score all candidates and keep the best
    for U in candidates:
        if time.time() > deadline - 5:
            break
        s = _score_set_fast(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Phase 2: Local search on the best candidate
    if time.time() < deadline - 10:
        improved_U, improved_score = _local_search(best_U, max_iters=4000, deadline=deadline - 5, max_size=35)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    # Phase 3: Polish
    if time.time() < deadline - 3:
        polished_U, polished_score = _polish(best_U, deadline=deadline - 1, max_iters=1500)
        if polished_score > best_score:
            best_score = polished_score
            best_U = polished_U
    
    # Phase 4: Try local search on a few other good candidates
    if time.time() < deadline - 5:
        # Re-score all candidates and pick top 3
        scored = [(s, U) for U in candidates for s in [_score_set_fast(U)]]
        scored.sort(key=lambda x: -x[0])
        
        for s, U in scored[:3]:
            if time.time() > deadline - 3:
                break
            if s < best_score - 0.02:
                continue
            improved_U, improved_score = _local_search(U, max_iters=2000, deadline=deadline - 2, max_size=35)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
