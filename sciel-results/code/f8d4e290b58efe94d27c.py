import math
import time
import random
import numpy as np

def _score(u):
    """Compute the score for set u."""
    if not u:
        return 0.0
    u = list(u)
    max_u = max(u)
    if max_u == 0:
        return 1.0
    diffs = set()
    sums = set()
    for a in u:
        for b in u:
            diffs.add(a - b)
            sums.add(a + b)
    num_diffs = len(diffs)
    num_sums = len(sums)
    if num_sums == 0:
        return 0.0
    ratio = num_diffs / num_sums
    denom = 2 * max_u + 1
    if denom <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(denom)

def _score_fast(u):
    """Faster score using numpy for moderate-sized sets."""
    u = np.array(u, dtype=np.int64)
    n = len(u)
    if n == 0:
        return 0.0
    max_u = int(u.max())
    if max_u == 0:
        return 1.0
    
    # For small sets, use the exact method
    if n <= 200:
        return _score(u.tolist())
    
    # For larger sets, use FFT-based convolution
    # Sum set: compute convolution of indicator function with itself
    # Difference set: compute convolution of indicator with reversed indicator
    
    M = 2 * max_u + 1
    if M > 2**22:  # Too large for FFT, fall back
        return _score(u.tolist())
    
    indicator = np.zeros(M, dtype=np.float64)
    indicator[u] = 1.0
    
    # Sum set via convolution
    # conv[i] = sum_j indicator[j] * indicator[i-j] for valid j
    # Use FFT
    size = 1
    while size < 2 * M - 1:
        size *= 2
    f = np.fft.fft(indicator, size)
    conv = np.fft.ifft(f * f).real
    # Sum set elements are where conv[i] > 0 for i in [0, 2*max_u]
    # But conv[i] = sum_{j} ind[j]*ind[i-j], and i-j must be in [0, max_u]
    # Actually for sum set, we want a+b where a,b in U
    # conv[i] = sum_j ind[j] * ind[i-j], this counts pairs (j, i-j) both in U
    # So sum set = {i : conv[i] > 0}
    sum_set_size = np.sum(conv[:M] > 0.5)
    
    # Difference set: a-b where a,b in U
    # rev[i] = ind[max_u - i]
    rev = np.zeros(M, dtype=np.float64)
    rev[u] = 1.0
    # diff_conv[i] = sum_j rev[j] * ind[i+j] = sum_j ind[max_u-j] * ind[i+j]
    # Let k = max_u - j, then j = max_u - k
    # diff_conv[i] = sum_k ind[k] * ind[i + max_u - k]
    # This is the cross-correlation
    # Alternatively: diff set = {a-b : a,b in U} = {d : exists a,b in U with a-b=d}
    # = {d : sum_a ind[a]*ind[a-d] > 0} for d in [-max_u, max_u]
    # Let's compute: corr[d] = sum_a ind[a] * ind[a-d]
    # This is convolution of ind with reversed ind
    rev2 = ind[::-1].copy()
    size2 = 1
    while size2 < 2 * M - 1:
        size2 *= 2
    f2 = np.fft.fft(ind, size2)
    f2r = np.fft.fft(rev2, size2)
    diff_conv = np.fft.ifft(f2 * f2r).real
    # diff_conv[i] for i in [0, 2*max_u], but we need to map to [-max_u, max_u]
    # Actually, the convolution ind * rev2 gives us the autocorrelation
    # corr[d] = sum_k ind[k]*ind[k-d]
    # In convolution terms: (ind * rev2)[m] = sum_k ind[k]*rev2[m-k] = sum_k ind[k]*ind[max_u-(m-k)] = sum_k ind[k]*ind[max_u-m+k]
    # Let j = k, then this is sum_j ind[j]*ind[max_u-m+j]
    # We want sum_j ind[j]*ind[j-d], so set max_u-m+j = j-d, i.e., max_u-m = -d, d = m - max_u
    # So corr[d] = conv[m] where m = max_u + d
    # d ranges from -max_u to max_u, so m ranges from 0 to 2*max_u
    # diff_set_size = number of d in [-max_u, max_u] where corr[d] > 0
    diff_set_size = np.sum(diff_conv[:2*max_u+1] > 0.5)
    
    ratio = diff_set_size / sum_set_size
    denom = 2 * max_u + 1
    if denom <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(denom)

def _construct_quadratic_residues(p):
    """
    Construct U as a subset related to quadratic residues mod p.
    For prime p, the set of quadratic residues mod p has size (p-1)/2.
    We map them to integers in [0, p-1].
    """
    if p < 5:
        return [0]
    # Quadratic residues mod p: {x^2 mod p : x = 1, ..., (p-1)/2}
    # Plus 0
    qr = set()
    for x in range(1, (p-1)//2 + 1):
        qr.add((x*x) % p)
    qr.add(0)
    # Scale to get good range
    # The set is in [0, p-1], max = p-1
    # Let's use it directly
    return sorted(qr)

def _construct_golomb_like(n):
    """
    Construct a Golomb-like set: a set where all pairwise differences are distinct.
    This maximizes |U-U| relative to |U|.
    """
    # Start with a known Golomb ruler construction
    # Simple approach: use powers of 2 (exponential growth)
    # But that gives huge max U
    # Better: greedy construction
    u = [0]
    for i in range(1, n):
        # Try to find the smallest value that doesn't create duplicate differences
        diffs = set()
        for a in u:
            for b in u:
                diffs.add(abs(a - b))
        val = 1
        while True:
            new_diffs = set()
            for a in u:
                new_diffs.add(abs(a - val))
            if new_diffs & diffs:
                val += 1
            else:
                break
        u.append(val)
    return u

def _simulated_annealing(initial_u, max_time, rng):
    """
    Simulated annealing to optimize the set.
    """
    current = list(initial_u)
    current_score = _score(current)
    best = list(current)
    best_score = current_score
    
    T = 0.1
    cooling_rate = 0.99999
    min_T = 0.0001
    
    deadline = time.time() + max_time
    max_val = max(current) + 50 if current else 100
    
    steps = 0
    while time.time() < deadline and T > min_T:
        steps += 1
        # Random move
        op = rng.random()
        new_set = list(current)
        
        if op < 0.4:
            # Add element
            candidate = rng.randint(0, max_val)
            if candidate not in new_set:
                new_set.append(candidate)
                new_set.sort()
        elif op < 0.7:
            # Remove element (not 0)
            if len(new_set) > 1:
                idx = rng.randint(1, len(new_set) - 1)
                new_set.pop(idx)
        else:
            # Swap
            if len(new_set) > 1:
                idx = rng.randint(1, len(new_set) - 1)
                old_val = new_set[idx]
                new_val = rng.randint(0, max_val)
                if new_val != old_val and new_val not in new_set:
                    new_set[idx] = new_val
                    new_set.sort()
        
        if len(new_set) == len(current):
            new_score = _score(new_set)
            delta = new_score - current_score
            if delta > 0:
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
                    if max_val < max(current) + 50:
                        max_val = max(current) + 50
                        max_val = min(max_val, 100000)
            else:
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    current = new_set
                    current_score = new_score
        
        T *= cooling_rate
        # Periodically check time
        if steps % 10000 == 0 and time.time() >= deadline:
            break
    
    return best

def _local_search_refine(u, max_time, rng):
    """
    Local search with structured moves: try to improve by
    1. Removing elements that reduce the ratio
    2. Adding elements that increase the ratio
    3. Trying arithmetic progressions
    """
    current = list(u)
    current_score = _score(current)
    best = list(current)
    best_score = current_score
    
    deadline = time.time() + max_time
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        max_val = max(current) + 20 if current else 100
        max_val = min(max_val, 50000)
        
        # Try removing each non-zero element
        for i in range(len(current) - 1, 0, -1):
            if time.time() >= deadline:
                break
            new_set = current[:i] + current[i+1:]
            new_score = _score(new_set)
            if new_score > current_score + 1e-9:
                current = new_set
                current_score = new_score
                improved = True
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
                break  # Restart scan after improvement
        
        if improved:
            continue
        
        no_improve_count += 1
        
        # Try adding elements
        tried = 0
        while tried < 50 and time.time() < deadline:
            tried += 1
            candidate = rng.randint(0, max_val)
            if candidate in current:
                continue
            new_set = current + [candidate]
            new_set.sort()
            new_score = _score(new_set)
            if new_score > current_score + 1e-9:
                current = new_set
                current_score = new_score
                improved = True
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
                break
        
        if improved:
            continue
        
        # Try swapping
        if len(current) > 1:
            idx = rng.randint(1, len(current) - 1)
            old_val = current[idx]
            new_val = rng.randint(0, max_val)
            if new_val != old_val and new_val not in current:
                new_set = list(current)
                new_set[idx] = new_val
                new_set.sort()
                new_score = _score(new_set)
                if new_score > current_score + 1e-9:
                    current = new_set
                    current_score = new_score
                    improved = True
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best = list(current)
    
    return best

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the sumset/difference-set exponent lower bound.
    
    Strategy: Use multiple construction methods and local search to find
    a set that maximizes the score.
    """
    start_time = time.time()
    total_time = 145  # Leave 5s margin from 150s limit
    rng = random.Random(12345)
    
    # Try multiple initial constructions
    candidates = []
    
    # 1. Small seed sets
    seeds = [
        [0, 1, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 4],
        [0, 1, 2, 3, 5],
        [0, 1, 3, 4, 6],
        [0, 1, 2, 4, 5],
        [0, 1, 4, 5],
        [0, 1, 2, 5, 6],
        [0, 1, 3, 5, 7],
        [0, 2, 5, 7, 12],
    ]
    for s in seeds:
        candidates.append(s)
    
    # 2. Golomb-like sets
    try:
        g = _construct_golomb_like(8)
        candidates.append(g)
        g2 = _construct_golomb_like(12)
        candidates.append(g2)
    except Exception:
        pass
    
    # 3. Quadratic residues for small primes
    for p in [7, 13, 31, 61]:
        try:
            qr = _construct_quadratic_residues(p)
            candidates.append(qr)
        except Exception:
            pass
    
    # 4. Exponential sets
    exp_set = [0]
    val = 1
    while val < 5000:
        exp_set.append(val)
        val = val * 2 + 1
    candidates.append(exp_set[:15])
    
    # 5. Random sets
    for size in [10, 20, 30]:
        rset = [0]
        for _ in range(size - 1):
            rset.append(rng.randint(1, 5000))
        rset = sorted(set(rset))
        candidates.append(rset)
    
    # Evaluate all candidates
    best_set = None
    best_score = -1
    for c in candidates:
        try:
            s = _score(c)
            if s > best_score:
                best_score = s
                best_set = list(c)
        except Exception:
            continue
    
    if best_set is None:
        best_set = [0, 1, 3]
        best_score = _score(best_set)
    
    # Phase 1: Local search refinement (40% of time)
    time_left = total_time - (time.time() - start_time)
    phase1_time = time_left * 0.4
    if phase1_time > 5:
        best_set = _local_search_refine(best_set, phase1_time, rng)
        best_score = _score(best_set)
    
    # Phase 2: Simulated annealing (30% of time)
    time_left = total_time - (time.time() - start_time)
    phase2_time = time_left * 0.4
    if phase2_time > 10:
        result = _simulated_annealing(best_set, phase2_time, rng)
        result_score = _score(result)
        if result_score > best_score:
            best_score = result_score
            best_set = result
    
    # Phase 3: Final local search polish (remaining time)
    time_left = total_time - (time.time() - start_time)
    if time_left > 5:
        best_set = _local_search_refine(best_set, time_left * 0.9, rng)
    
    # Ensure 0 is in the set
    if 0 not in best_set:
        best_set = sorted([0] + best_set)
    
    return best_set

if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Result size: {len(result)}")
    print(f"Max value: {max(result) if result else 0}")
    print(f"Score: {_score(result)}")
    print(f"Time: {t1 - t0:.2f}s")
