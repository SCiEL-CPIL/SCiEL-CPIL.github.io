"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from numpy.fft import fft, ifft

def _next_pow2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def _score_set_fft(U):
    """Compute the score for a finite set U using FFT for speed."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute sumset using FFT
    sum_len = 2 * max_u + 1
    fft_len = _next_pow2(2 * sum_len)
    
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    freq = fft(indicator)
    conv = ifft(freq * freq).real
    
    sumset_size = int(np.sum(conv[:sum_len] > 1e-9))
    
    # Compute difference set using FFT
    diff_len = 2 * max_u + 1
    fft_len2 = _next_pow2(2 * diff_len)
    
    diff_vec = np.zeros(fft_len2, dtype=np.float64)
    for u in U:
        diff_vec[u + max_u] = 1.0
    
    freq_d = fft(diff_vec)
    diff_result = ifft(freq_d * np.conj(freq_d)).real
    
    diffset_size = int(np.sum(np.abs(diff_result[:diff_len]) > 1e-9))
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set_exact(U):
    """Exact O(n^2) score computation for small sets (high precision)."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    for i in range(n):
        for j in range(n):
            sumset.add(U[i] + U[j])
            diffset.add(U[i] - U[j])
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    if n <= 25:
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * max_u + 1)
        if log_denom == 0:
            return 1.0
        return 1.0 + log_ratio / log_denom
    else:
        return _score_set_fft(U)

def _golomb_ruler(n):
    """Generate a Golomb ruler of order n using greedy construction."""
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
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77],
        15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 68, 77, 86],
    }
    if n in known:
        return known[n]
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

def _exhaustive_small_set_search(max_n=8, max_val=30, deadline=None):
    """Exhaustively search for optimal small sets containing 0."""
    from itertools import combinations
    best_U = [0]
    best_score = _score_set(best_U)
    
    for n in range(2, max_n + 1):
        if deadline and time.time() > deadline:
            break
        # Search over combinations of n-1 elements from 1 to max_val
        for combo in combinations(range(1, max_val + 1), n - 1):
            if deadline and time.time() > deadline:
                break
            U = [0] + list(combo)
            s = _score_set(U)
            if s > best_score + 1e-12:
                best_score = s
                best_U = U[:]
    
    return sorted(best_U), best_score

def _prime_step_construction(num_steps, max_prime=None):
    """Construct U using cumulative sums of distinct primes for additive independence."""
    if max_prime is None:
        max_prime = 500
    
    # Generate primes up to max_prime
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    primes = [p for p in range(2, max_prime + 1) if is_prime(p)]
    
    # Greedy selection: at each step, pick the prime that maximizes score
    U = [0]
    current = 0
    best_score = _score_set(U)
    remaining_primes = set(primes)
    
    for step in range(num_steps):
        best_p = None
        best_s = best_score
        # Evaluate all remaining primes if few, otherwise a larger random subset
        candidates = list(remaining_primes)
        if len(candidates) <= 80:
            # Try all
            pass
        else:
            random.shuffle(candidates)
            candidates = candidates[:min(80, len(candidates))]
        
        # Sort by heuristic: prefer primes that are coprime to current differences
        # and spread out from existing elements
        def prime_heuristic(p):
            new_current = current + p
            # Score: prefer larger primes (more spread) but not too large
            spread_score = math.log(p + 1)
            # Penalize if new_current is close to existing elements
            min_dist = min(abs(new_current - u) for u in U) if U else p
            dist_score = math.log(min_dist + 1)
            return spread_score + dist_score
        
        candidates.sort(key=prime_heuristic, reverse=True)
        # Mix: take top by heuristic + random sample
        top_candidates = candidates[:min(30, len(candidates))]
        if len(candidates) > 30:
            random.shuffle(candidates[30:])
            top_candidates += candidates[30:50]
        
        for p in top_candidates:
            new_current = current + p
            new_U = U + [new_current]
            s = _score_set(new_U)
            if s > best_s + 1e-12:
                best_s = s
                best_p = p
        
        if best_p is not None:
            current += best_p
            U.append(current)
            best_score = best_s
            remaining_primes.discard(best_p)
        else:
            break
    
    return sorted(U)

def _hierarchical_staircase(num_blocks, block_sizes=None, deadline=None):
    """
    Construct U using superincreasing block gaps.
    Block i is placed at offset O_i where O_{i+1} - O_i > 2*max(block_i).
    This ensures sums from different blocks never collide, decoupling the problem.
    Each block is a small Golomb-ruler-like structure optimized independently.
    """
    if block_sizes is None:
        # Start with small blocks, grow them
        block_sizes = [3, 4, 5, 6, 7, 8, 9, 10]
        block_sizes = block_sizes[:num_blocks]
    
    U = [0]
    current_offset = 0
    
    for bi, bsize in enumerate(block_sizes):
        if deadline and time.time() > deadline:
            break
        
        # Build a small Golomb ruler for this block
        block = _golomb_ruler(bsize)
        if not block:
            continue
        
        # Shift block so its minimum is at current_offset
        block_min = min(block)
        shifted_block = [current_offset + (x - block_min) for x in block]
        
        # Check: does adding this block improve the score?
        new_U = U + [x for x in shifted_block if x > 0]
        new_U = sorted(set(new_U))
        
        if new_U == U:
            # No new elements, just update offset
            current_offset = max(shifted_block)
            continue
        
        s_new = _score_set(new_U)
        s_old = _score_set(U)
        
        if s_new >= s_old - 1e-12:
            U = new_U
            current_offset = max(shifted_block)
        else:
            # Try a smaller block
            for try_size in range(bsize - 1, 2, -1):
                if deadline and time.time() > deadline:
                    break
                block2 = _golomb_ruler(try_size)
                if not block2:
                    continue
                block2_min = min(block2)
                shifted2 = [current_offset + (x - block2_min) for x in block2]
                new_U2 = U + [x for x in shifted2 if x > 0]
                new_U2 = sorted(set(new_U2))
                s_new2 = _score_set(new_U2)
                if s_new2 >= s_old - 1e-12:
                    U = new_U2
                    current_offset = max(shifted2)
                    break
    
    return sorted(set(U))

def _hierarchical_with_primes(num_blocks, base_block_size, max_prime=200, deadline=None):
    """
    Hierarchical construction where each block is built via prime steps
    and blocks are separated by superincreasing gaps.
    """
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    primes = [p for p in range(2, max_prime + 1) if is_prime(p)]
    
    U = [0]
    current_offset = 0
    used_primes = set()
    
    for bi in range(num_blocks):
        if deadline and time.time() > deadline:
            break
        
        # Build a small prime-step sequence for this block
        block = [0]
        current = 0
        available = [p for p in primes if p not in used_primes]
        
        for step in range(base_block_size - 1):
            if deadline and time.time() > deadline:
                break
            best_p = None
            best_s = _score_set([current_offset + x for x in block])
            
            for p in available[:20]:
                new_current = current + p
                new_block = block + [new_current]
                new_U = [current_offset + x for x in new_block]
                s = _score_set(new_U)
                if s > best_s + 1e-12:
                    best_s = s
                    best_p = p
            
            if best_p is not None:
                current += best_p
                block.append(current)
                used_primes.add(best_p)
        
        # Shift block to current_offset
        shifted_block = [current_offset + x for x in block]
        new_U = U + [x for x in shifted_block if x > 0]
        new_U = sorted(set(new_U))
        
        s_new = _score_set(new_U)
        s_old = _score_set(U)
        
        if s_new >= s_old - 1e-12:
            U = new_U
            current_offset = max(shifted_block)
    
    return sorted(set(U))

def _greedy_prime_selection(U_base, max_prime=300, max_steps=50):
    """Greedy selection of prime steps to maximize score incrementally."""
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    primes = [p for p in range(2, max_prime + 1) if is_prime(p)]
    
    U = [0]
    current = 0
    best_score = _score_set(U)
    
    used_primes = set()
    remaining = list(primes)
    
    for step in range(max_steps):
        best_p = None
        best_s = best_score
        # Try a larger subset of remaining primes with heuristic ordering
        candidates = [p for p in remaining if p not in used_primes]
        
        def prime_heuristic(p):
            new_current = current + p
            spread_score = math.log(p + 1)
            min_dist = min(abs(new_current - u) for u in U) if U else p
            dist_score = math.log(min_dist + 1)
            return spread_score + dist_score
        
        candidates.sort(key=prime_heuristic, reverse=True)
        top_candidates = candidates[:min(60, len(candidates))]
        if len(candidates) > 60:
            random.shuffle(candidates[60:])
            top_candidates += candidates[60:90]
        
        for p in top_candidates:
            new_current = current + p
            new_U = U + [new_current]
            s = _score_set(new_U)
            if s > best_s + 1e-12:
                best_s = s
                best_p = p
        
        if best_p is not None:
            current += best_p
            U.append(current)
            best_score = best_s
            used_primes.add(best_p)
        else:
            break
    
    return sorted(U)

def _local_search(U, max_iters=5000, deadline=None):
    """Improve set U via local search (swap elements, two-swap moves)."""
    U = sorted(set(U))
    n = len(U)
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 800
    
    no_improve_count = 0
    max_no_improve = 120
    
    # For larger sets, use a smaller neighborhood to stay within time
    if n > 50:
        local_range = 40
        rand_count = 20
        two_swap_trials = 15
    elif n > 30:
        local_range = 60
        rand_count = 30
        two_swap_trials = 20
    else:
        local_range = 80
        rand_count = 40
        two_swap_trials = 25
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if no_improve_count >= max_no_improve:
            break
        
        improved = False
        indices_to_try = list(range(len(U)))
        random.shuffle(indices_to_try)
        
        for i in indices_to_try:
            if U[i] == 0:
                continue
            old_val = U[i]
            candidates = set()
            # Local neighborhood
            for offset in range(-local_range, local_range + 1):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            # Some random candidates
            for _ in range(rand_count):
                candidates.add(random.randint(1, upper_bound))
            # Candidates near other elements (to create structured patterns)
            for j in range(len(U)):
                if i != j:
                    candidates.add(max(1, U[j] + random.randint(-20, 20)))
                    candidates.add(max(1, 2 * U[j] - old_val + random.randint(-12, 12)))
                    # Try values that maintain good spacing
                    candidates.add(max(1, U[j] + old_val // 2 + random.randint(-10, 10)))
            
            for new_val in sorted(candidates, key=lambda x: abs(x - old_val)):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            
            # Try two-swap move: change two elements simultaneously
            if len(U) >= 3 and no_improve_count < max_no_improve // 2:
                i1, i2 = random.sample(range(1, len(U)), 2)
                for _ in range(two_swap_trials):
                    nv1 = max(1, U[i1] + random.randint(-45, 45))
                    nv2 = max(1, U[i2] + random.randint(-45, 45))
                    if nv1 == U[i1] and nv2 == U[i2]:
                        continue
                    new_U = U[:]
                    new_U[i1] = nv1
                    new_U[i2] = nv2
                    new_U_set = set(new_U)
                    if len(new_U_set) != len(new_U):
                        continue
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve_count = 0
                        break
            
            if not improved and len(U) < 300:
                add_trials = 40 if n > 50 else 60
                for _ in range(add_trials):
                    new_val = random.randint(1, upper_bound)
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve_count = 0
                        break
            
            if not improved:
                if len(U) > 2:
                    for i in range(len(U)):
                        if U[i] == 0:
                            continue
                        new_U = U[:i] + U[i+1:]
                        s = _score_set(new_U)
                        if s > best_score + 1e-9:
                            best_score = s
                            best_U = new_U[:]
                            U = new_U[:]
                            improved = True
                            no_improve_count = 0
                            break
        
        if not improved:
            if random.random() < 0.4:
                new_U = U[:]
                num_changes = random.randint(2, min(12, len(new_U) - 1))
                for _ in range(num_changes):
                    idx = random.randint(1, len(new_U) - 1)
                    new_U[idx] = random.randint(1, upper_bound)
                new_U_set = set(new_U)
                if len(new_U_set) == len(new_U):
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve_count = 0
            
            if not improved and no_improve_count >= max_no_improve:
                break
    
    return sorted(set(best_U)), best_score

def _modular_residue_construction(max_prime=200, max_target_size=60, deadline=None):
    """
    Construct U by selecting elements from a residue class modulo a prime M.
    U = {x in [0, M-1] : x mod k == r} for some prime M, small k, and residue r.
    This creates structured sum collisions (mod k) while differences remain diverse.
    """
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    best_U = [0]
    best_score = _score_set(best_U)
    
    primes = [p for p in range(2, max_prime + 1) if is_prime(p)]
    
    for M in primes:
        if deadline and time.time() > deadline:
            break
        # Try different moduli k and residues r
        for k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            if deadline and time.time() > deadline:
                break
            for r in range(k):
                # Build candidate set: all x in [0, M) with x % k == r
                candidate = [x for x in range(M) if x % k == r]
                if 0 not in candidate:
                    continue
                if len(candidate) > max_target_size:
                    # Subsample: take every other element or first N
                    if len(candidate) > max_target_size:
                        # Keep elements spread out
                        step = len(candidate) // max_target_size
                        candidate = candidate[::step][:max_target_size]
                
                if len(candidate) < 2:
                    continue
                
                # Ensure 0 is in the set
                if 0 not in candidate:
                    candidate.append(0)
                    candidate = sorted(set(candidate))
                
                s = _score_set(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = candidate[:]
    
    return sorted(set(best_U)), best_score

def _polish(U, deadline=None):
    """High-precision polish pass using exact scoring for small sets."""
    U = sorted(set(U))
    n = len(U)
    if n > 60:
        return _local_search(U, max_iters=3000, deadline=deadline)
    
    best_score = _score_set_exact(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 500
    
    no_improve = 0
    max_no_improve = 80  # Slightly more iterations
    
    # Tighter neighborhood for polish
    polish_range = 50
    
    for it in range(20000):
        if deadline and time.time() > deadline:
            break
        if no_improve >= max_no_improve:
            break
        
        improved = False
        indices_to_try = list(range(1, len(U)))
        random.shuffle(indices_to_try)
        
        for i in indices_to_try:
            old_val = U[i]
            for offset in range(-polish_range, polish_range + 1):
                nv = old_val + offset
                if nv <= 0:
                    continue
                new_U = U[:]
                new_U[i] = nv
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set_exact(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
            if len(U) < 80:
                for _ in range(50):
                    nv = random.randint(1, upper_bound)
                    if nv in U:
                        continue
                    new_U = U + [nv]
                    s = _score_set_exact(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
            if not improved and len(U) > 3:
                for i in range(1, len(U)):
                    new_U = U[:i] + U[i+1:]
                    s = _score_set_exact(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        no_improve = 0
                        break
    
    # Final verification: try a few more targeted moves with wider range
    if deadline and time.time() < deadline - 2:
        for i in range(1, len(best_U)):
            if deadline and time.time() > deadline - 1:
                break
            old_val = best_U[i]
            # Try values that are sums/differences of existing elements
            for j in range(len(best_U)):
                if i == j:
                    continue
                for candidate_val in [best_U[j] + old_val, abs(best_U[j] - old_val), best_U[j] * 2 - old_val]:
                    if candidate_val <= 0 or candidate_val == old_val:
                        continue
                    if candidate_val in best_U:
                        continue
                    new_U = best_U[:]
                    new_U[i] = candidate_val
                    if len(set(new_U)) != len(new_U):
                        continue
                    s = _score_set_exact(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U[:]
    
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    random.seed(42)
    
    candidates = []
    
    # Exhaustive search for small sets (n=4 to 8)
    if time.time() < deadline - 60:
        ex_U, ex_score = _exhaustive_small_set_search(max_n=8, max_val=25, deadline=deadline - 60)
        if ex_score > best_score:
            best_score = ex_score
            best_U = ex_U
        candidates.append(ex_U)
    
    # Golomb rulers of various sizes
    for n in range(3, 35):
        if time.time() > deadline - 55:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)
    
    # Prime-step constructions with greedy selection (more varied parameters)
    for num_steps in [6, 10, 14, 18, 22, 26, 30, 34, 38]:
        if time.time() > deadline - 55:
            break
        for max_p in [150, 300, 500, 800]:
            if time.time() > deadline - 55:
                break
            ps = _prime_step_construction(num_steps, max_prime=max_p)
            candidates.append(ps)
    
    # Greedy prime selection (improved, more trials)
    for _ in range(12):
        if time.time() > deadline - 55:
            break
        for mp in [250, 450, 700]:
            if time.time() > deadline - 55:
                break
            gp = _greedy_prime_selection([0], max_prime=mp, max_steps=40)
            candidates.append(gp)
    
    # Hierarchical staircase constructions
    for nb in range(3, 9):
        if time.time() > deadline - 55:
            break
        hs = _hierarchical_staircase(nb, deadline=deadline - 55)
        candidates.append(hs)
    
    # Hierarchical with prime blocks
    for nb in range(3, 7):
        if time.time() > deadline - 55:
            break
        for bsz in [3, 4, 5]:
            if time.time() > deadline - 55:
                break
            hp = _hierarchical_with_primes(nb, bsz, max_prime=250, deadline=deadline - 55)
            candidates.append(hp)
    
    # Random sets with various sizes
    for trial in range(35):
        if time.time() > deadline - 55:
            break
        n = random.randint(4, 50)
        max_val = random.randint(40, 2000)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Structured sets: arithmetic progressions with gaps
    for base in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
        if time.time() > deadline - 55:
            break
        U = [0]
        for i in range(1, 25):
            U.append(i * base)
        candidates.append(U[:min(len(U), 25)])
    
    # Powers-based sets
    for p in [2, 3, 4, 5]:
        if time.time() > deadline - 55:
            break
        U = [0]
        for i in range(1, 22):
            U.append(p ** i)
            if U[-1] > 40000:
                break
        candidates.append(U)
    
    # Sparse random sets
    for trial in range(20):
        if time.time() > deadline - 55:
            break
        n = random.randint(6, 45)
        max_val = random.randint(150, 4000)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Hybrid: start with Golomb ruler prefix, extend with primes
    for n_prefix in [5, 7, 9, 11]:
        if time.time() > deadline - 55:
            break
        gr = _golomb_ruler(n_prefix)
        current = gr[-1]
        U = gr[:]
        def is_prime(x):
            if x < 2:
                return False
            if x == 2:
                return True
            if x % 2 == 0:
                return False
            for i in range(3, int(math.sqrt(x)) + 1, 2):
                if x % i == 0:
                    return False
            return True
        primes = [p for p in range(2, 500) if is_prime(p)]
        for p in primes[:25]:
            if time.time() > deadline - 55:
                break
            new_current = current + p
            new_U = U + [new_current]
            s = _score_set(new_U)
            if s > _score_set(U):
                U = new_U
                current = new_current
        candidates.append(U)
    
    # Modular residue constructions
    for max_p in [100, 200, 350, 500]:
        if time.time() > deadline - 50:
            break
        mr_U, mr_score = _modular_residue_construction(max_prime=max_p, max_target_size=60, deadline=deadline - 50)
        if len(mr_U) >= 2:
            candidates.append(mr_U)
    
    # Evaluate candidates and keep top ones
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline - 45:
            break
        s = _score_set(U)
        scored_candidates.append((s, U))
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Local search on best candidate
    if time.time() < deadline - 30:
        best_U, best_score = _local_search(best_U, max_iters=12000, deadline=deadline - 30)
    
    # Try local search on top few other candidates
    for s, U in scored_candidates[:12]:
        if time.time() > deadline - 30:
            break
        if U == best_U:
            continue
        if s > best_score - 0.04:
            improved_U, improved_score = _local_search(U, max_iters=5000, deadline=deadline - 30)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    # Final high-precision polish pass
    if time.time() < deadline - 8:
        best_U, best_score = _polish(best_U, deadline=deadline - 3)
    
    # Verify 0 is in the set and all elements are nonnegative integers
    best_U = [int(x) for x in best_U if int(x) >= 0]
    if 0 not in best_U:
        best_U.append(0)
    best_U = sorted(set(best_U))
    
    return best_U
# EVOLVE-BLOCK-END
