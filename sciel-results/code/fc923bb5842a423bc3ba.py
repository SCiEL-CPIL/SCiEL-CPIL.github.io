"""Rewritten program using scipy.sparse for fast sumset/difference set computation."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
import scipy.sparse as sp
from itertools import combinations

def _score_set_fast(U):
    """
    Compute the score for a finite set U containing 0 using sparse matrix multiplication.
    Score = 1 + log(|U-U|/|U+U|) / log(2 * max(U) + 1)
    """
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 1.0
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    
    # Sumset U+U: 
    # Let A be a binary matrix of size (2*max_u + 1) x n, where A[i, j] = 1 if U[j] == i.
    # Then (A @ A.T)[i, j] is the number of pairs (u, v) in U such that u = i, v = j.
    # This is not quite right for sumset.
    # Better: Sumset S = {a+b | a,b in U}.
    # We can represent the set U as a sparse vector v of length 2*max_u + 1.
    # The sumset indicator is the support of the convolution v * v.
    # However, scipy.sparse doesn't have direct convolution.
    # Alternative: Use matrix multiplication.
    # Let M be a (2*max_u + 1) x n matrix where M[i, j] = 1 if U[j] == i.
    # Sumset: For each sum s, count if there exists a,b in U with a+b=s.
    # This is equivalent to checking if the (s, 0) entry of (M @ M.T) is non-zero? No.
    # (M @ M.T)[i, j] = sum_k M[i, k] M[j, k]. This counts common elements, not sums.
    
    # Correct approach for Sumset using MM:
    # Let U = {u_1, ..., u_n}.
    # Create a sparse matrix A of size (2*max_u + 1) x n, A[i, j] = 1 if u_j == i.
    # This is just the identity-like matrix for the set.
    # Sumset size is the number of distinct values in {u_i + u_j}.
    # We can construct a sparse matrix B of size (2*max_u + 1) x (2*max_u + 1) where B[i, j] = 1 if i+j is in the sumset? No.
    
    # Efficient method:
    # 1. Create sparse vector v of length 2*max_u + 1 with 1s at positions U.
    # 2. Sumset is support of v * v (convolution).
    #    scipy.signal.convolve is O(N log N) but creates dense arrays.
    #    For max_u ~ 10000, N=20000, this is fast.
    # 3. Diffset is support of v * v_flipped.
    
    # Let's try numpy FFT based convolution for speed, as it's O(N log N).
    # N = 2 * max_u + 1.
    # If max_u is small (< 10000), numpy FFT is extremely fast.
    
    N = 2 * max_u + 1
    v = np.zeros(N, dtype=np.float64)
    v[U] = 1.0
    
    # Sumset: convolution of v with itself
    # We only need the support (non-zero entries)
    # np.convolve uses FFT internally if method='auto' or 'fft'
    try:
        sum_conv = np.convolve(v, v, mode='full')
        sumset_size = np.count_nonzero(sum_conv > 1e-9)
        
        # Diffset: convolution of v with reversed v
        # v_rev[i] = v[N-1-i]
        v_rev = v[::-1]
        diff_conv = np.convolve(v, v_rev, mode='full')
        # The difference d = a - b corresponds to index in diff_conv.
        # Specifically, if a-b = k, it appears at index (N-1) + k in the full convolution?
        # Let's verify:
        # (v * v_rev)[m] = sum_i v[i] v_rev[m-i] = sum_i v[i] v[N-1-(m-i)] = sum_i v[i] v[N-1-m+i]
        # Let j = N-1-m+i => i = m - (N-1) + j.
        # We want a-b = k.
        # Standard result: support of v * v_rev contains differences.
        # The range of differences is [-max_u, max_u].
        # The convolution result has length 2*N - 1.
        # The zero difference is at index N-1.
        # So diffset_size is count_nonzero(diff_conv[N-1-max_u : N-1+max_u+1])?
        # Actually, just count non-zeros in the central part.
        # The full convolution length is 2*N-1.
        # Indices 0 to 2*N-2.
        # Center is N-1.
        # Differences range from -max_u to max_u.
        # So we look at diff_conv[N-1-max_u : N-1+max_u+1].
        
        # Note: np.convolve(v, v_rev) might have floating point errors.
        # Thresholding is safe.
        diff_conv_centered = diff_conv[N-1-max_u : N-1+max_u+1]
        diffset_size = np.count_nonzero(diff_conv_centered > 1e-9)
        
    except Exception:
        # Fallback for very large max_u or memory issues
        # Use set-based for very small n
        if n < 100:
            sumset = set()
            diffset = set()
            for a in U:
                for b in U:
                    sumset.add(a + b)
                    diffset.add(a - b)
            sumset_size = len(sumset)
            diffset_size = len(diffset)
        else:
            # If FFT fails or is too slow, we might return a conservative score
            # But for max_u < 50000, FFT should be fine.
            return 0.0

    if sumset_size == 0 or diffset_size == 0:
        return 1.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _score_set(U):
    """Wrapper for scoring."""
    try:
        return _score_set_fast(U)
    except Exception:
        # Fallback to original slow method if something breaks
        U_set = set(U)
        n = len(U_set)
        if n == 0:
            return 0.0
        max_u = max(U_set)
        if max_u == 0:
            return 1.0
        
        sumset = set()
        diffset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)
        sumset_size = len(sumset)
        diffset_size = len(diffset)
            
        if sumset_size == 0:
            return 0.0
            
        ratio = diffset_size / sumset_size
        if ratio <= 0:
            return 0.0
        if ratio == 1.0:
            return 1.0
            
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * max_u + 1)
        
        if log_denom == 0:
            return 1.0
            
        return 1.0 + log_ratio / log_denom

def _generate_initial_candidates(deadline=None):
    """Generate diverse initial candidates."""
    candidates = []
    
    # 1. Small Golomb Rulers
    grs = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 22],
    ]
    candidates.extend(grs)
    
    # 2. Prime Gap Sets (Cumulative sums of primes)
    def _is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0: return False
        return True
            
    primes = [p for p in range(2, 200) if _is_prime(p)]
    for n in range(3, 10):
        for combo in combinations(primes[:30], n-1):
            U = [0]
            curr = 0
            for p in combo:
                curr += p
                U.append(curr)
            if len(set(U)) == len(U):
                candidates.append(U)
            if deadline and time.time() > deadline:
                return candidates
                
    # 3. Random Sparse Sets
    random.seed(42)
    for _ in range(20):
        n = random.randint(10, 50)
        max_val = random.randint(100, 1000)
        U = [0] + random.sample(range(1, max_val), n-1)
        U.sort()
        candidates.append(U)
        
    return candidates

def _simulated_annealing_large(initial_U, deadline, max_iters=10000, max_size=2000, max_val_limit=5000):
    """
    Simulated Annealing allowing for larger sets.
    Uses fast scoring.
    """
    U = sorted(set(initial_U))
    best_U = U[:]
    best_score = _score_set(U)
    current_score = best_score
    current_U = U[:]
    
    T = 0.05
    cooling = 0.999
    # Dynamic search range
    current_max = max(current_U) if current_U else 0
    search_margin = 100
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Adjust temperature and search range
        current_max = max(current_U)
        search_range = current_max + search_margin
        if search_range > max_val_limit:
            search_range = max_val_limit
            
        move_type = random.random()
        new_U = current_U[:]
        
        # 1. Modify existing element (60%)
        if move_type < 0.6 and len(new_U) > 1:
            idx = random.randint(1, len(new_U) - 1)
            old_val = new_U[idx]
            # Perturb by a small amount or jump
            if random.random() < 0.5:
                step = random.randint(-10, 10)
                new_val = max(1, old_val + step)
            else:
                new_val = random.randint(1, search_range)
            
            if new_val != old_val and new_val not in new_U:
                new_U[idx] = new_val
                new_U.sort()
        
        # 2. Add new element (20%)
        elif move_type < 0.8 and len(new_U) < max_size:
            new_val = random.randint(1, search_range)
            if new_val not in new_U:
                new_U.append(new_val)
                new_U.sort()
        
        # 3. Remove element (20%)
        elif len(new_U) > 2:
            idx = random.randint(1, len(new_U) - 1)
            new_U.pop(idx)
        
        if len(new_U) < 2:
            T *= cooling
            continue
            
        new_score = _score_set(new_U)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-10)):
            current_U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = current_U[:]
                
        T *= cooling
        
        # Occasionally restart temperature to escape local minima
        if it % 500 == 0 and T < 0.01:
            T = 0.05
            
    return sorted(set(best_U)), best_score

def _local_search_large(U, deadline, max_iters=200, max_size=2000):
    """
    Greedy local search with add/swap/remove.
    """
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        
        # 1. Try swapping values
        if len(U) > 1:
            for i in range(1, len(U)):
                old_val = U[i]
                # Try a few random new values
                for _ in range(5):
                    new_val = random.randint(1, max(U) + 50)
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U[:]
                    new_U[i] = new_val
                    new_U.sort()
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U
                        improved = True
                        break
                if improved:
                    break
        
        # 2. Try adding a new element
        if not improved and len(U) < max_size:
            for _ in range(10):
                new_val = random.randint(1, max(U) + 100)
                if new_val in U:
                    continue
                new_U = U + [new_val]
                new_U.sort()
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U
                    improved = True
                    break
        
        # 3. Try removing an element
        if not improved and len(U) > 3:
            for _ in range(10):
                idx = random.randint(1, len(U) - 1)
                new_U = U[:idx] + U[idx+1:]
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U
                    improved = True
                    break
        
        if not improved:
            break
            
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # 1. Evaluate initial candidates
    candidates = _generate_initial_candidates(deadline=deadline)
    
    # Score all candidates and keep the best few for SA starts
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline:
            break
        s = _score_set(U)
        scored_candidates.append((s, U))
        
    if scored_candidates:
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        # Update global best
        if scored_candidates[0][0] > best_score:
            best_score = scored_candidates[0][0]
            best_U = scored_candidates[0][1][:]
            
    # 2. Run Simulated Annealing from top 3 candidates
    for s, U in scored_candidates[:3]:
        if time.time() > deadline - 20:
            break
        sa_U, sa_score = _simulated_annealing_large(U, deadline=deadline, max_iters=5000)
        if sa_score > best_score:
            best_score = sa_score
            best_U = sa_U
            
    # 3. Local Search on the best result
    if time.time() < deadline - 5:
        best_U, best_score = _local_search_large(best_U, deadline=deadline, max_iters=100)
        
    # 4. Another round of SA if time permits
    if time.time() < deadline - 10:
        sa_U, sa_score = _simulated_annealing_large(best_U, deadline=deadline, max_iters=3000)
        if sa_score > best_score:
            best_score = sa_score
            best_U = sa_U
            
    # 5. Final Polish
    if time.time() < deadline - 2:
        best_U, best_score = _local_search_large(best_U, deadline=deadline, max_iters=50)
        
    return sorted(set(best_U))

# EVOLVE-BLOCK-END
