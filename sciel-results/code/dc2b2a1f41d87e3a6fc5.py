import time
import random
import math
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin for 150s limit
    
    def compute_score_fft(U):
        """Compute score using FFT for O(M log M) set size computation."""
        if not U or len(U) <= 1:
            return 1.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        n = len(U)
        
        # For small sets, direct computation is faster
        if n <= 30:
            sumset = set()
            diffset = set()
            for i in range(n):
                for j in range(i, n):
                    s = U[i] + U[j]
                    sumset.add(s)
                    if i != j:
                        d = U[i] - U[j]
                        diffset.add(d)
                        diffset.add(-d)
            diffset.add(0)
            size_sum = len(sumset)
            size_diff = len(diffset)
        else:
            # FFT-based computation
            # Sumset: convolution of indicator with itself
            # f = indicator of U, sumset size = number of nonzero entries in f*f
            size = 2 * max_u + 1
            # Next power of 2 for FFT
            fft_size = 1
            while fft_size < size:
                fft_size *= 2
            
            f = np.zeros(fft_size, dtype=np.float64)
            for u in U:
                f[u] = 1.0
            
            # FFT for sumset
            F = np.fft.fft(f)
            F2 = F * F
            conv = np.fft.ifft(F2).real
            
            # Count nonzero entries (sumset)
            # Due to floating point, use threshold
            threshold = 0.5
            size_sum = int(np.sum(conv[:size] > threshold))
            
            # Difference set: |U-U| = number of distinct differences
            # This is related to autocorrelation
            # autocorr[k] = sum_j f[j] * f[j+k] for k >= 0
            # |U-U| = 1 + 2 * sum_{k>0, autocorr[k]>0} 1
            # Actually: |U-U| = number of distinct values in {a-b : a,b in U}
            # = number of k such that exists a,b in U with a-b=k
            # = number of k in [-max_u, max_u] where autocorr is nonzero
            
            # Compute autocorrelation via FFT
            # autocorr[k] = sum_j f[j+k] * f[j]
            # This is the cross-correlation of f with itself
            F_conj = np.conj(F)
            autocorr = np.fft.ifft(F * F_conj).real
            
            # autocorr is circular; we need linear autocorrelation
            # For k in [0, fft_size), autocorr[k] gives sum_j f[j] * f[(j+k) % fft_size]
            # We want sum_j f[j+k] * f[j] for k in [-max_u, max_u]
            # Note: autocorr[-k] = autocorr[k] for real signals
            
            # The linear autocorrelation at lag k is:
            # For k >= 0: sum_{j=0}^{max_u-k} f[j+k]*f[j]
            # For k < 0: sum_{j=0}^{max_u+k} f[j+k]*f[j] = sum_{j=0}^{max_u-k} f[j]*f[j-k]
            
            # Using FFT: ifft(F * conj(F)) gives circular autocorrelation
            # For our case where fft_size > 2*max_u, circular = linear for |k| <= max_u
            
            # autocorr[k] for k in [0, fft_size/2] gives sum_j f[j]*f[j+k] (circular)
            # But we need to be careful about the indexing
            
            # Let's compute directly: diffset size = number of distinct (a-b) for a,b in U
            # This equals the number of nonzero entries in the "difference polynomial"
            # D(x) = (sum_{a in U} x^a) * (sum_{b in U} x^{-b})
            # = sum_{a,b in U} x^{a-b}
            # The number of distinct exponents with nonzero coefficient = |U-U|
            
            # Using FFT: let g[x] = f[-x mod fft_size] (time-reversed)
            # Then D = f * g (convolution)
            g = np.zeros(fft_size, dtype=np.float64)
            for u in U:
                if u == 0:
                    g[0] = 1.0
                else:
                    g[fft_size - u] = 1.0
            
            G = np.fft.fft(g)
            D = np.fft.ifft(F * G).real
            
            # D[k] for k in [0, fft_size) represents coefficient of x^k in D(x)
            # But we want exponents in [-max_u, max_u]
            # k=0 -> exponent 0
            # k in [1, max_u] -> exponent k
            # k in [fft_size-max_u, fft_size-1] -> exponent k - fft_size (negative)
            
            # Count nonzero entries
            diff_nonzero = 0
            for k in range(0, max_u + 1):
                if abs(D[k]) > threshold:
                    diff_nonzero += 1
            for k in range(fft_size - max_u, fft_size):
                if abs(D[k]) > threshold:
                    diff_nonzero += 1
            
            size_diff = diff_nonzero
        
        if size_sum == 0:
            return 1.0
            
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
        
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        return 1.0 + math.log(ratio) / denom

    def compute_score_direct(U):
        """Fast direct score computation for small sets."""
        if not U or len(U) <= 1:
            return 1.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        n = len(U)
        sumset = set()
        diffset = set()
        for i in range(n):
            for j in range(i, n):
                sumset.add(U[i] + U[j])
                if i != j:
                    diffset.add(U[i] - U[j])
                    diffset.add(U[j] - U[i])
        diffset.add(0)
        size_sum = len(sumset)
        size_diff = len(diffset)
        if size_sum == 0:
            return 1.0
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
        return 1.0 + math.log(ratio) / denom

    best_score = 0.0
    best_set = [0]

    # 1. Exhaustive Search for Small Sets (Structural Change: Deterministic Optimality)
    # We search for the best set of size n for n=2,3,4,5,6 within a bounded range.
    # This guarantees we don't miss a small, high-scoring configuration.
    max_search_val = 60
    for n in range(2, 7):
        if time.time() > deadline:
            break
        # Generate combinations of size n from range(1, max_search_val)
        # Note: 0 is always included. So we pick n-1 positive integers.
        # To keep it manageable, we restrict the max value for larger n.
        search_max = max_search_val
        if n >= 5:
            search_max = 40
        if n >= 6:
            search_max = 30
            
        # Use combinations with replacement? No, set elements must be distinct.
        # combinations(range(1, search_max+1), n-1)
        # This can be large. Let's estimate:
        # n=2: C(39, 1) = 39
        # n=3: C(39, 2) = 741
        # n=4: C(39, 3) = 9139
        # n=5: C(39, 4) = 82251
        # n=6: C(29, 5) = 118755
        # Total ~ 210k evaluations. Each direct eval is fast (O(n^2)).
        # 210k * 36 ops ~ 7.5M ops. Should be < 1s.
        
        for combo in combinations(range(1, search_max + 1), n - 1):
            U = [0] + list(combo)
            # Optional pruning: if max(U) is too large relative to n, skip?
            # The score denominator grows with log(max U).
            # For fixed n, smaller max U is generally better if ratio is similar.
            # But we need to check.
            
            score = compute_score_direct(U)
            if score > best_score:
                best_score = score
                best_set = U

    # 2. Greedy Construction (Structural Change: Constructive Search)
    # Build a set greedily by adding elements that maximize score improvement.
    # This explores larger sets more intelligently than random.
    def greedy_construct(max_elements, max_value):
        current_set = [0]
        current_score = 1.0
        for _ in range(max_elements - 1):
            if time.time() > deadline:
                break
            best_new_score = current_score
            best_new_elem = None
            # Try adding each candidate element
            for candidate in range(1, max_value + 1):
                if candidate in current_set:
                    continue
                new_set = sorted(current_set + [candidate])
                score = compute_score_direct(new_set)
                if score > best_new_score:
                    best_new_score = score
                    best_new_elem = candidate
            
            if best_new_elem is None:
                break
            current_set.append(best_new_elem)
            current_set.sort()
            current_score = best_new_score
            
        return current_set, current_score

    # Try greedy construction with different parameters
    greedy_configs = [
        (10, 100),
        (15, 150),
        (20, 200),
    ]
    
    for max_elems, max_val in greedy_configs:
        if time.time() > deadline:
            break
        g_set, g_score = greedy_construct(max_elems, max_val)
        if g_score > best_score:
            best_score = g_score
            best_set = g_set

    # 3. Local Search from Best Found (Hybrid: Keep existing strength)
    # Use the best set found so far as a seed for local search to refine it.
    # This is kept from the original logic but starts from a potentially better seed.
    
    def local_search(seed_set, max_iter=2000):
        current_set = list(seed_set)
        current_score = compute_score_direct(current_set)
        no_improve = 0
        max_no_improve = 100
        
        while time.time() < deadline and no_improve < max_no_improve and max_iter > 0:
            max_iter -= 1
            neighbor = list(current_set)
            op = random.randint(0, 3)
            
            if op == 0 and len(neighbor) > 1:
                idx = random.randint(1, len(neighbor) - 1)
                neighbor.pop(idx)
                if len(neighbor) == 0:
                    continue
                if 0 not in neighbor:
                    neighbor.append(0)
                    neighbor.sort()
            elif op == 1:
                max_val = max(neighbor) if neighbor else 0
                upper_bound = max(2 * max_val + 15, 25)
                new_val = random.randint(1, upper_bound)
                if new_val not in neighbor:
                    neighbor.append(new_val)
                    neighbor.sort()
            elif op == 2:
                if len(neighbor) > 1:
                    idx = random.randint(1, len(neighbor) - 1)
                    old_val = neighbor[idx]
                    found = False
                    for delta in [-3, -2, -1, 1, 2, 3]:
                        new_val = old_val + delta
                        if new_val >= 0 and new_val not in neighbor:
                            neighbor[idx] = new_val
                            neighbor.sort()
                            found = True
                            break
                    if not found:
                        continue
            else:
                if len(neighbor) > 1:
                    idx = random.randint(1, len(neighbor) - 1)
                    removed_val = neighbor.pop(idx)
                    max_val = max(neighbor) if neighbor else 0
                    upper_bound = max(2 * max_val + 15, 25)
                    new_val = random.randint(1, upper_bound)
                    if new_val not in neighbor:
                        neighbor.append(new_val)
                        neighbor.sort()
                    else:
                        neighbor.append(removed_val)
                        neighbor.sort()
                        continue
            
            if not neighbor or 0 not in neighbor:
                continue
                
            new_score = compute_score_direct(neighbor)
            
            if new_score > current_score:
                current_set = neighbor
                current_score = new_score
                no_improve = 0
            else:
                no_improve += 1
                if random.random() < 0.01:
                    current_set = neighbor
                    current_score = new_score
        
        return current_set, current_score

    # Run local search from the best found
    ls_set, ls_score = local_search(best_set)
    if ls_score > best_score:
        best_score = ls_score
        best_set = ls_set

    # Run local search from a few random Golomb rulers if time remains
    golomb_rulers = [
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
    ]
    
    for gr in golomb_rulers:
        if time.time() > deadline:
            break
        ls_set, ls_score = local_search(gr)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set

    return best_set
# EVOLVE-BLOCK-END
