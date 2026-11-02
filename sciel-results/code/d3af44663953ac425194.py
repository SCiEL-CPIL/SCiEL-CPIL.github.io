"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from numpy.fft import fft, ifft
from scipy.fft import rfft, irfft

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
    # U + U: values range from 0 to 2*max_u
    sum_len = 2 * max_u + 1
    fft_len = _next_pow2(2 * sum_len)
    
    # Create indicator vector for U
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    # FFT to compute convolution (sumset)
    freq = fft(indicator)
    conv = ifft(freq * freq).real
    
    # Count nonzero entries in sumset (values 0 to 2*max_u)
    sumset_size = int(np.sum(conv[:sum_len] > 1e-9))
    
    # Compute difference set using FFT
    # U - U: values range from -max_u to max_u
    # Use autocorrelation: place U at offset max_u, then autocorrelate
    diff_len = 2 * max_u + 1
    fft_len2 = _next_pow2(2 * diff_len)
    
    # Place U in a vector of length fft_len2, shifted by max_u
    diff_vec = np.zeros(fft_len2, dtype=np.float64)
    for u in U:
        diff_vec[u + max_u] = 1.0
    
    # Autocorrelation via FFT
    freq_d = fft(diff_vec)
    diff_result = ifft(freq_d * np.conj(freq_d)).real
    
    # The difference d = a - b appears at position (max_u + d) in diff_result
    # So differences range from -max_u to max_u, appearing at positions 0 to 2*max_u
    diffset_size = int(np.sum(np.abs(diff_result[:diff_len]) > 1e-9))
    
    ratio = diffset_size / sumset_size
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
    
    # For small sets, use direct computation
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

def _local_search(U, max_iters=5000, deadline=None):
    """Improve set U via local search (swap elements)."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 200  # search space for swaps 
    
    no_improve_count = 0
    max_no_improve = 50
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        if no_improve_count >= max_no_improve:
            break
        
        improved = False
        # Try swapping one element - focus on promising moves first
        indices_to_try = list(range(len(U)))
        random.shuffle(indices_to_try)
        
        for i in indices_to_try:
            if U[i] == 0:
                continue  # must keep 0
            old_val = U[i]
            # Try several new values around old_val and random positions
            candidates = set()
            for offset in range(-30, 31):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            # Add some random candidates
            for _ in range(15):
                candidates.add(random.randint(1, upper_bound))
            
            # Sort by distance from old_val to try closest first
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
            # Try adding an element
            if len(U) < 300:
                for _ in range(30):
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
                # Try removing an element (if size > 2)
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
            # Try a larger random perturbation
            if random.random() < 0.15:
                new_U = U[:]
                # Randomly change a few elements
                num_changes = random.randint(2, min(8, len(new_U) - 1))
                for _ in range(num_changes):
                    idx = random.randint(1, len(new_U) - 1)  # skip index 0 (value 0)
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

def _simulated_annealing(U, start_temp=10.0, end_temp=0.01, max_iters=2000, deadline=None):
    """Simulated annealing to escape local minima."""
    U = sorted(set(U))
    current_score = _score_set(U)
    current_U = U[:]
    best_score = current_score
    best_U = U[:]
    
    max_u = max(U)
    upper_bound = max_u + 500
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Cooling schedule
        progress = it / max_iters
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Generate neighbor
        new_U = current_U[:]
        move_type = random.random()
        
        if move_type < 0.5:
            # Swap one element
            idx = random.randint(1, len(new_U) - 1)
            new_U[idx] = random.randint(1, upper_bound)
        elif move_type < 0.7:
            # Add an element
            if len(new_U) < 500:
                new_val = random.randint(1, upper_bound)
                if new_val not in new_U:
                    new_U.append(new_val)
                else:
                    continue
        elif move_type < 0.85:
            # Remove an element
            if len(new_U) > 3:
                idx = random.randint(1, len(new_U) - 1)
                new_U.pop(idx)
            else:
                continue
        else:
            # Swap two elements
            if len(new_U) >= 3:
                i, j = random.sample(range(1, len(new_U)), 2)
                new_U[i], new_U[j] = new_U[j], new_U[i]
            else:
                continue
        
        # Validate
        new_U = sorted(set(new_U))
        if 0 not in new_U or len(new_U) != len(set(new_U)):
            continue
        
        new_score = _score_set(new_U)
        delta = new_score - current_score
        
        # Accept if better or with probability exp(-delta/temp)
        if delta > 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
            current_score = new_score
            current_U = new_U[:]
            if current_score > best_score:
                best_score = current_score
                best_U = current_U[:]
    
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Try different initial constructions
    candidates = []
    
    # Golomb rulers of various sizes
    for n in range(3, 40):
        if time.time() > deadline - 60:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)
    
    # Random sets with various sizes
    random.seed(42)
    for trial in range(40):
        if time.time() > deadline - 60:
            break
        n = random.randint(3, 60)
        max_val = random.randint(30, 800)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Structured sets: arithmetic progressions with gaps
    for base in [2, 3, 5, 7, 11, 13]:
        if time.time() > deadline - 60:
            break
        U = [0]
        for i in range(1, 25):
            U.append(i * base)
        candidates.append(U[:min(len(U), 25)])
    
    # Powers-based sets
    for p in [2, 3, 4]:
        if time.time() > deadline - 60:
            break
        U = [0]
        for i in range(1, 20):
            U.append(p ** i)
            if U[-1] > 20000:
                break
        candidates.append(U)
    
    # Sparse random sets (higher sparsity)
    for trial in range(20):
        if time.time() > deadline - 60:
            break
        n = random.randint(5, 40)
        max_val = random.randint(100, 2000)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Evaluate candidates and keep top ones
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline - 50:
            break
        s = _score_set(U)
        scored_candidates.append((s, U))
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Local search on best candidate
    if time.time() < deadline - 20:
        best_U, best_score = _local_search(best_U, max_iters=8000, deadline=deadline - 20)
    
    # Try local search on top few other candidates
    for s, U in scored_candidates[:10]:
        if time.time() > deadline - 20:
            break
        if U == best_U:
            continue
        if s > best_score - 0.05:  # only if reasonably close
            improved_U, improved_score = _local_search(U, max_iters=3000, deadline=deadline - 20)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    # Simulated annealing to escape local minima
    if time.time() < deadline - 10:
        sa_U, sa_score = _simulated_annealing(best_U, start_temp=10.0, end_temp=0.01, max_iters=3000, deadline=deadline - 10)
        if sa_score > best_score:
            best_score = sa_score
            best_U = sa_U
    
    # Final local search polish
    if time.time() < deadline - 5:
        best_U, best_score = _local_search(best_U, max_iters=5000, deadline=deadline - 5)
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
