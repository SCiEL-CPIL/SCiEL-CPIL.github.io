import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150 s kill

def _score_from_vector(vec, max_val):
    """
    Compute the score for a set represented by a binary indicator vector.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    n = np.sum(vec)
    if n < 2 or max_val < 1:
        return 0.0
    
    L = len(vec)
    required_len = 2 * (max_val + 1)
    if L < required_len:
        vec = np.pad(vec, (0, required_len - L), mode='constant')
        L = len(vec)
        
    fft_len = 1
    while fft_len < 2 * L - 1:
        fft_len *= 2
        
    f = np.fft.fft(vec, n=fft_len)
    
    # Sumset: |U+U|
    conv = np.fft.ifft(f * f)
    sums = np.abs(conv.real) > 1e-9
    num_sums = int(np.sum(sums))
    
    # Differenceset: |U-U|
    corr = np.fft.ifft(np.conj(f) * f)
    diffs = np.abs(corr.real) > 1e-9
    num_diffs = int(np.sum(diffs))
    
    if num_sums == 0:
        return 0.0
        
    ratio = num_diffs / float(num_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * max_val + 1)

def _list_to_vec(lst, length):
    """Convert list of indices to binary vector of given length."""
    vec = np.zeros(length, dtype=np.float64)
    for i in lst:
        if i < length:
            vec[i] = 1.0
    return vec

def _generate_ap_set(start, step, n, max_val):
    """Generate an arithmetic progression set."""
    u = []
    for i in range(n):
        val = start + i * step
        if val > max_val:
            break
        u.append(val)
    if 0 not in u:
        u.append(0)
    return sorted(set(u))

def _generate_sparse_ap_set(base, n, max_val):
    """Generate a set with sparse arithmetic progression structure."""
    # Start with 0, then add elements in a pattern that creates large gaps
    u = [0]
    current = 1
    for i in range(1, n):
        # Alternate between small and large steps to create structure
        if i % 3 == 0:
            current = min(current * base, max_val)
        elif i % 3 == 1:
            current = min(current + int(base ** (i // 3)), max_val)
        else:
            current = min(current + 1, max_val)
        if current > max_val:
            break
        if current not in u:
            u.append(current)
    return sorted(u)

def _generate_golomb_like(n, max_val):
    """Generate a Golomb-like sequence (distinct differences)."""
    u = [0]
    next_val = 1
    diffs = set()
    while len(u) < n and next_val <= max_val:
        # Check if adding next_val creates duplicate differences
        valid = True
        for x in u:
            d = next_val - x
            if d in diffs or -d in diffs:
                valid = False
                break
        if valid:
            u.append(next_val)
            for x in u[:-1]:
                diffs.add(next_val - x)
                diffs.add(x - next_val)
        next_val += 1
    return u

def _local_search_sa(u, max_val_limit, time_limit):
    """
    Simulated annealing local search.
    Moves: move one element to a new position, or swap two elements.
    """
    current_u = sorted(u)
    current_max = max(current_u)
    fixed_vec_len = 2 * (max_val_limit + 1)
    
    current_vec = _list_to_vec(current_u, fixed_vec_len)
    current_score = _score_from_vector(current_vec, current_max)
    
    best_score = current_score
    best_u = current_u[:]
    
    start_time = time.time()
    initial_temp = 0.1
    cooling_rate = 0.999
    
    while time.time() < time_limit:
        temp = initial_temp * (cooling_rate ** (int((time.time() - start_time) * 100)))
        
        idx = random.randint(0, len(current_u) - 1)
        old_val = current_u[idx]
        
        # Choose move type: 70% move, 30% swap
        if random.random() < 0.7:
            # Move to new position
            new_val = random.randint(0, max_val_limit)
            if new_val == old_val or new_val in current_u:
                continue
            new_u = current_u[:]
            new_u[idx] = new_val
        else:
            # Swap two elements
            if len(current_u) < 2:
                continue
            jdx = random.randint(0, len(current_u) - 1)
            if jdx == idx:
                continue
            new_u = current_u[:]
            new_u[idx], new_u[jdx] = new_u[jdx], new_u[idx]
        
        new_u.sort()
        new_max = max(new_u)
        
        new_vec = _list_to_vec(new_u, fixed_vec_len)
        new_score = _score_from_vector(new_vec, new_max)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / max(temp, 1e-10)):
            current_u = new_u
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_u = new_u[:]
                
    return best_u, best_score

def construct():
    global _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    best_score = 0.0
    best_set = [0, 1]
    
    # Strategy:
    # 1. Generate diverse initial sets with additive structure.
    # 2. Refine using simulated annealing with FFT-based scoring.
    # 3. Keep the best result.
    
    configs = [
        # (n, M_limit, description)
        (15, 2000, "small_n"),
        (30, 10000, "med_n"),
        (50, 50000, "large_n"),
        (80, 200000, "very_large_n"),
    ]
    
    for n, M_limit, desc in configs:
        if time.time() > _DEADLINE - 15.0:
            break
            
        candidates = []
        
        # 1. Arithmetic progression variants
        for step in [1, 2, 3, 5]:
            u_ap = _generate_ap_set(0, step, n, M_limit)
            if len(u_ap) >= 2:
                candidates.append(u_ap)
                
        # 2. Sparse AP structure
        for base in [2, 3, 4]:
            u_sparse = _generate_sparse_ap_set(base, n, M_limit)
            if len(u_sparse) >= 2:
                candidates.append(u_sparse)
                
        # 3. Golomb-like
        try:
            u_golomb = _generate_golomb_like(n, M_limit)
            if len(u_golomb) >= 2:
                candidates.append(u_golomb)
        except:
            pass
            
        # 4. Random sorted
        for _ in range(2):
            others = random.sample(range(1, M_limit + 1), min(n - 1, M_limit))
            u_rand = [0] + sorted(others)
            candidates.append(u_rand)
            
        # Refine each candidate
        num_candidates = len(candidates)
        time_per_candidate = max(0.5, (115.0 - (time.time() - _START)) / (num_candidates + 2))
        
        for u_init in candidates:
            if time.time() > _DEADLINE - 5.0:
                break
                
            if 0 not in u_init:
                u_init = [0] + u_init
            u_init = sorted(set(u_init))
            
            if len(u_init) > 150:
                others = [x for x in u_init if x != 0]
                random.shuffle(others)
                u_init = [0] + others[:150]
                
            u_opt, score_opt = _local_search_sa(u_init, M_limit, time_per_candidate)
            
            if score_opt > best_score:
                best_score = score_opt
                best_set = u_opt
                
    # Final sanity check
    best_set = sorted(set(int(x) for x in best_set))
    if 0 not in best_set:
        best_set.append(0)
    if len(best_set) < 2:
        best_set = [0, 1]
        
    return best_set
# EVOLVE-BLOCK-END
