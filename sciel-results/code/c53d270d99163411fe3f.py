import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 140.0  # Safety margin vs 150 s kill

def _fft_score(u):
    """
    Compute the score using FFT to handle large sets efficiently.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    
    We represent the set U as a binary vector f of length L.
    |U+U| is the number of non-zero elements in f * f (convolution).
    |U-U| is the number of non-zero elements in f * f_rev (convolution with reversed f).
    
    Note: For |U-U|, the convolution f * f_rev gives the autocorrelation.
    Specifically, (f * f_rev)[k] = sum_i f[i] f_rev[i-k] = sum_i f[i] f[i-k+L-1].
    Wait, standard convolution: (f * g)[k] = sum_i f[i] g[k-i].
    Let g = reverse(f). g[j] = f[L-1-j].
    (f * g)[k] = sum_i f[i] f[L-1-(k-i)] = sum_i f[i] f[L-1-k+i].
    Let j = L-1-k+i => i = j - (L-1-k).
    Sum over i such that f[i]!=0 and f[L-1-k+i]!=0.
    This counts pairs (i, j) such that j - i = L-1-k.
    So the index in the convolution corresponds to the difference d = j - i.
    The range of differences is -(L-1) to L-1.
    The convolution result will have length 2L-1.
    The center index L-1 corresponds to difference 0.
    The number of non-zero entries in the convolution gives the count of pairs (i,j) 
    that produce each difference.
    The number of distinct differences is the number of non-zero entries in this convolution.
    
    Similarly for sums:
    (f * f)[k] = sum_i f[i] f[k-i].
    Counts pairs (i,j) such that i+j=k.
    Number of distinct sums is the number of non-zero entries in this convolution.
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    # We need a grid size L such that all sums and differences fit.
    # Max sum = 2m. Max diff = m (if min is 0).
    # Convolution length is 2L-1.
    # We need indices 0..2m for sums to be covered.
    # We need indices corresponding to diffs -m..m to be covered.
    # If we use a vector of length L, indices 0..L-1.
    # Sums range 0..2(L-1).
    # Diffs range -(L-1)..(L-1).
    # So if 2m < 2(L-1), i.e., m < L-1, we are safe.
    # Let L = m + 1.
    
    L = m + 1
    
    # To speed up FFT, we can pad to next power of 2 or just let numpy handle it.
    # numpy fft is efficient.
    
    f = np.zeros(L, dtype=np.float64)
    for val in u:
        if val < L:
            f[val] = 1.0
        else:
            # Should not happen if L = m+1
            pass
    
    # Compute FFT
    F = np.fft.fft(f)
    
    # Sumset: f * f
    # Convolution in time domain is product in freq domain
    # But we need the inverse FFT of F^2
    F_sum = F * F
    conv_sum = np.fft.ifft(F_sum).real
    
    # Diffset: f * f_rev
    # f_rev is reverse of f
    f_rev = f[::-1]
    F_rev = np.fft.fft(f_rev)
    F_diff = F * F_rev
    conv_diff = np.fft.ifft(F_diff).real
    
    # Count non-zero elements
    # Due to floating point errors, use a threshold
    eps = 1e-6
    
    # Sums: indices 0 to 2L-2
    # The convolution result length is 2L-1.
    # np.fft.ifft returns length L if input is length L? 
    # No, np.fft.fft returns length L. 
    # Product of two length L arrays is length L.
    # ifft returns length L.
    # This is circular convolution!
    # We need linear convolution.
    # We must pad f to length at least 2L-1 to avoid wrap-around.
    
    # Let's redo with padding
    L_pad = 2 * L - 1
    # Next power of 2 for speed?
    L_fft = 1
    while L_fft < L_pad:
        L_fft *= 2
    
    f_pad = np.zeros(L_fft, dtype=np.float64)
    f_pad[:L] = f
    
    F_pad = np.fft.fft(f_pad)
    
    # Sumset
    F_sum_pad = F_pad * F_pad
    conv_sum_pad = np.fft.ifft(F_sum_pad).real
    
    # Diffset
    f_rev_pad = np.zeros(L_fft, dtype=np.float64)
    f_rev_pad[:L] = f_rev
    F_rev_pad = np.fft.fft(f_rev_pad)
    F_diff_pad = F_pad * F_rev_pad
    conv_diff_pad = np.fft.ifft(F_diff_pad).real
    
    # The valid part of the convolution is the first 2L-1 elements
    # Sums: 0..2L-2
    # Diffs: The convolution of f and f_rev gives autocorrelation.
    # The result at index k (0-indexed) in the linear convolution of length 2L-1:
    # (f * f_rev)[k] = sum_i f[i] f_rev[k-i]
    # f_rev[j] = f[L-1-j]
    # (f * f_rev)[k] = sum_i f[i] f[L-1-(k-i)] = sum_i f[i] f[L-1-k+i]
    # Let d = L-1-k+i => i = d - (L-1-k)
    # This counts pairs (i, j) such that j - i = L-1-k.
    # So index k corresponds to difference d = L-1-k.
    # Range of k: 0..2L-2.
    # Range of d: L-1 .. -(L-1).
    # We just need the count of non-zero entries.
    
    valid_len = 2 * L - 1
    
    nonzero_sums = np.sum(np.abs(conv_sum_pad[:valid_len]) > eps)
    nonzero_diffs = np.sum(np.abs(conv_diff_pad[:valid_len]) > eps)
    
    if nonzero_sums == 0:
        return 0.0
        
    ratio = nonzero_diffs / float(nonzero_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _initial_set(n, max_val):
    """
    Generate an initial set with some structure.
    Mix of geometric and random.
    """
    if n < 2:
        return [0, 1]
    
    vals = [0]
    current = 0
    
    # Geometric backbone
    growth = random.uniform(1.2, 1.8)
    for i in range(1, n):
        next_val = int(current * growth * random.uniform(0.9, 1.1))
        if next_val <= current:
            next_val = current + 1
        if next_val > max_val:
            next_val = current + 1
            if next_val > max_val:
                break
        vals.append(next_val)
        current = next_val
        
    # Fill remaining with random small increments
    while len(vals) < n:
        next_val = vals[-1] + random.randint(1, max(1, max_val // n))
        if next_val > max_val:
            next_val = vals[-1] + 1
        if next_val in vals:
            continue
        vals.append(next_val)
        
    return sorted(set(vals))

def _simulated_annealing(u, max_iter=20000, T_start=1.0, T_end=0.01, timeout=100.0):
    """
    Simulated Annealing on the set.
    Moves:
    1. Replace an element.
    2. Add an element.
    3. Remove an element.
    """
    u = sorted(set(u))
    if len(u) < 2:
        u = [0, 1]
        
    current_score = _fft_score(u)
    best_u = u[:]
    best_score = current_score
    
    start_time = time.time()
    n_iter = 0
    
    while n_iter < max_iter:
        if time.time() > _DEADLINE - 5.0 or time.time() > start_time + timeout:
            break
            
        # Temperature schedule
        T = T_start * (T_end / T_start) ** (n_iter / max_iter)
        
        # Choose move
        move_type = random.choice(['replace', 'replace', 'add', 'remove'])
        new_u = u[:]
        
        if move_type == 'replace':
            if len(u) < 2:
                continue
            idx = random.randint(1, len(u) - 1)
            old_val = u[idx]
            # Perturb
            delta = random.randint(-max(1, int(math.sqrt(old_val))), max(1, int(math.sqrt(old_val))))
            new_val = old_val + delta
            if new_val < 0:
                new_val = 0
            if new_val in u:
                continue
            new_u[idx] = new_val
            new_u = sorted(set(new_u))
            if len(new_u) != len(u):
                continue
        elif move_type == 'add':
            if len(u) > 500:  # Cap size
                continue
            max_val = max(u)
            # Add near existing or extend
            if random.random() < 0.5:
                new_val = max_val + random.randint(1, max(1, max_val//50))
            else:
                idx = random.randint(0, len(u)-1)
                new_val = u[idx] + random.randint(-max(1, u[idx]//10), max(1, u[idx]//10))
            
            if new_val < 0 or new_val in u:
                continue
            new_u.append(new_val)
            new_u = sorted(set(new_u))
        elif move_type == 'remove':
            if len(u) < 3:
                continue
            idx = random.randint(1, len(u) - 1)
            new_u.pop(idx)
        
        new_score = _fft_score(new_u)
        
        # Acceptance criterion
        if new_score > current_score:
            u = new_u
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_u = u[:]
        else:
            # Metropolis criterion
            if random.random() < math.exp((new_score - current_score) / T):
                u = new_u
                current_score = new_score
        
        n_iter += 1
        
    return best_u, best_score

def construct():
    global _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    best_u = [0, 1]
    best_s = 0.0
    
    # Phase 1: Generate diverse initial candidates
    # Focus on larger sizes and ranges to exploit FFT speed
    sizes = [50, 100, 200, 300, 500]
    max_vals = [1000, 5000, 10000, 50000, 100000]
    
    candidates = []
    
    # Geometric progressions with noise
    for base in [1.1, 1.2, 1.5]:
        for n in [50, 100, 200]:
            u = [0]
            curr = 1
            for i in range(1, n):
                curr = int(curr * base * random.uniform(0.95, 1.05))
                u.append(curr)
            candidates.append(u)
            
    # Random structured sets
    for n in sizes:
        for mv in max_vals:
            if time.time() > _DEADLINE - 30.0:
                break
            u = _initial_set(n, mv)
            candidates.append(u)
            
    # Evaluate candidates
    for u in candidates:
        if time.time() > _DEADLINE - 10.0:
            break
        s = _fft_score(u)
        if s > best_s:
            best_s = s
            best_u = u[:]
            
    # Phase 2: Simulated Annealing from the best candidate
    if time.time() < _DEADLINE - 5.0:
        u_new, s_new = _simulated_annealing(best_u, max_iter=50000, T_start=0.5, T_end=0.01, timeout=120.0)
        if s_new > best_s:
            best_s = s_new
            best_u = u_new[:]
            
    # Final sanity check
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
    if len(best_u) < 2:
        best_u = [0, 1]
        
    return best_u
# EVOLVE-BLOCK-END
