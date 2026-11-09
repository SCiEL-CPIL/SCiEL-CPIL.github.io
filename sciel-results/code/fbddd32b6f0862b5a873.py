import math
import time
import random
import numpy as np

def _score(U):
    """Exact score calculation."""
    U = list(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    plus = set()
    minus = set()
    for a in U:
        for b in U:
            plus.add(a + b)
            minus.add(a - b)
    n_plus = len(plus)
    n_minus = len(minus)
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def _fast_score(U):
    """Fast score using FFT."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    
    size = 2 * mx + 1
    nfft = 1
    while nfft < size:
        nfft *= 2
    
    vec = np.zeros(nfft)
    for x in U:
        vec[x] = 1.0
    
    fft_vec = np.fft.fft(vec)
    conv_sum = np.fft.ifft(fft_vec * fft_vec)
    n_plus = int(np.sum(conv_sum.real > 0.5))
    
    rev_vec = np.zeros(nfft)
    for x in U:
        rev_vec[mx - x] = 1.0
    
    fft_rev = np.fft.fft(rev_vec)
    conv_diff = np.fft.ifft(fft_vec * fft_rev)
    n_minus = int(np.sum(conv_diff.real > 0.5))
    
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def _sa_binary(M, time_limit=5.0, T_start=0.1, T_end=0.001):
    """
    Simulated Annealing on a fixed binary vector of length M+1.
    Bit 0 is always 1.
    Returns the best set found.
    """
    # Initial state: random subset or small set
    # Start with a small set to ensure non-trivial ratio
    # Or start with random bits
    mask = 0
    mask |= 1  # 0 is always in U
    
    # Random initialization
    for i in range(1, M + 1):
        if random.random() < 0.3:
            mask |= (1 << i)
            
    def get_set(mask):
        s = set()
        for i in range(M + 1):
            if mask & (1 << i):
                s.add(i)
        return list(s)
        
    current_set = get_set(mask)
    current_score = _fast_score(current_set)
    best_set = list(current_set)
    best_score = current_score
    
    start_time = time.time()
    max_iter = 100000
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        progress = it / max_iter
        T = T_start * (T_end / T_start) ** progress
        
        # Pick a random bit to flip (not bit 0)
        i = random.randint(1, M)
        
        # Flip bit i
        new_mask = mask ^ (1 << i)
        new_set = get_set(new_mask)
        
        # Ensure set is not empty (it has 0, so it's never empty)
        # Calculate score
        new_score = _fast_score(new_set)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            mask = new_mask
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
                
    return best_set, best_score

def construct():
    deadline = time.time() + 130
    best_U = [0, 1, 2, 5]
    best_score = _fast_score(best_U)
    
    # Try a range of M values
    # Small M is favored due to denominator, but need enough elements for ratio > 1
    # Let's try M from 5 to 100
    # We can be smarter: try M that are powers of 2 or specific values
    
    m_values = list(range(5, 101))
    # Sort by potential? Smaller M first?
    # Actually, let's just iterate.
    
    # To save time, we can do fewer iterations for larger M
    # Or just let the time limit handle it
    
    for M in m_values:
        if time.time() > deadline - 10:
            break
            
        # Allocate time based on M?
        # Smaller M should get more time? Or uniform?
        # Let's give uniform time but stop if we've found a very good score.
        # Score 1.1 is very good.
        
        # We can also try multiple random restarts for small M
        # But for now, just one SA run per M
        
        U_sa, s_sa = _sa_binary(M, time_limit=2.0)
        if s_sa > best_score:
            best_score = s_sa
            best_U = U_sa
            
    # Final polish on the best U found
    # Use a generic local search (add/remove/shift) on the best U
    # This can adjust M slightly up or down
    
    U_polished = best_U
    s_polished = best_score
    start = time.time()
    while time.time() - start < 10:
        U_polished, s_polished = _local_search(U_polished, s_polished, time_limit=5.0)
        if s_polished <= best_score + 1e-9:
            break
        best_score = s_polished
        best_U = U_polished
        
    # Verify with exact score
    exact_score = _score(best_U)
    if exact_score > best_score:
        best_score = exact_score
        
    return sorted(set(best_U))

def _local_search(U, current_score, time_limit=5.0):
    """Local search with add/remove/shift moves."""
    U = sorted(set(U))
    best_U = list(U)
    best_score = current_score
    start = time.time()
    
    while time.time() - start < time_limit:
        # Try random moves
        mx = max(U) if U else 0
        move_type = random.choice(['add', 'remove', 'shift'])
        
        if move_type == 'add':
            if mx == 0:
                new_elem = 1
            else:
                new_elem = random.randint(1, mx + 10)
            if new_elem not in U:
                candidate = sorted(U + [new_elem])
                s = _fast_score(candidate)
                if s > best_score:
                    best_score = s
                    best_U = candidate
                    U = candidate
        elif move_type == 'remove':
            removable = [x for x in U if x != 0]
            if removable:
                rem = random.choice(removable)
                candidate = [x for x in U if x != rem]
                s = _fast_score(candidate)
                if s > best_score:
                    best_score = s
                    best_U = candidate
                    U = candidate
        elif move_type == 'shift':
            shiftable = [x for x in U if x != 0]
            if shiftable:
                old = random.choice(shiftable)
                delta = random.choice([-2, -1, 1, 2, 3, 5])
                new = old + delta
                if new >= 1 and new not in U:
                    candidate = sorted([x if x != old else new for x in U])
                    s = _fast_score(candidate)
                    if s > best_score:
                        best_score = s
                        best_U = candidate
                        U = candidate
                        
    return best_U, best_score
