import time
import math
import numpy as np
import random

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150s kill (5s buffer)

def _compute_sets_fft(u, max_val):
    """
    Compute |U+U| and |U-U| using FFT.
    u: list of nonnegative integers
    max_val: max(u)
    
    Returns: (size_sum, size_diff)
    """
    if not u:
        return 0, 0
    if len(u) == 1:
        return 1, 1
    
    M = max_val
    # N needs to be at least 2*M + 1 to avoid circular convolution aliasing
    N = 1
    while N < 2 * M + 1:
        N *= 2
    
    f = np.zeros(N, dtype=np.float64)
    for x in u:
        if 0 <= x < N:
            f[x] = 1.0
    
    F = np.fft.fft(f)
    
    # Sum set
    conv_sum = np.fft.ifft(F * F).real
    size_sum = np.sum(conv_sum[:2 * M + 1] > 0.5)
    
    # Difference set
    auto_diff = np.fft.ifft(F * np.conj(F)).real
    pos_diffs = np.sum(auto_diff[1:M + 1] > 0.5)
    size_diff = 1 + 2 * pos_diffs
    
    return int(size_sum), int(size_diff)

def _score(u):
    """
    Compute the score: 1 + log(|U-U|/|U+U|) / log(2*max U + 1)
    """
    if not u:
        return 0.0
    m = max(u)
    if m < 1:
        return 1.0
    
    size_sum, size_diff = _compute_sets_fft(u, m)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / float(size_sum)
    
    if ratio <= 0:
        return 0.0
        
    denom = math.log(2 * m + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def _fast_score_approx(u, max_val):
    """
    A faster approximate score for intermediate steps if needed, 
    but we will rely on the exact FFT score for the SA loop 
    because the set sizes are small enough (N < 200) that FFT is fast.
    """
    return _score(u)

def construct():
    global _DEADLINE
    _DEADLINE = max(time.time(), _START + 115.0)
    
    random.seed(42)
    
    # We search over a fixed range of max_val.
    # The score denominator is log(2*max_val + 1).
    # If we keep max_val small, the denominator is small, which is bad for the log(ratio) term 
    # if ratio < 1 (which it usually is, since |U-U| ~ |U|^2 and |U+U| ~ |U|^2/2, ratio ~ 2? 
    # Wait, for generic sets |U+U| ~ 2|U|^2/2 = |U|^2, |U-U| ~ |U|^2. Ratio is close to 1.
    # Actually, for random sets, |U+U| is roughly 2 * |U|^2 / 2 = |U|^2 (density 1/2 in [0, 2M]).
    # |U-U| is roughly |U|^2 (density 1 in [-M, M]).
    # So ratio |U-U|/|U+U| is roughly 2.
    # log(2) / log(2M+1).
    # To maximize, we want large ratio and small M.
    
    # Let's try to find a set with a "large" ratio.
    # Ratio > 1 means |U-U| > |U+U|.
    # This happens if the set is "sparse" in sums but "dense" in differences?
    # Actually, if U is an arithmetic progression, |U+U| ~ 2n, |U-U| ~ 2n. Ratio ~ 1.
    # If U is a Sidon set, |U+U| ~ n^2, |U-U| ~ n^2. Ratio ~ 1.
    # 
    # Let's look at the score formula again: 1 + log(ratio)/log(2M+1).
    # If ratio > 1, score > 1.
    # If ratio < 1, score < 1.
    # We need ratio > 1.
    # |U-U| / |U+U| > 1  => |U-U| > |U+U|.
    
    # Simulated Annealing parameters
    best_u = None
    best_s = -np.inf
    
    # We will run SA for a few different max_val bounds
    # Small max_val is preferred.
    # Let's try M in [10, 50, 100, 200]
    
    for M_max in [20, 50, 100, 200]:
        if time.time() > _DEADLINE - 10:
            break
            
        # Initialize a random set of size ~ M_max / 2
        # Must contain 0.
        n_init = max(2, M_max // 3)
        
        # Create a random set
        u = [0]
        # Randomly pick n_init - 1 from 1..M_max
        candidates = list(range(1, M_max + 1))
        random.shuffle(candidates)
        u.extend(candidates[:n_init - 1])
        u.sort()
        
        current_s = _score(u)
        
        if current_s > best_s:
            best_s = current_s
            best_u = list(u)
            
        # Simulated Annealing
        T = 0.1  # Initial temperature
        T_min = 0.001
        alpha = 0.99  # Cooling rate
        
        # Number of steps depends on time available
        # Each step involves a score calculation.
        # Let's estimate time per step.
        # FFT of size ~ 4*M_max is very fast.
        
        steps = 5000
        for step in range(steps):
            if time.time() > _DEADLINE - 5:
                break
                
            # Random move: add, remove, or swap an element
            move_type = random.choice(['add', 'remove', 'swap'])
            
            new_u = list(u)
            
            if move_type == 'add':
                # Add a random element from 1..M_max not in u
                if len(u) < M_max:
                    unused = [x for x in range(1, M_max + 1) if x not in set(u)]
                    if unused:
                        x = random.choice(unused)
                        new_u.append(x)
                        new_u.sort()
                    else:
                        continue
                else:
                    continue
            elif move_type == 'remove':
                # Remove a random element (not 0)
                if len(u) > 1:
                    # Pick index 1..len(u)-1
                    idx = random.randint(1, len(new_u) - 1)
                    new_u.pop(idx)
                    new_u.sort()
                else:
                    continue
            elif move_type == 'swap':
                # Remove one, add one
                if len(u) > 1:
                    idx = random.randint(1, len(new_u) - 1)
                    removed = new_u.pop(idx)
                    unused = [x for x in range(1, M_max + 1) if x not in set(new_u)]
                    if unused:
                        x = random.choice(unused)
                        new_u.append(x)
                        new_u.sort()
                    else:
                        continue
                else:
                    continue
            
            new_s = _score(new_u)
            
            # Acceptance probability
            delta_s = new_s - current_s
            if delta_s > 0:
                # Always accept improvement
                u = new_u
                current_s = new_s
                if current_s > best_s:
                    best_s = current_s
                    best_u = list(u)
            else:
                # Accept with probability exp(delta_s / T)
                # Note: delta_s is negative
                prob = math.exp(delta_s / T)
                if random.random() < prob:
                    u = new_u
                    current_s = new_s
            
            T *= alpha
            if T < T_min:
                T = T_min
        
        # Restart with a new random initialization for this M_max
        # To allow more exploration, we do a few restarts
        for restart in range(3):
            if time.time() > _DEADLINE - 5:
                break
                
            n_init = max(2, M_max // 3 + random.randint(-2, 2))
            u = [0]
            candidates = list(range(1, M_max + 1))
            random.shuffle(candidates)
            u.extend(candidates[:n_init - 1])
            u.sort()
            
            current_s = _score(u)
            
            if current_s > best_s:
                best_s = current_s
                best_u = list(u)
            
            T = 0.1
            steps = 2000
            for step in range(steps):
                if time.time() > _DEADLINE - 5:
                    break
                    
                move_type = random.choice(['add', 'remove', 'swap'])
                new_u = list(u)
                
                if move_type == 'add':
                    if len(u) < M_max:
                        unused = [x for x in range(1, M_max + 1) if x not in set(u)]
                        if unused:
                            x = random.choice(unused)
                            new_u.append(x)
                            new_u.sort()
                        else:
                            continue
                    else:
                        continue
                elif move_type == 'remove':
                    if len(u) > 1:
                        idx = random.randint(1, len(new_u) - 1)
                        new_u.pop(idx)
                        new_u.sort()
                    else:
                        continue
                elif move_type == 'swap':
                    if len(u) > 1:
                        idx = random.randint(1, len(new_u) - 1)
                        removed = new_u.pop(idx)
                        unused = [x for x in range(1, M_max + 1) if x not in set(new_u)]
                        if unused:
                            x = random.choice(unused)
                            new_u.append(x)
                            new_u.sort()
                        else:
                            continue
                    else:
                        continue
                
                new_s = _score(new_u)
                
                delta_s = new_s - current_s
                if delta_s > 0:
                    u = new_u
                    current_s = new_s
                    if current_s > best_s:
                        best_s = current_s
                        best_u = list(u)
                else:
                    prob = math.exp(delta_s / T)
                    if random.random() < prob:
                        u = new_u
                        current_s = new_s
                
                T *= alpha
                if T < T_min:
                    T = T_min

    # Final polish: local search on the best found set
    if best_u and time.time() < _DEADLINE - 2:
        # Try to improve by removing largest element if it doesn't hurt
        u_trial = list(best_u)
        if len(u_trial) > 1:
            u_trial.pop()
            s_trial = _score(u_trial)
            if s_trial > best_s:
                best_s = s_trial
                best_u = u_trial
        
        # Try swapping large elements with smaller unused ones
        M_current = max(best_u)
        used = set(best_u)
        improved = True
        while improved and time.time() < _DEADLINE - 1:
            improved = False
            for i in range(len(best_u) - 1, 0, -1): # from largest to smallest (excluding 0)
                x = best_u[i]
                # Try replacing x with a smaller unused y
                # Only try if y < x
                candidates = [y for y in range(1, x) if y not in used]
                if not candidates:
                    continue
                # Sample a few
                random.shuffle(candidates)
                for y in candidates[:10]:
                    new_u = list(best_u)
                    new_u[i] = y
                    new_u.sort()
                    s_new = _score(new_u)
                    if s_new > best_s:
                        best_s = s_new
                        best_u = new_u
                        used = set(best_u)
                        improved = True
                        break
                if improved:
                    break

    # Ensure valid output
    if not best_u:
        return [0, 1, 2]
        
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
        
    return best_u
# EVOLVE-BLOCK-END
