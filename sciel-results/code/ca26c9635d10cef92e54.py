import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 110.0  # safety margin vs 150 s kill

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
    # N must be at least 2*M + 1 to avoid wrap-around in convolution
    N = 1
    while N < 2 * M + 1:
        N *= 2
    
    # Create indicator array
    f = np.zeros(N, dtype=np.float64)
    for x in u:
        if 0 <= x < N:
            f[x] = 1.0
    
    # FFT
    F = np.fft.fft(f)
    
    # Sum set: convolution f * f
    conv_sum = np.fft.ifft(F * F).real
    # The valid range for sums is [0, 2*M].
    # We count k in [0, 2*M] where conv_sum[k] > 0.5.
    size_sum = np.sum(conv_sum[:2 * M + 1] > 0.5)
    
    # Difference set: autocorrelation
    # r[k] = sum_x f[x] f[x+k]
    # We need to count distinct differences.
    # Since differences are symmetric, |U-U| = 1 + 2 * count(k in [1, M] such that r[k] > 0).
    auto_diff = np.fft.ifft(F * np.conj(F)).real
    
    # Count positive differences
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

def _random_initial_set(n, max_val):
    """
    Generate a random set of size n containing 0, within [0, max_val].
    """
    if n <= 0:
        return [0]
    if n == 1:
        return [0]
    
    others = random.sample(range(1, max_val + 1), n - 1)
    u = [0] + others
    return sorted(u)

def _local_search(u, max_val, iterations, temperature_start=1.0, cooling=0.999):
    """
    Simulated Annealing local search.
    """
    current_u = list(u)
    current_score = _score(current_u)
    best_u = list(current_u)
    best_score = current_score
    
    temperature = temperature_start
    
    for i in range(iterations):
        if time.time() >= _DEADLINE:
            break
            
        neighbor = list(current_u)
        move_type = random.random()
        
        if move_type < 0.7:
            # Replace a random element (not 0, to keep 0)
            if len(neighbor) > 1:
                idx = random.randint(1, len(neighbor) - 1)
                new_val = random.randint(0, max_val)
                neighbor[idx] = new_val
                neighbor = sorted(set(neighbor))
                
                # Maintain size n if possible
                if len(neighbor) != len(current_u):
                    # If size decreased, try to add a random element
                    while len(neighbor) < len(current_u):
                        candidate = random.randint(0, max_val)
                        if candidate not in neighbor:
                            neighbor.append(candidate)
                            break
                    if len(neighbor) != len(current_u):
                        continue
                    # If size increased (shouldn't happen with replace), remove random non-zero
                    if len(neighbor) > len(current_u):
                        non_zero_indices = [j for j, v in enumerate(neighbor) if v != 0]
                        if non_zero_indices:
                            rm_idx = random.choice(non_zero_indices)
                            neighbor.pop(rm_idx)
                            neighbor = sorted(neighbor)
            else:
                continue
        else:
            # Perturb: add or remove an element
            if move_type < 0.85:
                # Add
                candidate = random.randint(0, max_val)
                if candidate not in neighbor:
                    neighbor.append(candidate)
                    neighbor = sorted(neighbor)
            else:
                # Remove
                if len(neighbor) > 1:
                    non_zero_indices = [j for j, v in enumerate(neighbor) if v != 0]
                    if non_zero_indices:
                        rm_idx = random.choice(non_zero_indices)
                        neighbor.pop(rm_idx)
                        neighbor = sorted(neighbor)
                else:
                    continue
        
        if not neighbor:
            continue
            
        neighbor_score = _score(neighbor)
        
        # Acceptance probability
        delta = neighbor_score - current_score
        if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-9)):
            current_u = neighbor
            current_score = neighbor_score
            
            if current_score > best_score:
                best_u = list(current_u)
                best_score = current_score
        
        temperature *= cooling
        
    return best_u, best_score

def construct():
    global _DEADLINE
    _DEADLINE = max(time.time(), _START + 110.0)
    
    random.seed(42)
    
    # Known good candidates from inspirations and general knowledge
    # Golomb rulers have distinct differences, which maximizes |U-U|.
    # We want to minimize |U+U| relative to |U-U|.
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 10, 12, 17, 20, 22],
        # Some random-looking sets that might be good
        [0, 1, 3, 7, 12, 18],
        [0, 2, 5, 11, 15, 21],
    ]
    
    best_u = candidates[0]
    best_s = _score(best_u)
    
    # Evaluate all candidates
    for u in candidates:
        s = _score(u)
        if s > best_s:
            best_s = s
            best_u = u
    
    # Run SA phases to improve
    num_phases = 200
    phase_iter = 300
    
    for phase in range(num_phases):
        if time.time() >= _DEADLINE - 10.0:
            break
            
        # Choose n and M
        # Bias towards small M to keep denominator small
        n = random.randint(3, 50)
        M = random.randint(n, 5 * n)
        
        # Initial candidate
        u_init = _random_initial_set(n, M)
        
        # Run SA
        u_best, s_best = _local_search(u_init, M, phase_iter, temperature_start=1.0, cooling=0.99)
        
        if s_best > best_s:
            best_s = s_best
            best_u = u_best
    
    # Final polish on the best found
    if time.time() < _DEADLINE - 5.0:
        m_best = max(best_u)
        # Try to polish with a slightly larger M to allow for better structure
        u_polished, s_polished = _local_search(best_u, m_best * 2, 1000, temperature_start=0.5, cooling=0.995)
        if s_polished > best_s:
            best_s = s_polished
            best_u = u_polished
            
    # Ensure 0 is in best_u and all are non-negative integers
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
        
    return best_u
# EVOLVE-BLOCK-END
