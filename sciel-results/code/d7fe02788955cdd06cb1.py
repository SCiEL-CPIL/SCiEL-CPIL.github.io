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
    # N needs to be at least 2*M + 1 to avoid circular convolution aliasing for sums [0, 2M]
    # and differences [-M, M]
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
    # Valid range for sums is [0, 2*M]
    size_sum = np.sum(conv_sum[:2 * M + 1] > 0.5)
    
    # Difference set
    # Autocorrelation r[k] = sum_x f[x] f[x+k]
    auto_diff = np.fft.ifft(F * np.conj(F)).real
    # r[k] for k in [1, M] corresponds to positive differences
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

def _generate_sidon_set(n):
    """
    Generate a Sidon set (Golomb ruler) of size n using a greedy approach.
    This provides a good initial candidate with high difference/sum ratio.
    """
    if n <= 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    u = [0]
    last = 0
    for i in range(1, n):
        # Try to find the smallest x > last such that {x + u_j - u_k} doesn't collide with existing diffs
        # Simple greedy: just pick a large enough jump to ensure new diffs
        # A known construction for small n is sufficient.
        # For n up to 20, we can use a lookup or a simple greedy that checks validity.
        found = False
        x = last + 1
        while not found:
            # Check if adding x creates duplicate differences
            valid = True
            new_diffs = set()
            for j in range(len(u)):
                for k in range(len(u)):
                    d = u[j] + x - u[k]
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    break
            # Also check against existing diffs? 
            # Actually, Sidon means all a+b are distinct. 
            # Equivalent to all a-b are distinct (except for sign).
            # Greedy: find smallest x such that x - u_i are not in existing diffs and don't collide with each other.
            if valid:
                u.append(x)
                last = x
                found = True
            else:
                x += 1
        # Optimization: The greedy above is O(n^3 * M). For small n it's fine.
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
                if len(neighbor) != len(current_u):
                    # Try to restore size if it changed
                    if len(neighbor) < len(current_u):
                        while len(neighbor) < len(current_u):
                            candidate = random.randint(0, max_val)
                            if candidate not in neighbor:
                                neighbor.append(candidate)
                                break
                    else:
                        non_zero_indices = [j for j, v in enumerate(neighbor) if v != 0]
                        if non_zero_indices:
                            rm_idx = random.choice(non_zero_indices)
                            neighbor.pop(rm_idx)
                            neighbor = sorted(neighbor)
                    if len(neighbor) != len(current_u):
                        continue
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
    
    candidates = []
    
    # 1. Generate small Sidon sets (Golomb rulers)
    # These are known to have good properties for this objective.
    for n in range(3, 15):
        try:
            s = _generate_sidon_set(n)
            candidates.append(s)
        except:
            pass
            
    # 2. Add some random sets with small M
    for _ in range(20):
        M = random.randint(5, 50)
        n = random.randint(3, min(10, M+1))
        if n > M+1: continue
        u = [0] + random.sample(range(1, M+1), n-1)
        candidates.append(sorted(u))
        
    # 3. Evaluate all candidates
    best_u = candidates[0] if candidates else [0, 1, 3]
    best_s = _score(best_u)
    
    for u in candidates:
        s = _score(u)
        if s > best_s:
            best_s = s
            best_u = u
            
    # 4. Local search from the best candidate
    if time.time() < _DEADLINE - 10.0:
        m_best = max(best_u)
        # Allow M to grow slightly to allow better swaps
        M_search = int(m_best * 2.0) + 20
        # Run SA
        u_polished, s_polished = _local_search(best_u, M_search, 2000, temperature_start=1.0, cooling=0.995)
        if s_polished > best_s:
            best_s = s_polished
            best_u = u_polished
            
    # 5. Final check and cleanup
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
        
    return best_u
# EVOLVE-BLOCK-END
