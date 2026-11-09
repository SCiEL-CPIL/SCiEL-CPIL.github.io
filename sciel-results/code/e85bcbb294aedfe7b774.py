"""Optimized program: Hybrid structured construction + SA with FFT-based score evaluation."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from math import log

def _compute_score_fft(U, fft_cache=None):
    """Compute the score for a given set U using FFT for speed."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n == 0 or 0 not in U:
        return 0.0
    max_u = int(U[-1])
    if max_u == 0:
        return 1.0
    
    # Use FFT to compute sumset and diffset sizes efficiently
    size = 2 * max_u + 1
    indicator = np.zeros(size, dtype=np.float64)
    indicator[U] = 1.0
    
    # Find appropriate FFT size
    fft_size = 1
    while fft_size < 2 * size:
        fft_size *= 2
    
    f = np.fft.rfft(indicator, n=fft_size)
    conv = np.fft.irfft(f * f, n=fft_size)
    sumset_mask = conv[:2 * max_u + 1] > 0.5
    len_sum = int(np.sum(sumset_mask))
    
    # Autocorrelation via FFT for diffset
    autocorr = np.fft.irfft(f * np.conj(f), n=fft_size)
    diff_counts = np.zeros(2 * max_u + 1, dtype=np.float64)
    diff_counts[max_u] = autocorr[0]
    for d in range(1, max_u + 1):
        diff_counts[max_u + d] = autocorr[d]
        diff_counts[max_u - d] = autocorr[d]
    
    len_diff = int(np.sum(diff_counts > 0.5))
    
    if len_sum == 0:
        return 0.0
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
        
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
        
    return 1 + log(ratio) / denominator

def _compute_score_direct(U):
    """Compute score directly (slower but exact, good for small sets)."""
    U = sorted(set(U))
    n = len(U)
    if n == 0 or 0 not in U:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    
    # Direct computation
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    
    len_sum = len(sumset)
    len_diff = len(diffset)
    
    if len_sum == 0:
        return 0.0
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
        
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
        
    return 1 + log(ratio) / denominator

def _generate_greedy_sidon(max_val, target_size):
    """Generate a greedy Sidon-like set."""
    U = {0}
    # Try to build a set where all pairwise sums are distinct (Sidon property)
    # This tends to maximize |U-U|/|U+U|
    for val in range(1, max_val + 1):
        if len(U) >= target_size:
            break
        # Check if adding val maintains good properties
        new_U = U | {val}
        # Quick check: ensure no immediate conflicts
        valid = True
        for a in U:
            for b in U:
                if a + b == val or b + val == a:  # Simple heuristic
                    pass  # We'll just add greedily
        U.add(val)
    return sorted(U)

def _generate_lacunary(max_val, target_size):
    """Generate a lacunary (rapidly growing) set."""
    U = [0]
    current = 1
    while len(U) < target_size and current <= max_val:
        U.append(current)
        current = current * 2 + random.randint(0, max(1, current // 4))
    # If we didn't reach target_size, fill in with random
    while len(U) < target_size:
        U.append(random.randint(1, max_val))
    return sorted(set(U))

def _generate_geometric(max_val, target_size):
    """Generate a geometric progression set."""
    if target_size <= 1:
        return [0]
    # Find a base that fits
    base = max(1, int((max_val) ** (1.0 / (target_size - 1))))
    U = [0]
    for i in range(1, target_size):
        U.append(base ** i)
    # If max exceeds max_val, scale down
    if U[-1] > max_val:
        scale = max_val / U[-1]
        U = sorted(set([int(x * scale) for x in U]))
        if 0 not in U:
            U.insert(0, 0)
    return sorted(U)

def _generate_hybrid(max_val, target_size):
    """Generate a hybrid set combining different growth patterns."""
    U = {0}
    # Start with geometric-like growth
    current = 1
    for i in range(min(target_size - 1, 8)):
        U.add(current)
        current = current * random.choice([2, 3, 4]) + random.randint(0, 2)
        if current > max_val:
            break
    # Fill remaining with random
    while len(U) < target_size:
        U.add(random.randint(1, max_val))
    return sorted(U)

def _sa_search(initial_set, max_val, target_size, timeout=30.0):
    """Simulated Annealing to find a good set."""
    start_time = time.time()
    
    # Initialize
    current = list(initial_set)
    if 0 not in current:
        current.append(0)
    current = sorted(set(current))
    current_score = _compute_score_fft(current)
    
    best = list(current)
    best_score = current_score
    
    T = 0.15  # Initial temperature - slightly higher for better exploration
    T_min = 1e-6
    cooling = 0.9998  # Slower cooling for more iterations
    
    while time.time() - start_time < timeout:
        if T < T_min:
            break
        
        # Generate a neighbor with more diverse moves
        move_type = random.random()
        
        if move_type < 0.35:
            # Add a random element
            new_elem = random.randint(0, max_val)
            if new_elem not in current:
                new_set = list(current) + [new_elem]
            else:
                continue
        elif move_type < 0.55:
            # Remove a random element (not 0)
            removable = [x for x in current if x != 0]
            if not removable:
                continue
            rem = random.choice(removable)
            new_set = [x for x in current if x != rem]
        elif move_type < 0.75:
            # Swap: remove one, add another
            removable = [x for x in current if x != 0]
            if not removable:
                continue
            rem = random.choice(removable)
            add = random.randint(0, max_val)
            new_set = [x for x in current if x != rem]
            if add not in new_set:
                new_set.append(add)
        elif move_type < 0.90:
            # Perturb: change a random element to a nearby value
            if len(current) <= 1:
                continue
            idx = random.randint(0, len(current) - 1)
            if current[idx] == 0:
                idx = random.randint(1, len(current) - 1) if len(current) > 1 else 0
            old_val = current[idx]
            # Larger perturbation range
            delta = random.choice([-10, -5, -3, -2, -1, 1, 2, 3, 5, 10])
            new_val = old_val + delta
            if new_val < 0 or new_val > max_val:
                new_val = random.randint(0, max_val)
            new_set = list(current)
            new_set[idx] = new_val
            if len(set(new_set)) != len(new_set):
                continue
        else:
            # Insert at a specific position to maintain structure
            if len(current) < 2:
                continue
            # Pick a gap and insert
            gaps = []
            for i in range(len(current) - 1):
                if current[i + 1] - current[i] > 1:
                    gaps.append((current[i], current[i + 1]))
            if not gaps:
                continue
            low, high = random.choice(gaps)
            if high - low <= 1:
                continue
            new_elem = random.randint(low + 1, high - 1)
            if new_elem not in current:
                new_set = list(current) + [new_elem]
            else:
                continue
        
        new_set = sorted(set(new_set))
        if 0 not in new_set or len(new_set) < 2:
            continue
        
        new_score = _compute_score_fft(new_set)
        
        # Acceptance criterion
        if new_score >= current_score or random.random() < np.exp((new_score - current_score) / T):
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = list(current)
        
        T *= cooling
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Start with a reasonable initial set
    best_result = [0, 1, 2, 4]
    best_score = _compute_score_fft(best_result)
    
    # Try structured constructions first
    structured_inits = []
    
    # Lacunary sets
    for max_val in [100, 200, 500, 1000]:
        for target_size in [8, 12, 15, 20]:
            structured_inits.append(_generate_lacunary(max_val, target_size))
    
    # Geometric sets
    for max_val in [100, 200, 500, 1000]:
        for target_size in [8, 12, 15]:
            structured_inits.append(_generate_geometric(max_val, target_size))
    
    # Hybrid sets
    for max_val in [100, 200, 500]:
        for target_size in [10, 15, 20]:
            structured_inits.append(_generate_hybrid(max_val, target_size))
    
    # Greedy Sidon-like
    for max_val in [100, 200, 500]:
        for target_size in [10, 15, 20]:
            structured_inits.append(_generate_greedy_sidon(max_val, target_size))
    
    # Evaluate all structured inits and keep the best few
    scored_inits = []
    for init in structured_inits:
        if not init or 0 not in init or len(init) < 2:
            continue
        score = _compute_score_fft(init)
        scored_inits.append((score, init))
    
    scored_inits.sort(key=lambda x: -x[0])
    
    # Update best with top structured constructions
    for score, init in scored_inits[:5]:
        if score > best_score:
            best_score = score
            best_result = list(init)
    
    # Now run SA from the top structured constructions
    top_inits = [init for _, init in scored_inits[:8]]
    
    # Add some random inits for diversity
    for _ in range(5):
        max_val = random.choice([100, 200, 500, 1000])
        target_size = random.randint(8, 25)
        top_inits.append(_generate_hybrid(max_val, target_size))
    
    # Run SA from each top init
    for init in top_inits:
        if time.time() > deadline - 15:
            break
        
        if not init or 0 not in init or len(init) < 2:
            continue
        
        max_u = max(init)
        max_val = min(int(max_u * 1.5) + 50, 2000)
        
        remaining = deadline - time.time()
        sa_timeout = min(12.0, remaining * 0.4)
        
        if sa_timeout < 2.0:
            break
        
        result, score = _sa_search(init, max_val, len(init), timeout=sa_timeout)
        
        if score > best_score:
            best_score = score
            best_result = result
    
    # Final refinement: run SA from the best result found with more time
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        max_u_best = max(best_result) if best_result else 0
        max_val_refine = min(int(max_u_best * 2) + 100, 3000)
        result, score = _sa_search(best_result, max_val_refine, len(best_result), timeout=min(25.0, remaining - 5))
        if score > best_score:
            best_score = score
            best_result = result
    
    # One more refinement with even wider range
    if time.time() < deadline - 5:
        remaining = deadline - time.time()
        max_u_best = max(best_result) if best_result else 0
        max_val_refine = min(int(max_u_best * 3) + 200, 5000)
        result, score = _sa_search(best_result, max_val_refine, len(best_result), timeout=min(15.0, remaining - 2))
        if score > best_score:
            best_score = score
            best_result = result
    
    # Final check
    final_score = _compute_score_fft(best_result)
    if final_score < 1.0:
        best_result = [0, 1, 2, 4]
        
    return best_result
# EVOLVE-BLOCK-END
