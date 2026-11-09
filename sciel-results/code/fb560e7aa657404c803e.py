# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from math import log

def compute_score_fast(U):
    """Compute score using FFT for sumset, O(n^2) for diffset (small n)."""
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Sumset via FFT
    size = 2 * max_u + 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    
    indicator = np.zeros(fft_size)
    for u in U:
        indicator[u] = 1.0
    
    fft_ind = np.fft.fft(indicator)
    conv = np.fft.ifft(fft_ind * fft_ind)
    # Round to handle floating point
    conv_real = np.real(conv)
    # Count entries > 0.5 (i.e., at least one pair sums to that value)
    sumset_count = int(np.sum(np.round(conv_real) > 0))
    
    if sumset_count == 0:
        return 0.0
    
    # Difference set
    diffset = set()
    n = len(U)
    for i in range(n):
        for j in range(n):
            diffset.add(U[i] - U[j])
    diffset_count = len(diffset)
    
    ratio = diffset_count / sumset_count
    if ratio <= 0:
        return 0.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def compute_score_exact(U):
    """Exact score computation, slower but reliable."""
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
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
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def local_search(U, time_limit=10, max_val=200):
    """Local search: try small perturbations to improve score."""
    start_time = time.time()
    U = sorted(set(U))
    best_U = list(U)
    best_score = compute_score_fast(U)
    current_U = list(U)
    current_score = best_score
    
    while (time.time() - start_time) < time_limit:
        improved = False
        n = len(current_U)
        
        # Try shifting each element (except 0)
        for i in range(1, n):
            if (time.time() - start_time) > time_limit:
                break
            old_val = current_U[i]
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_val = old_val + delta
                if new_val < 1 or new_val > max_val or new_val == old_val:
                    continue
                if new_val in current_U:
                    continue
                
                new_U = list(current_U)
                new_U[i] = new_val
                new_U.sort()
                
                new_score = compute_score_fast(new_U)
                if new_score > current_score:
                    current_U = new_U
                    current_score = new_score
                    improved = True
                    if new_score > best_score:
                        best_score = new_score
                        best_U = list(new_U)
                    break  # Restart search from new position
        
        # Try adding a new element
        if not improved and n < 20:
            tried = 0
            while tried < 50:
                tried += 1
                if (time.time() - start_time) > time_limit:
                    break
                new_val = random.randint(1, max_val)
                if new_val in current_U:
                    continue
                
                new_U = list(current_U)
                new_U.append(new_val)
                new_U.sort()
                
                new_score = compute_score_fast(new_U)
                if new_score > current_score:
                    current_U = new_U
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_U = list(new_U)
                    break
        
        # Try removing an element
        if not improved and n > 3:
            for i in range(1, n):
                if (time.time() - start_time) > time_limit:
                    break
                new_U = list(current_U)
                new_U.pop(i)
                
                new_score = compute_score_fast(new_U)
                if new_score > current_score:
                    current_U = new_U
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_U = list(new_U)
                    break
    
    return best_U, best_score

def simulated_annealing(U, time_limit=20, max_val=150):
    """Simulated annealing with better acceptance."""
    start_time = time.time()
    U = sorted(set(U))
    best_U = list(U)
    best_score = compute_score_fast(U)
    current_U = list(U)
    current_score = best_score
    
    T = 0.5
    cooling = 0.995
    
    while (time.time() - start_time) < time_limit:
        if T < 0.001:
            break
        
        n = len(current_U)
        pert_type = random.random()
        new_U = None
        
        if pert_type < 0.5:
            # Shift an element
            if n >= 2:
                i = random.randint(1, n - 1)
                old_val = current_U[i]
                delta = random.randint(-5, 5)
                new_val = old_val + delta
                if new_val >= 1 and new_val <= max_val and new_val != old_val and new_val not in current_U:
                    new_U = list(current_U)
                    new_U[i] = new_val
                    new_U.sort()
        
        elif pert_type < 0.7:
            # Add element
            if n < 18:
                new_val = random.randint(1, max_val)
                if new_val not in current_U:
                    new_U = list(current_U)
                    new_U.append(new_val)
                    new_U.sort()
        
        elif pert_type < 0.85:
            # Remove element
            if n > 3:
                i = random.randint(1, n - 1)
                new_U = list(current_U)
                new_U.pop(i)
        
        else:
            # Swap two elements
            if n >= 3:
                i, j = random.sample(range(1, n), 2)
                new_U = list(current_U)
                new_U[i], new_U[j] = new_U[j], new_U[i]
                new_U.sort()
        
        if new_U is None or len(new_U) < 2:
            T *= cooling
            continue
        
        new_score = compute_score_fast(new_U)
        
        if new_score >= current_score:
            current_U = new_U
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_U = list(new_U)
        else:
            delta = current_score - new_score
            if delta < 5 * T:
                prob = np.exp(-delta / max(T, 1e-10))
                if random.random() < prob:
                    current_U = new_U
                    current_score = new_score
        
        T *= cooling
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_U = [0, 1, 3]
    best_score = compute_score_fast(best_U)
    
    # Try known good starting points - Golomb rulers and variations
    starting_sets = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 18, 23],
        [0, 1, 7, 12, 18, 23, 31],
        [0, 1, 7, 12, 18, 25, 34, 41],
        [0, 1, 7, 12, 18, 25, 34, 44, 51],
        [0, 1, 7, 12, 18, 25, 34, 44, 52, 62],
        # Some non-Golomb sets that might have better sumset properties
        [0, 1, 3, 6, 9, 12, 15],
        [0, 1, 2, 4, 7, 11, 16],
        [0, 1, 3, 6, 10, 15, 21],
        [0, 2, 3, 7, 10, 14, 19],
    ]
    
    # Evaluate all starting sets
    for start_set in starting_sets:
        if time.time() > deadline - 5:
            break
        U = sorted(set(start_set))
        if 0 not in U or len(U) < 2:
            continue
        score = compute_score_fast(U)
        if score > best_score:
            best_score = score
            best_U = list(U)
    
    # Run local search on best
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 10
        if remaining > 3:
            improved_U, improved_score = local_search(best_U, time_limit=remaining, max_val=100)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    # Run simulated annealing from multiple starts
    for start_set in starting_sets[:6]:
        if time.time() > deadline - 10:
            break
        U = sorted(set(start_set))
        if 0 not in U or len(U) < 2:
            continue
        
        remaining = deadline - time.time() - 5
        if remaining > 5:
            improved_U, improved_score = simulated_annealing(U, time_limit=remaining, max_val=120)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    # Final polish with local search on best
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 2
        if remaining > 2:
            improved_U, improved_score = local_search(best_U, time_limit=remaining, max_val=80)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    return best_U
# EVOLVE-BLOCK-END
