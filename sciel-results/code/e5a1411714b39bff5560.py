import numpy as np
import time
import random

N = 50

def compute_ratio(h):
    h = np.array(h, dtype=float)
    if np.all(h == 0):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    # Linear convolution
    conv = np.convolve(h, h, mode='full')
    l_inf = np.max(np.abs(conv))
    if l1 * l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_h = None
    best_score = 0.0
    
    # Candidate 1: Single spike
    h1 = [0.0] * N
    h1[0] = 1.0
    s1 = compute_ratio(h1)
    if s1 > best_score:
        best_score = s1
        best_h = h1[:]
        
    # Candidate 2: Two spikes
    h2 = [0.0] * N
    h2[0] = 1.0
    h2[N//2] = 1.0
    s2 = compute_ratio(h2)
    if s2 > best_score:
        best_score = s2
        best_h = h2[:]
        
    # Candidate 3: Random sparse set
    for _ in range(100):
        if time.time() > deadline:
            break
        k = random.randint(1, 10)
        indices = random.sample(range(N), k)
        h = [0.0] * N
        for idx in indices:
            h[idx] = 1.0
        s = compute_ratio(h)
        if s > best_score:
            best_score = s
            best_h = h[:]
            
    # Local search on best_h
    if best_h is not None:
        current_h = best_h[:]
        current_score = best_score
        no_improve_count = 0
        
        while time.time() < deadline:
            improved = False
            # Try moving a 1 to a new position
            non_zero_indices = [i for i, val in enumerate(current_h) if val > 1e-9]
            if non_zero_indices:
                i = random.choice(non_zero_indices)
                new_i = random.randint(0, N-1)
                if new_i != i:
                    old_val = current_h[i]
                    current_h[i] = 0.0
                    current_h[new_i] = old_val
                    new_score = compute_ratio(current_h)
                    if new_score > current_score + 1e-12:
                        current_score = new_score
                        best_score = current_score
                        best_h = current_h[:]
                        improved = True
                        no_improve_count = 0
                    else:
                        # Revert
                        current_h[new_i] = 0.0
                        current_h[i] = old_val
                        no_improve_count += 1
                        if no_improve_count > 1000:
                            break
            # Try adding a new 1
            if time.time() < deadline:
                new_i = random.randint(0, N-1)
                if current_h[new_i] < 1e-9:
                    old_val = current_h[new_i]
                    current_h[new_i] = 1.0
                    new_score = compute_ratio(current_h)
                    if new_score > current_score + 1e-12:
                        current_score = new_score
                        best_score = current_score
                        best_h = current_h[:]
                        improved = True
                        no_improve_count = 0
                    else:
                        current_h[new_i] = old_val
                        no_improve_count += 1
                        if no_improve_count > 1000:
                            break
            # Try removing a 1
            if time.time() < deadline and len(non_zero_indices) > 1:
                i = random.choice(non_zero_indices)
                old_val = current_h[i]
                current_h[i] = 0.0
                new_score = compute_ratio(current_h)
                if new_score > current_score + 1e-12:
                    current_score = new_score
                    best_score = current_score
                    best_h = current_h[:]
                    improved = True
                    no_improve_count = 0
                else:
                    current_h[i] = old_val
                    no_improve_count += 1
                    if no_improve_count > 1000:
                        break
                        
            if not improved and no_improve_count > 5000:
                break

    if best_h is None:
        return [1.0] * N
    return best_h
