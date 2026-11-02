import numpy as np
import time

N = 100

def compute_autocorr(h):
    H = np.fft.fft(h)
    C = np.fft.ifft(H * np.conj(H)).real
    return C

def project(h):
    lo, hi = 0.0, 1.0
    target = N / 2.0
    
    for _ in range(40):
        mid = (lo + hi) / 2.0
        proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(proj)
        if s > target:
            lo = mid
        else:
            hi = mid
            
    return np.clip(h - (lo + hi) / 2.0, 0.0, 1.0)

def local_search_binary(h0):
    h = h0.copy()
    zeros = np.where(h == 0)[0]
    ones = np.where(h == 1)[0]
    
    C = compute_autocorr(h)
    C[0] = -np.inf
    min_val = np.min(C)
    
    improved = True
    while improved:
        improved = False
        k_min = np.argmin(C)
        
        score_i = np.roll(h, -k_min) + np.roll(h, k_min)
        score_j = np.roll(h, -k_min) + np.roll(h, k_min)
        
        best_gain = 0
        best_i = -1
        best_j = -1
        
        max_si = -1
        for i in zeros:
            if score_i[i] > max_si:
                max_si = score_i[i]
                best_i = i
                
        min_sj = 2
        for j in ones:
            if score_j[j] < min_sj:
                min_sj = score_j[j]
                best_j = j
                
        if best_i != -1 and best_j != -1:
            gain = score_i[best_i] - score_j[best_j]
            if gain > 0:
                h[best_i] = 1
                h[best_j] = 0
                improved = True
                C = compute_autocorr(h)
                C[0] = -np.inf
                min_val = np.min(C)
                zeros = np.where(h == 0)[0]
                ones = np.where(h == 1)[0]
                
    return h, min_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.full(N, 0.5)
    best_val = -np.inf
    
    # Binary Local Search
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        h0 = np.random.choice([0.0, 1.0], N)
        sum_h = np.sum(h0)
        target = N / 2
        if sum_h > target:
            ones = np.where(h0 == 1)[0]
            to_remove = np.random.choice(ones, int(sum_h - target), replace=False)
            h0[to_remove] = 0
        elif sum_h < target:
            zeros = np.where(h0 == 0)[0]
            to_add = np.random.choice(zeros, int(target - sum_h), replace=False)
            h0[to_add] = 1
            
        h_opt, val = local_search_binary(h0)
        
        if val > best_val:
            best_val = val
            best_h = h_opt
            
    # Continuous Refinement
    if time.time() < deadline - 5:
        h = best_h.copy()
        # Run a few subgradient steps
        lr = 0.01
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            C = compute_autocorr(h)
            C[0] = -np.inf
            k_min = np.argmin(C)
            val = C[k_min]
            
            if val > best_val:
                best_val = val
                best_h = h.copy()
                
            grad = np.roll(h, -k_min) + np.roll(h, k_min)
            h = h + lr * grad
            h = project(h)
            
    return best_h
