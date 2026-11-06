import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 129
    n_fft = 512
    
    def compute_R(v):
        F = np.fft.fft(v, n_fft)
        C = np.fft.ifft(F * F).real
        Rs = np.zeros(129)
        for i, k in enumerate(range(128, 257)):
            if k % 2 == 0:
                half = k // 2
                if 0 <= half < N:
                    Rs[i] = (C[k] + v[half]) / 2.0
                else:
                    Rs[i] = C[k] / 2.0
            else:
                Rs[i] = C[k] / 2.0
        return Rs

    def scalar_cost(v, Rs):
        zeros = np.sum(Rs < 1)
        max_r = np.max(Rs)
        size = np.sum(v)
        # If zeros > 0, we want to fix them.
        # Cost = max_r + 10000 * zeros + 0.001 * size
        return max_r + 10000 * zeros + 0.001 * size

    best_v = None
    best_cost = float('inf')
    
    # Initial random sets
    for _ in range(20):
        if time.time() > deadline:
            break
            
        M = 20 + np.random.randint(20) # 20 to 40
        v = np.zeros(N, dtype=int)
        v[np.random.choice(N, M, replace=False)] = 1
        
        Rs = compute_R(v)
        cost = scalar_cost(v, Rs)
        
        if cost < best_cost:
            best_cost = cost
            best_v = v.copy()
            
        # SA
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        steps = 0
        
        while T > min_T:
            if time.time() > deadline:
                break
            steps += 1
            
            v_new = v.copy()
            zeros = np.sum(Rs < 1)
            
            # Bias moves
            if zeros > 0:
                # Must add
                candidates = np.where(v == 0)[0]
                if len(candidates) == 0: continue
                x = candidates[np.random.randint(len(candidates))]
                v_new[x] = 1
            else:
                # Try to reduce size or peak
                r = np.random.rand()
                if r < 0.3: # Remove
                    candidates = np.where(v == 1)[0]
                    if len(candidates) == 0: continue
                    x = candidates[np.random.randint(len(candidates))]
                    v_new[x] = 0
                elif r < 0.7: # Swap
                    candidates_in = np.where(v == 1)[0]
                    candidates_out = np.where(v == 0)[0]
                    if len(candidates_in) == 0 or len(candidates_out) == 0: continue
                    x_in = candidates_in[np.random.randint(len(candidates_in))]
                    x_out = candidates_out[np.random.randint(len(candidates_out))]
                    v_new[x_in] = 0
                    v_new[x_out] = 1
                else: # Add
                    candidates = np.where(v == 0)[0]
                    if len(candidates) == 0: continue
                    x = candidates[np.random.randint(len(candidates))]
                    v_new[x] = 1
            
            Rs_new = compute_R(v_new)
            cost_new = scalar_cost(v_new, Rs_new)
            
            delta = cost_new - cost
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                v = v_new
                Rs = Rs_new
                cost = cost_new
                
                if cost < best_cost:
                    best_cost = cost
                    best_v = v.copy()
                    
            T *= cooling
            
    if best_v is None:
        return list(range(64, 129))
        
    return list(np.where(best_v == 1)[0])
