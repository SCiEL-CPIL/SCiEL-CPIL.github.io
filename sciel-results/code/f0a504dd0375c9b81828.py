import time
import numpy as np
from scipy.optimize import minimize

N = 100
SCALE = 2.0 / N

def _compute_O(h):
    one_minus_h = 1.0 - h
    O_pos = np.array([np.dot(h[k:], one_minus_h[:N-k]) for k in range(N)])
    O_neg = np.array([np.dot(h[:N-k], one_minus_h[k:]) for k in range(N)])
    return np.concatenate([O_pos, O_neg])

def _score(h):
    O = _compute_O(h)
    return float(O.max())

def _project(z):
    lo = float(z.min()) - 1.0
    hi = float(z.max()) + 1.0
    target = N / 2.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        s = np.clip(z - mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(z - t, 0.0, 1.0)

def _polish(h_init, deadline):
    """
    Polish the binary solution using gradient descent on the soft-max objective.
    """
    h = _project(h_init.copy())
    best_h = h.copy()
    best_val = _score(h)
    
    tau = 50.0
    eta = 0.05
    
    for it in range(1000):
        if time.time() > deadline - 2.0:
            break
        
        O = _compute_O(h)
        Omax = float(O.max())
        z = tau * (O - Omax)
        ez = np.exp(z)
        w = ez / ez.sum()
        w_pos, w_neg = w[:N], w[N:]
        
        # Gradient computation (approximate)
        # We can use the gradient from the seed
        K = np.arange(N)[:, None]
        I = np.arange(N)[None, :]
        
        ge = (I >= K)
        lt = (I < (N - K))
        
        idx_p = np.clip(I - K, 0, N - 1)
        idx_m = np.clip(I + K, 0, N - 1)
        
        Gpos = SCALE * (ge.astype(float) - ge * h[idx_p] - lt * h[idx_m])
        Gneg = SCALE * (lt.astype(float) - lt * h[idx_m] - ge * h[idx_p])
        
        grad = w_pos @ Gpos + w_neg @ Gneg
        
        h_new = _project(h - eta * grad)
        val_new = _score(h_new)
        
        if val_new < best_val:
            best_h = h_new.copy()
            best_val = val_new
            h = h_new
            eta = min(eta * 1.05, 0.2)
        else:
            h = best_h.copy()
            eta *= 0.9
            if eta < 1e-5:
                break
        
        if it % 200 == 0 and it > 0:
            tau *= 1.5
            
    return best_h, best_val

def construct():
    start = time.time()
    deadline = start + 140.0
    
    best_h = None
    best_score = float('inf')
    
    # 1. Combinatorial Search (Simulated Annealing)
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
            
        h = np.zeros(N)
        indices = np.arange(N)
        np.random.shuffle(indices)
        h[indices[:N//2]] = 1.0
        
        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
        T = 1.0
        T_min = 0.01
        cooling = 0.99
        
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        
        steps = 3000
        for s in range(steps):
            if time.time() > deadline - 30.0:
                break
                
            a = ones[np.random.randint(len(ones))]
            b = zeros[np.random.randint(len(zeros))]
            
            h[a] = 0.0
            h[b] = 1.0
            
            new_score = _score(h)
            
            if new_score < current_score:
                current_score = new_score
                ones = ones[ones != a]
                zeros = zeros[zeros != b]
                ones = np.append(ones, b)
                zeros = np.append(zeros, a)
                
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                delta = new_score - current_score
                if np.random.rand() < np.exp(-delta / T):
                    current_score = new_score
                    ones = ones[ones != a]
                    zeros = zeros[zeros != b]
                    ones = np.append(ones, b)
                    zeros = np.append(zeros, a)
                else:
                    h[a] = 1.0
                    h[b] = 0.0
            
            T = max(T * cooling, T_min)

    # 2. Continuous Polish
    if best_h is not None and time.time() < deadline - 5.0:
        h_polished, val_polished = _polish(best_h, deadline)
        if val_polished < best_score:
            best_h = h_polished
            best_score = val_polished

    # Ensure sum is exactly N/2
    best_h = _project(best_h)
    for _ in range(10):
        d = N / 2.0 - best_h.sum()
        if abs(d) < 1e-12:
            break
        best_h = np.clip(best_h + d / N, 0.0, 1.0)
        best_h = _project(best_h)
        
    return best_h
