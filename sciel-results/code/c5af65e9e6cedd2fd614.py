import numpy as np
import time

N = 400
START_TIME = time.time()
TIME_LIMIT = 145.0

def _score(f):
    s = np.sum(f)
    if abs(s) < 1e-9:
        return 1e10
    conv = np.convolve(f, f, mode='full')
    max_abs = np.max(np.abs(conv))
    return max_abs / (s * s)

def construct():
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    np.random.seed(2024)
    
    # Structural change: use multiple diverse initializations
    # and a two-phase optimization: coarse then fine
    
    initializations = [
        np.ones(N),
        np.ones(N) + np.random.randn(N) * 0.2,
        np.clip(np.random.randn(N), -0.5, 2.0),
        np.ones(N) * np.abs(np.random.randn(N)) + 0.5,
    ]
    
    for f_init in initializations:
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        
        f = f_init.copy()
        current_score = _score(f)
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
        
        # Phase 1: Coarse search with larger steps
        T = 0.05
        step_size = 0.2
        for i in range(2000):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            idx = np.random.randint(N)
            f[idx] += np.random.randn() * step_size
            new_score = _score(f)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_f = f.copy()
            else:
                delta = new_score - current_score
                if np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    f[idx] -= np.random.randn() * step_size  # Revert (approximate)
            
            T *= 0.999
            step_size *= 0.9995
        
        # Phase 2: Fine search with smaller steps
        T = 0.005
        step_size = 0.02
        for i in range(2000):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            idx = np.random.randint(N)
            f[idx] += np.random.randn() * step_size
            new_score = _score(f)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_f = f.copy()
            else:
                delta = new_score - current_score
                if np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    f[idx] -= np.random.randn() * step_size
        
    return best_f
