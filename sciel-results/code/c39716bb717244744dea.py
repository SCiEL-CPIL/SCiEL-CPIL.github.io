import numpy as np
import time

def _eval_sup_norm(coeffs, n_samples=20000, refine=True):
    n = len(coeffs)
    thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    k = np.arange(n)
    powers = np.exp(1j * np.outer(k, thetas))
    P = coeffs @ powers
    max_val = np.max(np.abs(P))
    
    if refine and max_val > 0:
        idx = np.argmax(np.abs(P))
        theta_max = thetas[idx]
        delta = 2 * np.pi / n_samples
        a = theta_max - delta
        b = theta_max + delta
        
        def f(theta_val):
            z_val = np.exp(1j * theta_val)
            val = 0
            zk = 1
            for c in coeffs:
                val += c * zk
                zk *= z_val
            return np.abs(val) ** 2
        
        gr = (np.sqrt(5) - 1) / 2
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        f1 = f(x1)
        f2 = f(x2)
        
        for _ in range(50):
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - gr * (b - a)
                f1 = f(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + gr * (b - a)
                f2 = f(x2)
                
        refined_max = np.sqrt(max(f1, f2))
        if refined_max > max_val:
            max_val = refined_max
            
    return max_val

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 24
    best_coeffs = None
    best_score = float('inf')
    
    # 1. Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    golay = np.array(p[:n])
    
    initial_candidates = [golay]
    # 2. Random starts
    for _ in range(10):
        initial_candidates.append(np.random.choice([-1, 1], size=n))
        
    k = np.arange(n)
    
    # Phase 1: Coarse Local Search
    search_n_samples = 5000
    thetas = np.linspace(0, 2 * np.pi, search_n_samples, endpoint=False)
    powers = np.exp(1j * np.outer(k, thetas))
    
    T = 0.1
    cooling_rate = 0.999
    
    for candidate in initial_candidates:
        if time.time() > deadline:
            break
            
        current_coeffs = candidate.copy()
        current_score = np.max(np.abs(current_coeffs @ powers))
        
        if current_score < best_score:
            # Quick refine to update best_score
            refined = _eval_sup_norm(current_coeffs, n_samples=10000, refine=True)
            if refined < best_score:
                best_score = refined
                best_coeffs = current_coeffs.copy()
        
        iterations = 0
        while time.time() < deadline - 10: # Leave 10s for polish
            iterations += 1
            if iterations > 1000:
                break
                
            P = current_coeffs @ powers
            deltas = 2 * current_coeffs[:, None] * powers
            new_P = P[None, :] - deltas
            new_scores = np.max(np.abs(new_P), axis=1)
            min_idx = np.argmin(new_scores)
            min_score = new_scores[min_idx]
            
            if min_score < current_score:
                current_coeffs[min_idx] *= -1
                current_score = min_score
                if current_score < best_score:
                    if iterations % 20 == 0:
                        refined = _eval_sup_norm(current_coeffs, n_samples=10000, refine=True)
                        if refined < best_score:
                            best_score = refined
                            best_coeffs = current_coeffs.copy()
            else:
                delta = min_score - current_score
                if np.random.random() < np.exp(-delta / T):
                    current_coeffs[min_idx] *= -1
                    current_score = min_score
                else:
                    rand_idx = np.random.randint(n)
                    current_coeffs[rand_idx] *= -1
                    current_score = np.max(np.abs(current_coeffs @ powers))
                
                T *= cooling_rate
                if T < 0.001:
                    T = 0.001

    # Phase 2: Fine Polish on Best Candidate
    if best_coeffs is not None:
        dense_n_samples = 20000
        dense_thetas = np.linspace(0, 2 * np.pi, dense_n_samples, endpoint=False)
        dense_powers = np.exp(1j * np.outer(k, dense_thetas))
        
        current_coeffs = best_coeffs.copy()
        current_score = np.max(np.abs(current_coeffs @ dense_powers))
        
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            P = current_coeffs @ dense_powers
            deltas = 2 * current_coeffs[:, None] * dense_powers
            new_P = P[None, :] - deltas
            new_scores = np.max(np.abs(new_P), axis=1)
            min_idx = np.argmin(new_scores)
            min_score = new_scores[min_idx]
            
            if min_score < current_score:
                current_coeffs[min_idx] *= -1
                current_score = min_score
            else:
                # Small random kick
                if np.random.random() < 0.1:
                    rand_idx = np.random.randint(n)
                    current_coeffs[rand_idx] *= -1
                    current_score = np.max(np.abs(current_coeffs @ dense_powers))
                
        final_score = _eval_sup_norm(current_coeffs, n_samples=20000, refine=True)
        if final_score < best_score:
            best_score = final_score
            best_coeffs = current_coeffs.copy()

    if best_coeffs is None:
        return p[:n]
        
    return list(best_coeffs)
