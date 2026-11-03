import numpy as np
import time

N = 50

def _sidon_set(n_max, target_size):
    """Find a Sidon (B_2) set of given size in [0, n_max)."""
    # Greedy construction
    S = []
    used_diffs = set()
    for x in range(n_max):
        diffs = set()
        valid = True
        for s in S:
            d = abs(x - s)
            if d in used_diffs:
                valid = False
                break
            diffs.add(d)
        if valid:
            S.append(x)
            used_diffs.update(diffs)
            if len(S) == target_size:
                break
    return S

def _autoconv_sup(h):
    """Compute supremum norm of circular autoconvolution."""
    h = np.array(h, dtype=float)
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    return np.max(np.abs(conv))

def _score(h):
    """Compute the ratio: L2^2 / (L1 * sup_autoconv)."""
    h = np.array(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(np.abs(conv))
    if sup_conv == 0:
        return 0.0
    return (l2_sq ** 2) / (l1 * sup_conv)

def construct():
    deadline = time.time() + 120  # Safety margin before 150s kill
    
    # Start with Sidon set unit spikes
    sidon = _sidon_set(N, 8)
    best_h = np.zeros(N)
    for s in sidon:
        best_h[s] = 1.0
    best_score = _score(best_h)
    
    # Try different Sidon set sizes
    for size in range(5, 10):
        if time.time() > deadline - 30:
            break
        S = _sidon_set(N, size)
        h = np.zeros(N)
        for s in S:
            h[s] = 1.0
        sc = _score(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
    
    # Local optimization: try adjusting heights on the Sidon set
    if time.time() < deadline - 20:
        # Gradient-free local search on the support
        support = np.where(best_h > 0)[0]
        current = best_h.copy()
        current_score = best_score
        
        for iteration in range(500):
            if time.time() > deadline - 15:
                break
            
            improved = False
            # Try scaling each element
            for idx in support:
                for factor in [0.5, 0.75, 1.25, 1.5, 2.0]:
                    trial = current.copy()
                    trial[idx] *= factor
                    sc = _score(trial)
                    if sc > current_score + 1e-10:
                        current = trial
                        current_score = sc
                        best_h = trial.copy()
                        best_score = sc
                        improved = True
                        break
            
            # Try adding a new spike at a random position
            if not improved and iteration % 10 == 0:
                trial = current.copy()
                pos = np.random.randint(N)
                if trial[pos] == 0:
                    trial[pos] = 1.0
                    sc = _score(trial)
                    if sc > current_score + 1e-10:
                        current = trial
                        current_score = sc
                        best_h = trial.copy()
                        best_score = sc
                        support = np.where(current > 0)[0]
        
        # If no improvement, try fresh Sidon sets with randomization
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            # Random Sidon-like set
            S = list(np.random.choice(N, size=8, replace=False))
            S.sort()
            # Check Sidon property
            diffs = set()
            valid = True
            for i in range(len(S)):
                for j in range(i+1, len(S)):
                    d = S[j] - S[i]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            
            if valid:
                h = np.zeros(N)
                for s in S:
                    h[s] = 1.0
                sc = _score(h)
                if sc > best_score:
                    best_score = sc
                    best_h = h.copy()
    
    return list(best_h)
