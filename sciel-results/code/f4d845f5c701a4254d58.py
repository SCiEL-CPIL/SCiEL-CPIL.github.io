# EVOLVE-BLOCK-START
import numpy as np

# Pre-compute bin indices for fast entropy calculation
_BINS = {}
def _init_bins():
    x = np.arange(10)
    y = np.arange(10)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Slope -1: X - Y. Range -9..9. Offset 9. Size 19.
    vals = (X - Y + 9).flatten()
    _BINS['D'] = (vals, 19)
    
    # Slope 0: X. Range 0..9. Size 10.
    vals = X.flatten()
    _BINS['H0'] = (vals, 10)
    
    # Slope 1: X + Y. Range 0..18. Size 19.
    vals = (X + Y).flatten()
    _BINS['H1'] = (vals, 19)
    
    # Slope 2: X + 2Y. Range 0..27. Size 28.
    vals = (X + 2 * Y).flatten()
    _BINS['H2'] = (vals, 28)

_init_bins()

def _calc_ratio(P_flat):
    """
    Calculate D/M for a flat array of 100 non-negative weights.
    """
    total = np.sum(P_flat)
    if total < 1e-9:
        return 0.0
        
    def _ent(counts):
        # counts are unnormalized weights
        nz = counts > 1e-12
        if not np.any(nz):
            return 0.0
        p = counts[nz] / total
        return -np.sum(p * np.log2(p))
        
    vals, size = _BINS['D']
    counts = np.bincount(vals, weights=P_flat, minlength=size)
    D = _ent(counts)
    
    vals, size = _BINS['H0']
    counts = np.bincount(vals, weights=P_flat, minlength=size)
    H0 = _ent(counts)
    
    vals, size = _BINS['H1']
    counts = np.bincount(vals, weights=P_flat, minlength=size)
    H1 = _ent(counts)
    
    vals, size = _BINS['H2']
    counts = np.bincount(vals, weights=P_flat, minlength=size)
    H2 = _ent(counts)
    
    M = max(H0, H1, H2)
    
    if M < 1e-9:
        return 0.0
        
    return D / M

def _local_search(P_init, steps=20000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
        
    P = P_init.copy()
    best_val = _calc_ratio(P)
    best_P = P.copy()
    current_val = best_val
    current_P = P.copy()
    
    # Adaptive step size
    step_size = 0.1
    
    for i in range(steps):
        # Decay step size
        step_size = max(0.001, 0.1 * (0.9995 ** i))
        
        idx1 = rng.integers(0, 100)
        idx2 = rng.integers(0, 100)
        if idx1 == idx2:
            continue
            
        delta = step_size * rng.uniform(0.1, 1.0)
        
        # Direction: move from idx1 to idx2
        if P[idx1] < delta:
            delta = P[idx1] * 0.5
            if delta < 1e-12:
                continue
                
        P_new = current_P.copy()
        P_new[idx1] -= delta
        P_new[idx2] += delta
        
        val_new = _calc_ratio(P_new)
        
        # Accept if better, or with small prob if worse (Simulated Annealing style)
        # Simple hill climbing: accept if better
        if val_new > current_val:
            current_val = val_new
            current_P = P_new
            if current_val > best_val:
                best_val = current_val
                best_P = current_P.copy()
        else:
            # Occasionally accept worse moves to escape local minima
            # Temperature decays
            temp = 0.01 * (0.999 ** i)
            if rng.random() < temp:
                current_val = val_new
                current_P = P_new
            # Else reject, keep current_P
        
    return best_P, best_val

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the arithmetic Kakeya entropy ratio using Local Search with Simulated Annealing.
    """
    best_P_global = None
    best_val_global = 0.0
    
    rng = np.random.default_rng(42)
    
    # Initializations
    P_uni = np.ones(100) / 100.0
    
    P_pat1 = np.zeros(100)
    for y in range(10):
        P_pat1[y, 0] = 1.0
        P_pat1[y, 9] = 1.0
    P_pat1 /= 20.0
    
    P_pat2 = np.zeros(100)
    for i in range(10):
        P_pat2[i, i] = 1.0
    P_pat2 /= 10.0
    
    P_pat3 = np.zeros(100)
    for i in range(10):
        P_pat3[i, 9-i] = 1.0
    P_pat3 /= 10.0
        
    P_pat4 = np.zeros(100)
    idx = rng.choice(100, size=20, replace=False)
    P_pat4[idx] = 1.0
    P_pat4 /= 20.0
    
    P_pat5 = np.zeros(100)
    idx = rng.choice(100, size=50, replace=False)
    P_pat5[idx] = 1.0
    P_pat5 /= 50.0

    initializations = [P_uni, P_pat1, P_pat2, P_pat3, P_pat4, P_pat5]
    
    for init in initializations:
        P_opt, val = _local_search(init, steps=10000, rng=rng)
        
        if val > best_val_global:
            best_val_global = val
            best_P_global = P_opt
            
    if best_P_global is None:
        return P_uni.reshape(10, 10)
        
    P_final = best_P_global.reshape(10, 10)
    P_final = np.clip(P_final, 0, None)
    s = np.sum(P_final)
    if s < 1e-9:
        return np.ones((10,10))/100.0
    P_final /= s
        
    return P_final
# EVOLVE-BLOCK-END
