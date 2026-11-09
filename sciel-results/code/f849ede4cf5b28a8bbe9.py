# EVOLVE-BLOCK-START
import numpy as np

def _calc_score(P):
    """
    Calculate the score D/M for a given 10x10 probability matrix P.
    Returns 0 if invalid (M <= 1e-9).
    """
    if np.any(P < -1e-12) or np.any(np.isnan(P)) or np.any(np.isinf(P)):
        return 0.0
    
    P = np.clip(P, 0, None)
    s = np.sum(P)
    if s < 1e-9:
        return 0.0
    P = P / s
    
    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Precompute bin indices for the 4 slopes
    # r = -1: val = x - y. range -9 to 9. index = val + 9. size 19.
    bins_m1 = (X - Y + 9).astype(int).ravel()
    # r = 0: val = x. range 0 to 9. index = val. size 10.
    bins_0 = X.astype(int).ravel()
    # r = 1: val = x + y. range 0 to 18. index = val. size 19.
    bins_1 = (X + Y).astype(int).ravel()
    # r = 2: val = x + 2y. range 0 to 27. index = val. size 28.
    bins_2 = (X + 2 * Y).astype(int).ravel()
    
    P_flat = P.ravel()
    
    def get_entropy(bins, num_bins):
        bin_probs = np.bincount(bins, weights=P_flat, minlength=num_bins)
        bin_probs = bin_probs[bin_probs > 1e-12]
        if len(bin_probs) == 0:
            return 0.0
        return -np.sum(bin_probs * np.log2(bin_probs))
    
    H_m1 = get_entropy(bins_m1, 19)
    H_0 = get_entropy(bins_0, 10)
    H_1 = get_entropy(bins_1, 19)
    H_2 = get_entropy(bins_2, 28)
    
    D = H_m1
    M = max(H_0, H_1, H_2)
    
    if M < 1e-9:
        return 0.0
        
    return D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Uses Multi-Start Hill Climbing to maximize the arithmetic Kakeya entropy ratio.
    """
    n = 10
    best_score = -1.0
    best_P = np.ones((n, n)) / (n * n)
    
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones((n, n)) / (n * n))
    
    # 2. Diagonal
    P_diag = np.zeros((n, n))
    np.fill_diagonal(P_diag, 1.0/n)
    candidates.append(P_diag)
    
    # 3. Anti-diagonal
    P_anti = np.zeros((n, n))
    for i in range(n):
        P_anti[i, n-1-i] = 1.0/n
    candidates.append(P_anti)
    
    # 4. Cross (center rows/cols)
    P_cross = np.zeros((n, n))
    P_cross[4, :] = 1.0/(2*n)
    P_cross[5, :] = 1.0/(2*n)
    P_cross[:, 4] += 1.0/(2*n)
    P_cross[:, 5] += 1.0/(2*n)
    P_cross /= np.sum(P_cross)
    candidates.append(P_cross)
    
    # 5. Corners
    P_corner = np.zeros((n, n))
    P_corner[0,0] = 0.25
    P_corner[0,n-1] = 0.25
    P_corner[n-1,0] = 0.25
    P_corner[n-1,n-1] = 0.25
    candidates.append(P_corner)
    
    # 6. Random
    np.random.seed(42)
    P_rand = np.random.rand(n, n)
    P_rand /= np.sum(P_rand)
    candidates.append(P_rand)
    
    max_iterations = 2000
    step_size = 0.05
    no_improve_limit = 200
    
    for P_init in candidates:
        P = P_init.copy()
        current_score = _calc_score(P)
        
        if current_score > best_score:
            best_score = current_score
            best_P = P.copy()
            
        no_improve = 0
        
        for _ in range(max_iterations):
            if no_improve > no_improve_limit:
                break
                
            i1, j1 = np.random.randint(0, n, 2)
            i2, j2 = np.random.randint(0, n, 2)
            
            if i1 == i2 and j1 == j2:
                continue
                
            delta = step_size * P[i1, j1]
            
            if delta < 1e-6:
                continue
                
            P_new = P.copy()
            P_new[i1, j1] -= delta
            P_new[i2, j2] += delta
            
            if P_new[i1, j1] >= -1e-9:
                score_new = _calc_score(P_new)
                if score_new > current_score:
                    P = P_new
                    current_score = score_new
                    no_improve = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_P = P.copy()
                    continue
            
            delta2 = step_size * P[i2, j2]
            if delta2 < 1e-6:
                continue
                
            P_new2 = P.copy()
            P_new2[i2, j2] -= delta2
            P_new2[i1, j1] += delta2
            
            if P_new2[i2, j2] >= -1e-9:
                score_new2 = _calc_score(P_new2)
                if score_new2 > current_score:
                    P = P_new2
                    current_score = score_new2
                    no_improve = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_P = P.copy()
                    continue
            
            no_improve += 1
            
    return best_P
# EVOLVE-BLOCK-END
