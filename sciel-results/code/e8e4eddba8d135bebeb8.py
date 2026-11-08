# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition is exact. Returns max error."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _to_half_int(x, tol=1e-6):
    """Round to nearest half-integer if close."""
    rounded = np.round(x * 2) / 2.0
    if np.max(np.abs(x - rounded)) < tol:
        return rounded
    return x

def _find_3x3x2_rank(target_rank, max_time=30.0):
    """
    Find a rank-target_rank decomposition for 3x3x2 matrix multiplication.
    n=3, m=3, p=2. U:(9, rank), V:(6, rank), W:(6, rank).
    Uses multiple initialization strategies and a polish step.
    """
    n, m, p = 3, 3, 2
    start_time = time.time()
    
    def _make_init(strategy, seed):
        np.random.seed(seed)
        if strategy == 'random':
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
        elif strategy == 'perturbed_schoolbook':
            # Start from schoolbook (rank 18), remove (18-target_rank) columns, add noise
            U_full = np.zeros((n * m, 18))
            V_full = np.zeros((m * p, 18))
            W_full = np.zeros((p * n, 18))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_full[i * m + j, col] = 1.0
                        V_full[j * p + k, col] = 1.0
                        W_full[k * n + i, col] = 1.0
                        col += 1
            if target_rank < 18:
                cols_to_remove = np.random.choice(18, 18 - target_rank, replace=False)
                cols_to_keep = [c for c in range(18) if c not in cols_to_remove]
                U = U_full[:, cols_to_keep]
                V = V_full[:, cols_to_keep]
                W = W_full[:, cols_to_keep]
            else:
                U = U_full[:, :target_rank].copy()
                V = V_full[:, :target_rank].copy()
                W = W_full[:, :target_rank].copy()
            # Add small perturbation to escape local minima
            U += np.random.randn(n * m, target_rank) * 0.05
            V += np.random.randn(m * p, target_rank) * 0.05
            W += np.random.randn(p * n, target_rank) * 0.05
        elif strategy == 'structured':
            # Each column contributes to a specific (i,k) output
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            for col in range(target_rank):
                i_idx = col % n
                k_idx = (col // n) % p
                for j in range(m):
                    U[i_idx * m + j, col] = 1.0
                    V[j * p + k_idx, col] = 1.0
                    W[k_idx * n + i_idx, col] = 1.0
            U += np.random.randn(n * m, target_rank) * 0.1
            V += np.random.randn(m * p, target_rank) * 0.1
            W += np.random.randn(p * n, target_rank) * 0.1
        return U, V, W
    
    def _objective(x):
        Uf = x[:n*m*target_rank].reshape(n*m, target_rank)
        Vf = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        Wf = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Uf[i*m+j, :], Vf[j*p+k, :] * Wf[k*n+i, :])
                    err += (val - 1.0)**2
        return err
    
    def _verify(Uf, Vf, Wf, tol=1e-10):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Uf[i*m+j, :], Vf[j*p+k, :] * Wf[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err < tol
    
    # Try multiple strategies and seeds
    strategies = ['perturbed_schoolbook', 'structured', 'random']
    seeds_per_strategy = 30
    
    for strategy in strategies:
        for seed in range(seeds_per_strategy):
            if time.time() - start_time > max_time:
                return None, None, None
            
            U, V, W = _make_init(strategy, seed)
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = minimize(_objective, x0, method='L-BFGS-B',
                                 options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
                
                if result.fun < 1e-10:
                    Uf = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    Vf = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    Wf = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                    
                    if _verify(Uf, Vf, Wf, tol=1e-10):
                        return Uf, Vf, Wf
            except Exception:
                continue
    
    return None, None, None

def _construct_3x3x6_from_3x3x2(U2, V2, W2, rank2):
    """
    Construct 3x3x6 decomposition from three 3x3x2 decompositions.
    Total rank = 3 * rank2.
    """
    n, m, p = 3, 3, 6
    rank = 3 * rank2
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for g in range(3):
        for r in range(rank2):
            col = g * rank2 + r
            U[:, col] = U2[:, r]
            for j in range(m):
                for local_k in range(2):
                    global_k = 2 * g + local_k
                    V[j * p + global_k, col] = V2[j * 2 + local_k, r]
            for local_k in range(2):
                for i in range(n):
                    global_k = 2 * g + local_k
                    W[global_k * n + i, col] = W2[local_k * n + i, r]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0

    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try rank 17 for 3x3x2 -> Total rank 51
    if time.time() - start_time < deadline - 15:
        U2, V2, W2 = _find_3x3x2_rank(17, max_time=25.0)
        if U2 is not None:
            U51, V51, W51 = _construct_3x3x6_from_3x3x2(U2, V2, W2, 17)
            err51 = _check_decomposition(U51, V51, W51, n, m, p)
            if err51 < 1e-8:
                best_rank = 51
                best_U, best_V, best_W = U51, V51, W51
                U_rounded = _to_half_int(best_U)
                V_rounded = _to_half_int(best_V)
                W_rounded = _to_half_int(best_W)
                if _check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p) < 1e-6:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                U_out = [[float(x) for x in row] for row in best_U]
                V_out = [[float(x) for x in row] for row in best_V]
                W_out = [[float(x) for x in row] for row in best_W]
                return {"U": U_out, "V": V_out, "W": W_out}

    # Try rank 16 for 3x3x2 -> Total rank 48
    if time.time() - start_time < deadline - 15:
        U2, V2, W2 = _find_3x3x2_rank(16, max_time=30.0)
        if U2 is not None:
            U48, V48, W48 = _construct_3x3x6_from_3x3x2(U2, V2, W2, 16)
            err48 = _check_decomposition(U48, V48, W48, n, m, p)
            if err48 < 1e-8:
                best_rank = 48
                best_U, best_V, best_W = U48, V48, W48
                U_rounded = _to_half_int(best_U)
                V_rounded = _to_half_int(best_V)
                W_rounded = _to_half_int(best_W)
                if _check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p) < 1e-6:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                U_out = [[float(x) for x in row] for row in best_U]
                V_out = [[float(x) for x in row] for row in best_V]
                W_out = [[float(x) for x in row] for row in best_W]
                return {"U": U_out, "V": V_out, "W": W_out}

    # Try rank 15 for 3x3x2 -> Total rank 45
    if time.time() - start_time < deadline - 10:
        U2, V2, W2 = _find_3x3x2_rank(15, max_time=25.0)
        if U2 is not None:
            U45, V45, W45 = _construct_3x3x6_from_3x3x2(U2, V2, W2, 15)
            err45 = _check_decomposition(U45, V45, W45, n, m, p)
            if err45 < 1e-8:
                best_rank = 45
                best_U, best_V, best_W = U45, V45, W45
                U_rounded = _to_half_int(best_U)
                V_rounded = _to_half_int(best_V)
                W_rounded = _to_half_int(best_W)
                if _check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p) < 1e-6:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                U_out = [[float(x) for x in row] for row in best_U]
                V_out = [[float(x) for x in row] for row in best_V]
                W_out = [[float(x) for x in row] for row in best_W]
                return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback to schoolbook
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
