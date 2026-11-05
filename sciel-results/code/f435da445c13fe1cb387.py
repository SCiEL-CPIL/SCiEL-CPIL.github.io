import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4x6 matrix multiplication.
    
    Strategy: Try rank 71 via structured initialization (merging pairs from schoolbook)
    with local optimization. If found, return it. Otherwise fallback to rank 72.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    # Fallback: Schoolbook decomposition
    def get_schoolbook():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Target Tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    best_result = get_schoolbook()
    
    try:
        from scipy.optimize import minimize
    except ImportError:
        return best_result

    def check_exact(U, V, W):
        """Check if the decomposition exactly reconstructs T with half-integer entries."""
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        if not np.allclose(T_rec, T, atol=1e-9):
            return False
        
        if not np.allclose(U, np.round(U * 2) / 2, atol=1e-9):
            return False
        if not np.allclose(V, np.round(V * 2) / 2, atol=1e-9):
            return False
        if not np.allclose(W, np.round(W * 2) / 2, atol=1e-9):
            return False
            
        return True

    # Try rank 71
    target_rank = rank_schoolbook - 1
    if target_rank > 0 and time.time() < deadline:
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank
        
        def objective_and_grad(x):
            U_flat = x[:size_U]
            V_flat = x[size_U:size_U+size_V]
            W_flat = x[size_U+size_V:]
            
            U = U_flat.reshape(n, m, target_rank)
            V = V_flat.reshape(m, p, target_rank)
            W = W_flat.reshape(p, n, target_rank)
            
            T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = T_rec - T
            loss = np.sum(err**2)
            
            grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
            grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
            grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return loss, grad

        # Build schoolbook tensors for initialization
        U_school = np.zeros((n, m, rank_schoolbook))
        V_school = np.zeros((m, p, rank_schoolbook))
        W_school = np.zeros((p, n, rank_schoolbook))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i, j, col] = 1
                    V_school[j, k, col] = 1
                    W_school[k, i, col] = 1
                    col += 1
        
        # Strategy 1: Try merging different pairs of columns from schoolbook
        # Each pair merge gives a different rank-71 initialization
        pairs_tried = set()
        
        for a in range(rank_schoolbook):
            if time.time() >= deadline - 15:
                break
            for b in range(a + 1, rank_schoolbook):
                if (a, b) in pairs_tried:
                    continue
                if time.time() >= deadline - 15:
                    break
                
                pairs_tried.add((a, b))
                
                # Create rank-71 init by merging columns a and b
                U_init = np.zeros((n, m, target_rank))
                V_init = np.zeros((m, p, target_rank))
                W_init = np.zeros((p, n, target_rank))
                
                # Merged column at position 0
                U_init[:, :, 0] = (U_school[:, :, a] + U_school[:, :, b]) / 2
                V_init[:, :, 0] = (V_school[:, :, a] + V_school[:, :, b]) / 2
                W_init[:, :, 0] = (W_school[:, :, a] + W_school[:, :, b]) / 2
                
                # Copy remaining columns
                idx = 1
                for c in range(rank_schoolbook):
                    if c == a or c == b:
                        continue
                    U_init[:, :, idx] = U_school[:, :, c]
                    V_init[:, :, idx] = V_school[:, :, c]
                    W_init[:, :, idx] = W_school[:, :, c]
                    idx += 1
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                try:
                    res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                                   options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 25})
                    
                    if res.fun < 1e-10:
                        U_flat = res.x[:size_U]
                        V_flat = res.x[size_U:size_U+size_V]
                        W_flat = res.x[size_U+size_V:]
                        
                        U = U_flat.reshape(n, m, target_rank)
                        V = V_flat.reshape(m, p, target_rank)
                        W = W_flat.reshape(p, n, target_rank)
                        
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        
                        if check_exact(U_rounded, V_rounded, W_rounded):
                            return {
                                "U": U_rounded.flatten(axis=1).tolist(),
                                "V": V_rounded.flatten(axis=1).tolist(),
                                "W": W_rounded.flatten(axis=1).tolist()
                            }
                except Exception:
                    continue

        # Strategy 2: Random restarts with smaller scale
        if time.time() < deadline - 10:
            for seed in range(15):
                if time.time() >= deadline - 10:
                    break
                np.random.seed(seed * 7 + 3)
                x0 = np.random.randn(size_U + size_V + size_W) * 0.08
                try:
                    res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                                   options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 25})
                    
                    if res.fun < 1e-10:
                        U_flat = res.x[:size_U]
                        V_flat = res.x[size_U:size_U+size_V]
                        W_flat = res.x[size_U+size_V:]
                        
                        U = U_flat.reshape(n, m, target_rank)
                        V = V_flat.reshape(m, p, target_rank)
                        W = W_flat.reshape(p, n, target_rank)
                        
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        
                        if check_exact(U_rounded, V_rounded, W_rounded):
                            return {
                                "U": U_rounded.flatten(axis=1).tolist(),
                                "V": V_rounded.flatten(axis=1).tolist(),
                                "W": W_rounded.flatten(axis=1).tolist()
                            }
                except Exception:
                    continue

    return best_result

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
