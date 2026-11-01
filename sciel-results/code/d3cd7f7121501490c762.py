"""Matrix multiplication 3x3x5 tensor decomposition - rank minimization search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]


def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[u,v,w]."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T


def _verify_decomposition(U, V, W, T, tol=1e-6):
    """Verify that U @ V @ W reconstruction matches target tensor T."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Reconstruct: T_reconstructed[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    # Efficient: for each r, outer product of U[:,r], V[:,r], W[:,r]
    rank = U.shape[1]
    T_rec = np.zeros(T.shape)
    for r in range(rank):
        T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_rec, T, atol=tol)


def _round_to_halfint(arr, tol=1e-3):
    """Round array to nearest half-integers, return (rounded, max_error)."""
    rounded = np.round(arr * 2.0) / 2.0
    max_err = np.max(np.abs(arr - rounded))
    return rounded, max_err


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    T = _build_target_tensor(n, m, p)

    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    best_result, best_rank = schoolbook()

    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But practically, for 3x3x5, known results suggest rank around 30-40 might be achievable
    
    # Use Nelder-Mead optimization to find low-rank decompositions
    def loss_function(params, rank):
        """Loss: sum of squared differences between reconstructed and target tensor."""
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        T_rec = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        diff = T_rec - T
        return np.sum(diff * diff)

    # Try ranks from just below schoolbook down
    for target_rank in range(n * m * p - 1, max(14, n * m * p - 30), -1):
        if time.time() > deadline - 20:
            break

        # Multiple random restarts
        found = False
        n_restarts = 3 if target_rank > 35 else 5
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break

            np.random.seed(42 + restart * 1000 + target_rank)
            # Initialize with small random values
            total_params = (n * m + m * p + p * n) * target_rank
            params0 = np.random.randn(total_params) * 0.5

            # Use Nelder-Mead
            try:
                from scipy.optimize import minimize
                result = minimize(
                    loss_function,
                    params0,
                    args=(target_rank,),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if result.success or result.fun < 1e-4:
                    # Extract factors
                    U = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Round to half-integers
                    U_rounded, err_u = _round_to_halfint(U)
                    V_rounded, err_v = _round_to_halfint(V)
                    W_rounded, err_w = _round_to_halfint(W)
                    
                    max_err = max(err_u, err_v, err_w)
                    
                    if max_err < 1e-3:
                        # Verify the rounded solution
                        if _verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-6):
                            best_result = {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }
                            best_rank = target_rank
                            found = True
                            break
                    elif result.fun < 1e-6:
                        # Try tighter rounding
                        U_rounded = np.round(U * 2.0) / 2.0
                        V_rounded = np.round(V * 2.0) / 2.0
                        W_rounded = np.round(W * 2.0) / 2.0
                        if _verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-4):
                            best_result = {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }
                            best_rank = target_rank
                            found = True
                            break
            except Exception:
                pass

        if found:
            # Found a valid decomposition at this rank, try to go lower
            continue
        else:
            # If we can't find at this rank, likely can't at higher ranks either
            # But try a couple more ranks down
            break

    return best_result
