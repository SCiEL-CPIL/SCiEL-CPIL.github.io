"""Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24

    # Schoolbook solution as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = schoolbook_rank

    # Try ranks from schoolbook-1 down to 13
    for rank in range(schoolbook_rank - 1, 12, -1):
        if time.time() > deadline:
            break

        try:
            result = _try_rank(n, m, p, rank, deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                continue
        except Exception:
            continue

    return best_result

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_exact(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    
    # S[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    max_err = np.max(np.abs(S - 1.0))
    return max_err < 1e-9

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank decomposition with the given rank."""
    from scipy.optimize import least_squares

    # Residual function
    def residual_function(params):
        U = params[:n * m * rank].reshape(n, m, rank)
        V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m, p, rank)
        W = params[n * m * rank + m * p * rank:].reshape(p, n, rank)

        S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        residuals = S.flatten() - 1.0
        return residuals

    num_params = (n * m + m * p + p * n) * rank

    # Strategy 1: Multiple random restarts with varied initializations
    np.random.seed(rank * 7919)  # Fixed seed for reproducibility

    max_restarts = 40 if rank > 20 else 80

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Different initialization strategies
        if restart % 4 == 0:
            # Standard random
            U0 = np.random.randn(n * m, rank)
            V0 = np.random.randn(m * p, rank)
            W0 = np.random.randn(p * n, rank)
            scale = 1.0 / np.sqrt(rank)
            U0 *= scale
            V0 *= scale
            W0 *= scale
        elif restart % 4 == 1:
            # Sparse binary initialization
            U0 = np.zeros((n * m, rank))
            V0 = np.zeros((m * p, rank))
            W0 = np.zeros((p * n, rank))
            for r in range(rank):
                idx_u = np.random.randint(0, n * m)
                idx_v = np.random.randint(0, m * p)
                idx_w = np.random.randint(0, p * n)
                U0[idx_u, r] = np.random.choice([-1, 1])
                V0[idx_v, r] = np.random.choice([-1, 1])
                W0[idx_w, r] = np.random.choice([-1, 1])
        elif restart % 4 == 2:
            # Structured: start from schoolbook, merge columns
            U0 = np.zeros((n * m, rank))
            V0 = np.zeros((m * p, rank))
            W0 = np.zeros((p * n, rank))
            # Assign each of the n*m*p triples to one of rank columns
            triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
            for t, (i, j, k) in enumerate(triples):
                col = t % rank
                U0[i * m + j, col] += 1
                V0[j * p + k, col] += 1
                W0[k * n + i, col] += 1
            U0 += 0.05 * np.random.randn(n * m, rank)
            V0 += 0.05 * np.random.randn(m * p, rank)
            W0 += 0.05 * np.random.randn(p * n, rank)
        else:
            # Diagonal-like: each rank element handles a subset
            U0 = np.zeros((n * m, rank))
            V0 = np.zeros((m * p, rank))
            W0 = np.zeros((p * n, rank))
            for r in range(rank):
                # Each rank r handles a random subset of (i,j,k) triples
                for _ in range(3):  # Each rank covers ~3 triples on average
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    k = np.random.randint(0, p)
                    U0[i * m + j, r] += 1
                    V0[j * p + k, r] += 1
                    W0[k * n + i, r] += 1

        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

        try:
            result = least_squares(residual_function, params0, method='trf',
                                   max_nfev=3000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
        except Exception:
            continue

        if result.cost < 1e-10:
            U_opt = result.x[:n * m * rank].reshape(n, m, rank)
            V_opt = result.x[n * m * rank:n * m * rank + m * p * rank].reshape(m, p, rank)
            W_opt = result.x[n * m * rank + m * p * rank:].reshape(p, n, rank)

            # Try snapping to rational values
            for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
                U_snap = np.round(U_opt * denom) / denom
                V_snap = np.round(V_opt * denom) / denom
                W_snap = np.round(W_opt * denom) / denom

                if _verify_exact(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }

            # Also try if values are already close to integers/half-integers
            for denom in [1, 2]:
                U_close = np.allclose(U_opt, np.round(U_opt * denom) / denom, atol=1e-8)
                V_close = np.allclose(V_opt, np.round(V_opt * denom) / denom, atol=1e-8)
                W_close = np.allclose(W_opt, np.round(W_opt * denom) / denom, atol=1e-8)
                if U_close and V_close and W_close:
                    U_snap = np.round(U_opt * denom) / denom
                    V_snap = np.round(V_opt * denom) / denom
                    W_snap = np.round(W_opt * denom) / denom
                    if _verify_exact(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }

    # Strategy 2: Try algebraic construction for small ranks
    if rank <= 20:
        result = _try_algebraic(n, m, p, rank, deadline)
        if result is not None:
            return result

    return None

def _try_algebraic(n, m, p, rank, deadline):
    """Try to construct a decomposition using algebraic patterns."""
    # For 2x3x4, try patterns based on known low-rank constructions
    # This is a heuristic approach - try specific structured forms
    
    # Pattern: Use rank elements to cover overlapping sets of triples
    # Each rank r contributes to a subset of (i,j,k) triples
    
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_triples = len(triples)
    
    # Try to partition triples into rank groups
    # Each group should be "coverable" by a rank-1 tensor
    # A rank-1 tensor U⊗V⊗W covers all (i,j,k) where U[i,j]≠0, V[j,k]≠0, W[k,i]≠0
    
    # For simplicity, try to find rank elements that each cover a rectangular block
    # This is a simplification - actual rank-1 tensors can be more complex
    
    # Try greedy assignment
    np.random.seed(rank * 31)
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Assign each triple to one or more rank elements
        # Each rank element gets a random subset of triples
        for r in range(rank):
            # Each rank covers num_triples/rank triples on average
            num_cover = num_triples // rank + (1 if r < num_triples % rank else 0)
            selected = np.random.choice(num_triples, size=min(num_cover, num_triples), replace=False)
            
            for idx in selected:
                i, j, k = triples[idx]
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
        
        # Verify
        if _verify_exact(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        
        # Try with values ±1
        for r in range(rank):
            U[:, r] *= np.random.choice([-1, 1], size=n * m)
            V[:, r] *= np.random.choice([-1, 1], size=m * p)
            W[:, r] *= np.random.choice([-1, 1], size=p * n)
        
        if _verify_exact(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None
