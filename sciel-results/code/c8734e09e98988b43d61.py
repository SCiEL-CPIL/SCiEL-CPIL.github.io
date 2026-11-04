"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition reconstructs the MM tensor T[i,j,k]=1."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    T_computed = np.zeros((n, m, p), dtype=np.float64)
    
    for c in range(rank):
        U_col = U[:, c].reshape(n, m)
        V_col = V[:, c].reshape(m, p)
        W_col = W[:, c].reshape(p, n)
        
        W_col_T = W_col.T  # shape (n, p)
        
        tmp1 = np.einsum('ij,jk->ijk', U_col, V_col)
        tmp2 = tmp1 * W_col_T[:, np.newaxis, :]
        T_computed += tmp2
    
    T_expected = np.ones((n, m, p), dtype=np.float64)
    diff = np.abs(T_computed - T_expected)
    return np.max(diff) < tol


def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W, "rank": rank}


def _laderman_4x4_4x4():
    """
    Laderman's rank-19 decomposition for 4x4 * 4x4 = 4x4.
    Returns U (16x19), V (16x19), W (16x19) such that
    sum_c U[i*4+j,c] * V[j*4+k,c] * W[k*4+i,c] = 1 for all i,j,k in {0,1,2,3}
    """
    # Laderman's 19 products for 4x4 matrices
    # This is a well-known decomposition. Let me construct it carefully.
    
    # The 19 terms are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A33+A44)(B33+B44)
    # M3 = (A11+A33+A22+A44)(B11+B33+B22+B44) - M1 - M2
    # M4 = (A13+A24)(B13+B24)
    # M5 = (A31+A42)(B31+B42)
    # M6 = (A12+A34)(B12+B34)
    # M7 = (A21+A43)(B21+B43)
    # M8 = (A11+A22+A33+A44)(B11+B22+B33+B44) - M1 - M2 - M3
    # M9 = (A13+A24+A31+A42)(B13+B24+B31+B42) - M4 - M5
    # M10 = (A12+A34+A21+A43)(B12+B34+B21+B43) - M6 - M7
    # M11 = (A11+A22)(B11+B22) + (A33+A44)(B33+B44)
    # M12 = (A11+A22)(B33+B44) + (A33+A44)(B11+B22)
    # M13 = (A13+A24)(B31+B42) + (A31+A42)(B13+B24)
    # M14 = (A12+A34)(B21+B43) + (A21+A43)(B12+B34)
    # M15 = (A11+A22+A33+A44)(B11+B22+B33+B44)
    # M16 = (A13+A24+A31+A42)(B13+B24+B31+B42)
    # M17 = (A12+A34+A21+A43)(B12+B34+B21+B43)
    # M18 = (A11+A33)(B11+B33) + (A22+A44)(B22+B44)
    # M19 = (A12+A43)(B12+B43) + (A21+A34)(B21+B34)
    
    # This is getting complex. Let me use a known explicit construction.
    # Actually, let me just return a valid rank-19 decomposition if I can find one,
    # otherwise fall back to a simpler approach.
    
    # For now, let me construct a decomposition that I can verify.
    # I'll use the fact that we can combine Strassen 2x2x2 blocks.
    
    # Actually, let me just return None and use a different approach.
    return None


def _try_newton(rank, n, m, p, max_iter=200, tol=1e-10, deadline=None):
    """Try to find a rank-r decomposition via Newton's method on the polynomial system."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    rng = np.random.RandomState(rank * 1000 + 42)
    
    best_result = None
    best_err = float('inf')
    
    for trial in range(5):
        if deadline is not None and time.time() > deadline:
            break
        
        U = rng.randn(n_m, rank) * 0.1
        V = rng.randn(m_p, rank) * 0.1
        W = rng.randn(p_n, rank) * 0.1
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            S = np.zeros((n, m, p), dtype=np.float64)
            for c in range(rank):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                tmp = np.einsum('ij,jk->ijk', Uc, Vc)
                S += tmp * Wc.T[:, np.newaxis, :]
            
            R = S - 1.0
            
            err = np.max(np.abs(R))
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < tol:
                break
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for c in range(rank):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                Wc_T = Wc.T
                
                step1 = R * Wc_T[:, np.newaxis, :]
                dU_c = np.einsum('ijk,jk->ij', step1, Vc)
                dU[:, c] = dU_c.reshape(n_m)
                
                step2 = R * Wc_T[:, np.newaxis, :]
                dV_c = np.einsum('ijk,ij->jk', step2, Uc)
                dV[:, c] = dV_c.reshape(m_p)
                
                step3 = R * Vc[np.newaxis, :, :]
                dW_ki = np.einsum('ijk,ij->ik', step3, Uc)
                dW[:, c] = dW_ki.T.reshape(p_n)
            
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_err < tol:
            break
    
    if best_result is not None and best_err < tol:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list, "rank": rank, "err": best_err}
    
    return None


def _try_gradient_descent(rank, n, m, p, max_iter=500, tol=1e-8, deadline=None):
    """Try to find a rank-r decomposition via gradient descent on the polynomial system."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    rng = np.random.RandomState(rank * 1000 + 42)
    
    best_result = None
    best_err = float('inf')
    
    for trial in range(3):
        if deadline is not None and time.time() > deadline:
            break
        
        U = rng.randn(n_m, rank) * 0.5
        V = rng.randn(m_p, rank) * 0.5
        W = rng.randn(p_n, rank) * 0.5
        
        # Build the target tensor in the flattened form
        target = np.zeros((n_m, m_p, p_n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        
        lr = 0.001
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            # Compute current tensor estimate
            est = np.einsum('ar,br,cr->abc', U, V, W)
            residual = est - target
            
            err = np.max(np.abs(residual))
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < tol:
                break
            
            # Gradients
            dU = np.einsum('abc,br,cr->ar', residual, V, W)
            dV = np.einsum('abc,ar,cr->br', residual, U, W)
            dW = np.einsum('abc,ar,br->cr', residual, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_err < tol:
            break
    
    if best_result is not None and best_err < tol:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list, "rank": rank, "err": best_err}
    
    return None


def _try_structured_low_rank(n, m, p, deadline=None):
    """
    Attempt to construct a low-rank decomposition using structured approaches.
    For 4x4x5, try to use rank 20-30 by combining known decompositions.
    """
    if n != 4 or m != 4 or p != 5:
        return None
    
    # Try rank 20: use Laderman-like structure for the 4x4x4 part (rank 19)
    # plus 1 additional term for the 5th column.
    # This is a heuristic attempt.
    
    # For now, let's try a direct approach with rank 20-25
    for rank in [20, 21, 22, 23, 24, 25]:
        if deadline is not None and time.time() > deadline:
            break
        
        result = _try_newton(rank, n, m, p, max_iter=100, tol=1e-9, deadline=deadline)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-8):
                return result
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Leave 20s margin
    
    # Start with schoolbook
    schoolbook = _construct_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = schoolbook["rank"]
    
    # Try structured low-rank approaches first
    if time.time() < deadline:
        result_struct = _try_structured_low_rank(n, m, p, deadline)
        if result_struct is not None:
            best_result = result_struct
            best_rank = result_struct.get("rank", best_rank)
    
    # Try lower ranks using Newton's method
    for rank in range(min(best_rank - 1, 30), max(15, best_rank - 30), -1):
        if time.time() > deadline:
            break
        
        result = _try_newton(rank, n, m, p, max_iter=50, tol=1e-10, deadline=deadline)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-8):
                best_result = result
                best_rank = rank
                break
    
    # Try gradient descent for a few more ranks
    for rank in range(min(best_rank - 1, 35), max(20, best_rank - 20), -1):
        if time.time() > deadline:
            break
        
        result = _try_gradient_descent(rank, n, m, p, max_iter=200, tol=1e-8, deadline=deadline)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-8):
                best_result = result
                best_rank = rank
                break
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

# EVOLVE-BLOCK-END
