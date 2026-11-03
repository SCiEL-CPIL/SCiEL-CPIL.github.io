"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that U @ diag @ V @ diag @ W reconstructs the MM tensor."""
    # MM tensor: T[i,j,k] = 1 (for all i,j,k), because (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The rank-r decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # We can check by building the full tensor from the factors.
    # For efficiency, we'll check a subset or all entries.
    
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    # Build tensor T of shape (n, m, p)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # We can vectorize: for each (i,j,k), compute the dot product
    # of U[i*m+j, :], V[j*p+k, :], W[k*n+i, :] element-wise then sum.
    
    # More efficient: 
    # For each c, the contribution to T[i,j,k] is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Sum over c.
    
    # Let's do it with broadcasting
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    # 
    # Let's compute: for each (i,j), sum over c of U_reshaped[i,j,c] * V_reshaped[j,k,c] gives a vector over k
    # Actually let's just do it directly for correctness
    # 
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # Let's compute for all (i,j,k)
    # 
    # One approach: for each c, compute the outer product contribution
    # But that's O(rank * n*m*p). Let's do it.
    
    T_computed = np.zeros((n, m, p), dtype=np.float64)
    
    for c in range(rank):
        # Contribution from term c:
        # U_col = U[:, c] of shape (n*m,)
        # V_col = V[:, c] of shape (m*p,)
        # W_col = W[:, c] of shape (p*n,)
        
        U_col = U[:, c].reshape(n, m)
        V_col = V[:, c].reshape(m, p)
        W_col = W[:, c].reshape(p, n)
        
        # For each (i,j,k): U_col[i,j] * V_col[j,k] * W_col[k,i]
        # 
        # Let's compute: 
        # temp[i,j,k] = U_col[i,j] * V_col[j,k] * W_col[k,i]
        # 
        # We can do: 
        # step1[i,j,k] = U_col[i,j] * V_col[j,k]  (broadcast U_col[i,j] over k, V_col[j,k] over i)
        # step2[i,j,k] = step1[i,j,k] * W_col[k,i]
        
        # U_col is (n, m), V_col is (m, p)
        # step1: for each i, j, k: U_col[i,j] * V_col[j,k]
        # This is like: for each i, U_col[i,:] (shape m) times V_col (shape m, p) -> (m, p) but we want (n, m, p)
        # Actually: step1[i,j,k] = U_col[i,j] * V_col[j,k]
        # We can compute as: 
        # tmp1 = np.einsum('ij,jk->ijk', U_col, V_col)  # (n, m, p)
        # tmp2 = tmp1 * W_col.T  # W_col is (p, n), so W_col.T is (n, p), but we need W_col[k,i]
        # W_col[k,i] = W_col.T[i,k], so:
        # tmp2[i,j,k] = tmp1[i,j,k] * W_col.T[i,k]
        # But W_col.T has shape (n, p), and we're multiplying with tmp1 of shape (n, m, p)
        # We need to broadcast W_col.T over j: W_col.T[i,k] -> (n, 1, p)
        
        W_col_T = W_col.T  # shape (n, p), W_col_T[i,k] = W_col[k,i]
        
        tmp1 = np.einsum('ij,jk->ijk', U_col, V_col)  # (n, m, p)
        tmp2 = tmp1 * W_col_T[:, np.newaxis, :]  # broadcast W_col_T over j
        T_computed += tmp2
    
    # The MM tensor T[i,j,k] = 1 for all i,j,k
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


def _try_newton(rank, n, m, p, max_iter=200, tol=1e-10, deadline=None):
    """Try to find a rank-r decomposition via Newton's method on the polynomial system."""
    # Variables: U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Total variables: (n*m + m*p + p*n) * rank
    
    n_m = n * m
    m_p = m * p
    p_n = p * n
    total_vars = (n_m + m_p + p_n) * rank
    
    # Initialize with small random values
    rng = np.random.RandomState(rank * 1000 + 42)
    
    best_result = None
    best_err = float('inf')
    
    for trial in range(5):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random initialization
        U = rng.randn(n_m, rank) * 0.1
        V = rng.randn(m_p, rank) * 0.1
        W = rng.randn(p_n, rank) * 0.1
        
        # The equations: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Number of equations: n*m*p
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            # Compute residual: T_computed - T_expected
            # T_computed[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            
            U_reshaped = U.reshape(n, m, rank)
            V_reshaped = V.reshape(m, p, rank)
            W_reshaped = W.reshape(p, n, rank)
            
            # Compute residual tensor
            # R[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1
            # 
            # Let's compute S[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
            # 
            # Using einsum:
            # First: A[i,j,k] = sum_c U[i,j,c]*V[j,k,c] -> but V[j,k,c] depends on j,k
            # Actually: for fixed j, V[j,:,c] is over k
            # Let's think differently.
            #
            # S[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            #
            # We can compute this as:
            # For each c: term_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # S = sum_c term_c
            
            S = np.zeros((n, m, p), dtype=np.float64)
            for c in range(rank):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                # term[i,j,k] = Uc[i,j] * Vc[j,k] * Wc[k,i]
                tmp = np.einsum('ij,jk->ijk', Uc, Vc)
                S += tmp * Wc.T[:, np.newaxis, :]
            
            R = S - 1.0  # residual
            
            err = np.max(np.abs(R))
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < tol:
                break
            
            # Compute Jacobian and Newton step
            # This is complex. Let's use a simplified approach:
            # The system is nonlinear. Let's use a damped Newton or just 
            # try to improve using gradient descent on ||R||^2.
            #
            # Actually, for a cleaner approach, let's use the fact that
            # we can compute gradients.
            #
            # Loss = 0.5 * sum_{i,j,k} R[i,j,k]^2
            # dL/dU[i*m+j, c] = sum_k R[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
            # dL/dV[j*p+k, c] = sum_i R[i,j,k] * U[i*m+j,c] * W[k*n+i,c]
            # dL/dW[k*n+i, c] = sum_j R[i,j,k] * U[i*m+j,c] * V[j*p+k,c]
            
            # Compute gradients
            U_reshaped = U.reshape(n, m, rank)
            V_reshaped = V.reshape(m, p, rank)
            W_reshaped = W.reshape(p, n, rank)
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for c in range(rank):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                Wc_T = Wc.T  # (n, p), Wc_T[i,k] = Wc[k,i]
                
                # dU[i*m+j, c] = sum_k R[i,j,k] * Vc[j,k] * Wc_T[i,k]
                # Let's compute: for each (i,j), sum_k R[i,j,k] * Vc[j,k] * Wc_T[i,k]
                # 
                # temp1[j,k] = Vc[j,k] * Wc_T[i,k] -- but depends on i
                # Let's do:
                # grad_U[i,j] = sum_k R[i,j,k] * Vc[j,k] * Wc_T[i,k]
                #
                # We can compute: 
                # step1[i,j,k] = R[i,j,k] * Wc_T[i,k]  (broadcast Wc_T over j)
                # grad_U[i,j] = sum_k step1[i,j,k] * Vc[j,k]
                
                step1 = R * Wc_T[:, np.newaxis, :]  # (n, m, p)
                dU_c = np.einsum('ijk,jk->ij', step1, Vc)  # (n, m)
                dU[:, c] = dU_c.reshape(n_m)
                
                # dV[j*p+k, c] = sum_i R[i,j,k] * Uc[i,j] * Wc_T[i,k]
                # step2[i,j,k] = R[i,j,k] * Wc_T[i,k]
                # dV[j,k] = sum_i step2[i,j,k] * Uc[i,j]
                
                step2 = R * Wc_T[:, np.newaxis, :]  # (n, m, p)
                dV_c = np.einsum('ijk,ij->jk', step2, Uc)  # (m, p)
                dV[:, c] = dV_c.reshape(m_p)
                
                # dW[k*n+i, c] = sum_j R[i,j,k] * Uc[i,j] * Vc[j,k]
                # dW[k,i] = sum_j R[i,j,k] * Uc[i,j] * Vc[j,k]
                # 
                # Let's compute: for each (i,k), sum_j R[i,j,k] * Uc[i,j] * Vc[j,k]
                
                # step3[i,j,k] = R[i,j,k] * Vc[j,k]
                # dW[k,i] = sum_j step3[i,j,k] * Uc[i,j]
                
                step3 = R * Vc[np.newaxis, :, :]  # (n, m, p)
                dW_ki = np.einsum('ijk,ij->ik', step3, Uc)  # (n, p)
                dW[:, c] = dW_ki.T.reshape(p_n)  # W indexed by k*n+i, so dW[k*n+i] = dW_ki[i,k]
            
            # Gradient descent step
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # After iterations, check if we got a good solution
        if best_err < tol:
            break
    
    if best_result is not None and best_err < tol:
        U, V, W = best_result
        # Convert to lists with integer/half-integer entries if possible
        # For now, just return as floats
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list, "rank": rank, "err": best_err}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Leave 20s margin
    
    # Start with schoolbook
    schoolbook = _construct_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = schoolbook["rank"]
    
    # Try lower ranks using Newton's method
    # The minimal rank for 4x4x5 MM is unknown, but we know it's <= 80
    # Let's try ranks from 79 down to some lower bound
    # 
    # For matrix multiplication, the tensor rank is at least max(n*m, m*p, p*n) / something
    # A lower bound is max(n*m, m*p, p*n) but that's not tight.
    # Actually, a simple lower bound is: rank >= max(n*m, m*p, p*n) is not true.
    # 
    # Let's just try a range of ranks.
    
    # Try ranks: we'll try from 79 down to 60
    for rank in range(79, 59, -1):
        if time.time() > deadline:
            break
        
        result = _try_newton(rank, n, m, p, max_iter=50, tol=1e-10, deadline=deadline)
        if result is not None:
            # Verify
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-8):
                best_result = result
                best_rank = rank
                break  # Found a valid lower rank, stop
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

# EVOLVE-BLOCK-END
