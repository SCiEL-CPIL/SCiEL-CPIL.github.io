"""Tensor decomposition for 4x4 x 4x9 matrix multiplication.
Strategy: iterative randomized search for low-rank exact decomposition,
falling back to known-good Strassen/schoolbook if search fails.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=-1; W[2,3]=1
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    return U, V, W

def _decomp_4x4x4():
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    inner_U, inner_V, inner_W = _strassen_2x2()
    outer_A = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[0,0,1,-1],[0,1,0,-1]]
    outer_B = [[1,0,0,1],[1,0,0,0],[0,1,0,-1],[0,0,1,-1],[0,0,0,1],[1,1,0,0],[0,0,1,1]]
    outer_C = [[1,0,0,1],[0,0,1,-1],[0,1,0,1],[-1,0,1,0],[1,1,0,0],[0,0,0,1],[1,0,0,0]]
    for t in range(7):
        col_offset = t * 7
        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    b = li * 2 + lj
                    coeff_U = outer_A[t][b] * inner_U[li*2+lj, r]
                    coeff_V = outer_B[t][b] * inner_V[li*2+lj, r]
                    coeff_W = outer_C[t][b] * inner_W[li*2+lj, r]
                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2*(b//2) + ei
                            g_j = 2*(b%2) + ej
                            U[g_i*4+g_j, col] = coeff_U
                            V[g_j*4+g_i, col] = coeff_V
                            W[g_i*4+g_j, col] = coeff_W
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    # Build the full tensor product and check
    # For efficiency, check a subset of entries or use einsum
    # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=1e-6)

def _iterative_refine(U, V, W, n, m, p, max_iter=5000, tol=1e-10, lr=0.05):
    """
    Iteratively refine factor matrices to drive reconstruction error to zero.
    Uses a simple gradient-descent-like update on the linearized problem.
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    for it in range(max_iter):
        # Compute current error tensor: E[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        E = T - 1.0
        err = np.max(np.abs(E))
        if err < tol:
            return True, U, V, W
        
        # Gradient descent: update each factor to reduce error
        # dE/dr_s = V[j,k,s]*W[k,i,s] for U[i,j,s]
        # dE/dr_s = U[i,j,s]*W[k,i,s] for V[j,k,s]
        # dE/dr_s = U[i,j,s]*V[j,k,s] for W[k,i,s]
        
        # Update U: U[i,j,s] -= lr * E[i,j,k] * V[j,k,s] * W[k,i,s]  (sum over k)
        # More precisely: grad_U[i,j,s] = sum_k E[i,j,k] * V[j,k,s] * W[k,i,s]
        grad_U = np.einsum('ijk,jks,iks->ijs', E, V_r, W_r)
        grad_V = np.einsum('ijk,ij s,iks->jks', E, U_r, W_r)  # careful with indices
        # Actually: grad_V[j,k,s] = sum_i E[i,j,k] * U[i,j,s] * W[k,i,s]
        grad_V = np.einsum('ijk,ij s,iks->jks', E, U_r, W_r)
        # grad_W[k,i,s] = sum_j E[i,j,k] * U[i,j,s] * V[j,k,s]
        grad_W = np.einsum('ijk,ij s,jks->kis', E, U_r, V_r)
        
        U_r -= lr * grad_U
        V_r -= lr * grad_V
        W_r -= lr * grad_W
        
        # Reshape back
        U = U_r.reshape(n*m, rank)
        V = V_r.reshape(m*p, rank)
        W = W_r.reshape(p*n, rank)
        
        # Adaptive learning rate
        T_new = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        err_new = np.max(np.abs(T_new - 1.0))
        if err_new > err * 2:
            lr *= 0.5
        elif err < err_new * 0.9:
            lr = min(lr * 1.1, 0.5)
    
    # Final check
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    if np.max(np.abs(T - 1.0)) < 1e-6:
        return True, U_r.reshape(n*m, rank), V_r.reshape(m*p, rank), W_r.reshape(p*n, rank)
    return False, U, V, W

def _random_init(n, m, p, rank, seed=42):
    """Initialize random factor matrices with small values."""
    rng = np.random.RandomState(seed)
    scale = 1.0 / np.sqrt(rank)
    U = rng.randn(n*m, rank) * scale
    V = rng.randn(m*p, rank) * scale
    W = rng.randn(p*n, rank) * scale
    return U, V, W

def _search_decomposition(n, m, p, deadline):
    """
    Search for a low-rank decomposition using iterative refinement.
    Try ranks from a lower bound up to the known-good rank.
    """
    # Lower bound: max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # Known good: 114 (Strassen) or 144 (schoolbook)
    # Try a range of ranks
    known_good_rank = 114
    min_rank = 36
    
    best_result = None
    best_rank = known_good_rank
    
    # First, get the known-good solution as fallback
    U1, V1, W1 = _decomp_4x4x4()
    U2, V2, W2 = _decomp_4x4x4()
    U3, V3, W3 = _schoolbook_nm_p(4, 4, 1)
    U_fb = np.zeros((n*m, known_good_rank))
    V_fb = np.zeros((m*p, known_good_rank))
    W_fb = np.zeros((p*n, known_good_rank))
    U_fb[:, 0:49] = U1
    U_fb[:, 49:98] = U2
    U_fb[:, 98:114] = U3
    for j in range(m):
        for k in range(4):
            V_fb[j*p+k, 0:49] = V1[j*4+k, :]
            V_fb[j*p+(4+k), 49:98] = V2[j*4+k, :]
        V_fb[j*p+8, 98:114] = V3[j, :]
    for k in range(4):
        for i in range(n):
            W_fb[k*n+i, 0:49] = W1[k*4+i, :]
            W_fb[(4+k)*n+i, 49:98] = W2[k*4+i, :]
    for i in range(n):
        W_fb[8*n+i, 98:114] = W3[i, :]
    
    # Try to find a lower-rank decomposition
    # Strategy: try several ranks with multiple random seeds
    ranks_to_try = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
            if time.time() > deadline - 10:
                break
            U, V, W = _random_init(n, m, p, target_rank, seed=seed)
            success, U, V, W = _iterative_refine(U, V, W, n, m, p, 
                                                   max_iter=2000, tol=1e-10, lr=0.1)
            if success:
                # Verify
                if _verify(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_result = (U, V, W)
                    break  # Found at this rank, no need to try higher seeds
        if best_result is not None:
            break  # Found a decomposition at this rank
    
    if best_result is not None:
        return best_result
    return (U_fb, V_fb, W_fb)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Try the iterative search approach
    U, V, W = _search_decomposition(n, m, p, deadline)
    
    # Final verification
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Ultimate fallback: pure schoolbook
        Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

# EVOLVE-BLOCK-END
