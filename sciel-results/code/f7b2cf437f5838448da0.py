"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication.
    
    The multiplication tensor T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k]
    has a term involving A[i,j]*B[j,k].
    
    We want to write T as sum_r U[:,r] * V[:,r] * W[:,r] (outer product of three vectors)
    where:
      U has shape (n*m, rank) with indices i*m+j
      V has shape (m*p, rank) with indices j*p+k
      W has shape (p*n, rank) with indices k*n+i
    
    For the 2x2*2x3 case, n=2, m=2, p=3.
    Schoolbook rank = n*m*p = 12.
    
    The known minimal rank for 2x2 * 2x3 matrix multiplication is 8 (from literature on
    small matrix multiplication). Let's try to find rank 8, 7, etc.
    
    Strategy: We'll use an algebraic approach. The tensor T[i,j,k] is 1 for all valid
    (i,j,k) in range(n) x range(m) x range(p). We want to decompose it into rank r.
    
    For a given rank r, we need to find U (2*2 x r), V (2*3 x r), W (3*2 x r) such that
    for all i in range(2), j in range(2), k in range(3):
      sum_{c=0}^{r-1} U[i*2+j, c] * V[j*3+k, c] * W[k*2+i, c] = 1
    
    And for all other combinations (which don't exist since all i,j,k are valid), it's 0.
    
    Actually, all (i,j,k) with i in [0,2), j in [0,2), k in [0,3) are valid, so all 12
    entries are 1.
    
    Let me think about this differently. The matrix multiplication tensor for m1 x m2 * m2 x m3
    has rank at least max(m1*m2, m2*m3, m1*m3) and at most m1*m2*m3.
    
    For 2x2 * 2x3:
    - Lower bound: max(4, 6, 6) = 6
    - Schoolbook: 12
    
    Known results: The rank of 2x2 * 2x3 matrix multiplication is 8.
    
    Let me try to construct a rank 8 decomposition.
    
    One approach: Use the fact that we can group terms. 
    
    Let me set up the system. We have 12 equations (one for each (i,j,k)) and 3*r unknowns
    (U has 4r entries, V has 6r entries, W has 6r entries, but they're coupled).
    
    This is a non-linear system. Let me try a different approach.
    
    Actually, let me try to use a known construction. For 2x2 * 2x3, one can achieve rank 8
    by combining two 2x2 multiplications (each rank 4 or less) plus some corrections.
    
    Wait, 2x2 * 2x3 can be seen as multiplying a 2x2 matrix by two 2x3 matrices side by side,
    but that's not quite right either.
    
    Let me just try to solve it numerically. I'll parameterize with r and try to find
    a solution.
    
    For rank r, I'll set up the equations and try to solve them.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=3
    
    # The tensor T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # Let's try ranks from 6 upward
    deadline = time.time() + 140  # Leave 10s margin
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook
    
    # First, create the schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try to find lower rank decompositions
    for r in range(6, best_rank):
        if time.time() > deadline:
            break
        
        if not _try_rank(n, m, p, r, deadline):
            continue
        
        # Try to find a decomposition of rank r
        result = _find_decomposition(n, m, p, r, deadline)
        if result is not None:
            best_result = result
            best_rank = r
            break  # We found the minimum rank, stop
    
    return best_result


def _try_rank(n, m, p, r, deadline):
    """Quick check if rank r is feasible (always true for r >= lower bound)."""
    # For 2x2 * 2x3, lower bound is max(n*m, m*p, n*p) = max(4,6,6) = 6
    return r >= max(n * m, m * p, n * p)


def _find_decomposition(n, m, p, r, deadline):
    """
    Try to find a rank-r decomposition.
    Uses a semi-random initialization and iterative refinement.
    """
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Scale to make the sum of products approximately 1
        for c in range(r):
            U[:, c] *= 2.0 / (r ** 0.5)
            V[:, c] *= 2.0 / (r ** 0.5)
            W[:, c] *= 2.0 / (r ** 0.5)
        
        # Iteratively refine
        converged = False
        for it in range(200):
            if time.time() > deadline:
                return None
            
            # For each (i,j,k), we need:
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            #
            # Fix U and V, solve for W:
            # For each k,n,i: sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1
            # But this is coupled across j for fixed i,k.
            #
            # Let's fix U and W, solve for V, or use a different strategy.
            #
            # Actually, let's fix two matrices and solve for the third.
            #
            # Fix U and V, solve for W:
            # For each (i,k): sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This doesn't factor nicely because of the sum over j.
            
            # Let me try a different approach: use the structure more carefully.
            # 
            # For fixed i: we have m equations (one for each j) for each k.
            # Actually, let's think of it as: for each i, we're multiplying a 1xm row
            # (from A) by an mxp matrix (from B). 
            #
            # Let me try: fix U and W, solve for V.
            # For each (j,k): sum_i sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is still coupled.
            
            # Let me use a simpler iterative method: gradient descent on the loss
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        loss += (val - 1.0) ** 2
            
            if loss < 1e-10:
                converged = True
                break
            
            # Compute gradients and update
            # dL/dU[i*m+j,c] = 2 * (val_ijc - 1) * V[j*p+k,c] * W[k*n+i,c]
            # But val depends on which (i,j,k) we're at, and each (i,j,k) contributes
            # to specific U,V,W entries.
            
            # Let's do a simple coordinate descent / alternating least squares approach.
            # Fix U and V, update W.
            # For each (i,k): we need sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Let M_{(i,k), c} = sum_j U[i*m+j,c] * V[j*p+k,c]
            # Then we need: M_{(i,k), :} . W[k*n+i, :] = 1
            # This is a linear system for W[k*n+i, :]!
            
            W_new = W.copy()
            for i in range(n):
                for k in range(p):
                    # M[c] = sum_j U[i*m+j, c] * V[j*p+k, c]
                    M = np.zeros(r)
                    for j in range(m):
                        M += U[i*m+j, :] * V[j*p+k, :]
                    # Solve M . W[k*n+i, :] = 1
                    # This is underdetermined if r > 1, so use least squares
                    # Actually we want M @ w = 1 where w is a vector of length r
                    # Use np.linalg.lstsq
                    try:
                        w_sol, residuals, rank_mat, sv = np.linalg.lstsq(M.reshape(-1,1), np.ones(1), rcond=None)
                        W_new[k*n+i, :] = w_sol.flatten()
                    except:
                        pass
            
            W = W_new
            
            # Similarly, fix V and W, update U.
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    # We need: sum_k V[j*p+k, :] * W[k*n+i, :] dot product with U[i*m+j, :] = 1
                    # Let N[c] = sum_k V[j*p+k, c] * W[k*n+i, c]
                    N = np.zeros(r)
                    for k in range(p):
                        N += V[j*p+k, :] * W[k*n+i, :]
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(N.reshape(-1,1), np.ones(1), rcond=None)
                        U_new[i*m+j, :] = u_sol.flatten()
                    except:
                        pass
            U = U_new
            
            # Fix U and W, update V.
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    # We need: sum_i U[i*m+j, :] * W[k*n+i, :] dot V[j*p+k, :] = 1
                    P = np.zeros(r)
                    for i in range(n):
                        P += U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(P.reshape(-1,1), np.ones(1), rcond=None)
                        V_new[j*p+k, :] = v_sol.flatten()
                    except:
                        pass
            V = V_new
        
        # Check if we converged
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    loss += (val - 1.0) ** 2
        
        if loss < 1e-10:
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify the rounded solution
            loss_rounded = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                        loss_rounded += (val - 1.0) ** 2
            
            if loss_rounded < 1e-9:
                # Convert to lists
                U_list = [[float(U_rounded[i, c]) for c in range(r)] for i in range(n*m)]
                V_list = [[float(V_rounded[i, c]) for c in range(r)] for i in range(m*p)]
                W_list = [[float(W_rounded[i, c]) for c in range(r)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
# EVOLVE-BLOCK-END
