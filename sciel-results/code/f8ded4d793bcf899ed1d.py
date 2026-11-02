"""Rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 150.0  # wall-clock seconds; evaluator kills at 150s
SAFETY_MARGIN = 5.0  # return best result this much before the limit


def _remaining_time():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def _verify_exact(U, V, W, n, m, p):
    """Check whether U,V,W (lists or arrays of shape (n*m,rank),(m*p,rank),(p*n,rank))
    exactly decompose the matmul tensor under the trace convention W[k*n+i, r].

    Returns True if for all i,j,k:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if i==j else 0
    (i.e., the (i,j,k) slice of the product equals delta_{i,j})
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # Build the full tensor: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # We check against the target tensor.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                target = 1.0 if i == j else 0.0
                if abs(s - target) > 1e-9:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification using einsum."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Reshape U to (n,m,rank), V to (m,p,rank), W to (p,n,rank)
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    # Compute T[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr, optimize=True)
    # Target: T[i,j,k] = 1 if i==j else 0
    target = np.zeros((n, m, p))
    for i in range(n):
        target[i, i, :] = 1.0
    return np.allclose(T, target, atol=1e-9)


def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition as lists of lists."""
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


def _structured_block_decomposition(n, m, p):
    """
    Try to find a low-rank decomposition by exploiting structure.

    Idea: The matmul tensor for C = A*B where A is n x m, B is m x p, C is n x p.
    C[i,k] = sum_j A[i,j] * B[j,k].
    
    The tensor is T[i,j,k] = delta_{i,j} (in the sense that the product equals identity when
    we contract appropriately). Actually the target is:
    sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = delta_{i,j}

    This is equivalent to: for each i, the "row" of U indexed by (i,*) times the corresponding
    V and W must reconstruct the identity pattern.

    Let's think of it differently. We want rank-r tensors U, V, W such that:
    For all i,j,k: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = delta_{i,j}

    One structured approach: split the multiplication into blocks.
    
    For n=3, m=5, p=6:
    - We can write the matmul as a sum of smaller matmuls.
    - For example, split m into groups.
    
    Alternative: Use the fact that we only need delta_{i,j}. This means:
    - When i != j, the contribution must be 0.
    - When i == j, the contribution must be 1 for all k.

    Let's try a decomposition where each "rank-1" term is localized.
    
    Actually, let me try a direct approach: 
    The tensor T[i,j,k] = delta_{i,j} has a natural decomposition.
    For each i, we need sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = delta_{i,j}.
    
    Consider: For fixed i, define vectors u_i[j,r] = U[i*m+j,r], v_j[k,r] = V[j*p+k,r], w_k[i,r] = W[k*n+i,r].
    Then sum_r u_i[j,r] v_j[k,r] w_k[i,r] = delta_{i,j}.
    
    This is a 3-tensor equation. The minimal rank is what we seek.
    
    Let's try a simple structured decomposition:
    - Use r terms where each term handles a specific (i,j) pair partially.
    
    For each i (3 values), we need the j=i row to give 1 and all other j to give 0.
    
    Idea: For each i, use m terms to handle the m different j values.
    But we can share terms across i values if possible.
    
    Let me try: rank = n * m = 15? Or n * p = 18? Or m * p = 30?
    
    Actually, let's think about it as:
    T[i,j,k] = delta_{i,j} * 1 (for all k)
    
    We can write this as:
    T[i,j,k] = sum_{r} U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    
    If we set for each (i,j) pair a dedicated set of terms, we get rank = n*m*p (schoolbook).
    
    To reduce rank, we need to share terms. 
    
    Let's try a block approach:
    - Group the p dimension. For each k, we need the same delta_{i,j} pattern.
    - So for each k, we need a rank-r decomposition of the n x m identity-like matrix.
    
    The matrix M_k[i,j] = delta_{i,j} for i,j in range(n) (but j goes up to m, so it's n x m with identity in top-left n x n if n<=m, else different).
    
    Here n=3, m=5, so M_k is a 3x5 matrix with 1s at (0,0),(1,1),(2,2) and 0s elsewhere.
    
    We need: M_k[i,j] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    
    For fixed k, define A_k[i,j] = U[i*m+j,r] (as a 3x5 matrix for each r), 
    B_k[j,r] = V[j*p+k,r] (as a 5 x rank matrix),
    C_k[i,r] = W[k*n+i,r] (as a 3 x rank matrix).
    
    Then M_k[i,j] = sum_r A_k[i,j,r] * B_k[j,r] * C_k[i,r]
    
    This is a "tensor" contraction that's unusual. Let me rewrite:
    M_k[i,j] = sum_r C_k[i,r] * A_k[i,j,r] * B_k[j,r]
    
    Note A_k depends on both i and j. This makes it non-standard.
    
    Let me try a different approach: use known bounds and search.
    
    For small matrices, the minimal rank is often close to the schoolbook rank but can be lower.
    For 2x2x2, the minimal rank is 7 (Strassen).
    For 3x3x3, the best known is 23 (Landsberg).
    
    For 3x5x6, let's try to find a decomposition with rank < 90.
    
    Let me try a simple heuristic: start with schoolbook and try to merge terms.
    """
    n, m, p = 3, 5, 6
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U = np.array(best["U"], dtype=float)
    best_V = np.array(best["V"], dtype=float)
    best_W = np.array(best["W"], dtype=float)
    
    # Try to reduce rank by finding linear dependencies
    # The decomposition is valid if the tensor equation holds.
    # We can try to find a lower-rank decomposition by solving a system of equations.
    
    # Approach: For a given rank r, we have r*n*m + r*m*p + r*p*n unknowns.
    # We have n*m*p equations (one for each (i,j,k) target).
    # The equations are cubic (trilinear in U,V,W), so this is non-convex.
    
    # Let's try a local search approach:
    # 1. Start with schoolbook rank 90.
    # 2. Try to zero out some entries and see if we can still satisfy the equations.
    # 3. Or try to reduce the number of non-zero terms.
    
    # Simpler approach: try structured decompositions with specific patterns.
    
    # Pattern 1: For each i, use m terms to handle all j,k.
    # U[i*m+j, r] is non-zero only for r in [i*m, i*m+m)
    # V[j*p+k, r] is non-zero for all r
    # W[k*n+i, r] is non-zero only for r in [i*m, i*m+m)
    
    # Let me try: rank = n*m = 15
    # For each i, j: U[i*m+j, i*m+j] = 1 (diagonal in the i*m+j block)
    # V[j*p+k, i*m+j] = ? 
    # W[k*n+i, i*m+j] = ?
    
    # This is getting complex. Let me just do a numerical search.
    
    # Try various ranks and use least-squares to find U,V,W.
    # Since the problem is trilinear, we can use alternating minimization.
    
    def _alternating_minimization(n, m, p, rank, max_iter=100, tol=1e-8):
        """Try to find U,V,W of given rank using alternating least squares."""
        # Initialize randomly
        random.seed(42)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Target tensor
        target = np.zeros((n, m, p))
        for i in range(n):
            target[i, i, :] = 1.0
        
        for it in range(max_iter):
            if _remaining_time() < 1.0:
                break
            
            # Fix V and W, solve for U
            # For each (i,j): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = target[i,j,k]
            # This is linear in U[i*m+j,:] for each (i,j,k).
            # But it's coupled across k. Let's write it as:
            # For fixed (i,j): for each k: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = target[i,j,k]
            # So U[i*m+j,:] * (V[j*p+k,:] * W[k*n+i,:])^T = target[i,j,k]
            # This is a system: for each k, we have an equation.
            # Let B_k = V[j*p+k,:] * W[k*n+i,:] (element-wise product, shape (rank,))
            # Then U[i*m+j,:] * B_k^T = target[i,j,k]
            # So we have p equations for U[i*m+j,:] (rank unknowns).
            # If rank <= p, we can solve exactly. If rank > p, least squares.
            
            # Build the system for U
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    # Build matrix A of shape (p, rank): A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                    b = np.array([target[i, j, k] for k in range(p)])
                    # Solve A^T * U[i*m+j,:]^T = b  (i.e., A @ U[i*m+j,:] = b)
                    # Actually: sum_r A[k,r] * U[i*m+j,r] = b[k] for all k
                    # So A @ U[i*m+j,:] = b
                    if rank <= p:
                        # Under-determined or square, use least squares
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U_new[i * m + j, :] = sol
                    else:
                        # Over-determined, least squares
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U_new[i * m + j, :] = sol
            U = U_new
            
            # Similarly for V
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    # For fixed (j,k): sum_r V[j*p+k,r] * U[i*m+j,r] * W[k*n+i,r] = target[i,j,k]
                    # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    A = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                    b = np.array([target[i, j, k] for i in range(n)])
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j * p + k, :] = sol
            V = V_new
            
            # Similarly for W
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    # For fixed (k,i): sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r] = target[i,j,k]
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(range(rank)):
                            A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                    b = np.array([target[i, j, k] for j in range(m)])
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k * n + i, :] = sol
            W = W_new
            
            # Check convergence
            if it % 10 == 0:
                # Compute residual
                Ur = U.reshape(n, m, rank)
                Vr = V.reshape(m, p, rank)
                Wr = W.reshape(p, n, rank)
                T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr, optimize=True)
                residual = np.linalg.norm(T - target)
                if residual < tol:
                    break
        
        # Final check
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr, optimize=True)
        residual = np.linalg.norm(T - target)
        return U, V, W, residual
    
    # Try decreasing ranks
    best_solution = None
    best_rank_found = best_rank
    
    for rank in range(89, 14, -1):  # Try ranks from 89 down to 15
        if _remaining_time() < 5.0:
            break
        
        U, V, W, residual = _alternating_minimization(n, m, p, rank, max_iter=200)
        
        if residual < 1e-6:
            # Verify more carefully
            if _verify_fast(U, V, W, n, m, p):
                best_solution = (U, V, W)
                best_rank_found = rank
                break  # Found a valid solution at this rank
    
    if best_solution is not None:
        U, V, W = best_solution
        # Convert to lists of lists with rounded values
        U_list = [[round(float(x), 6) for x in U[i]] for i in range(U.shape[0])]
        V_list = [[round(float(x), 6) for x in V[i]] for i in range(V.shape[0])]
        W_list = [[round(float(x), 6) for x in W[i]] for i in range(W.shape[0])]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better solution found, return schoolbook
    return best


def construct():
    """
    Returns a tensor decomposition for 3x5x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # First try to find a better than schoolbook decomposition
    try:
        result = _structured_block_decomposition(n, m, p)
        # Verify the result
        if _verify_fast(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)


# Ensure we don't exceed time limit
if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
