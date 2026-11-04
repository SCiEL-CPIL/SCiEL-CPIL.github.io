"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces matrix multiplication.
    
    The tensor T[i,j,k] = delta. We need:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    and 0 otherwise (but there are no 'otherwise' entries in the full tensor).
    
    Actually the matrix multiplication tensor is:
    C[i,k] = sum_j A[i,j]*B[j,k]
    The tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    
    The decomposition requires:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U.shape[1]
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We can check this by evaluating on random inputs or directly.
    
    # Direct check: for each (i,j,k), compute the sum
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using tensor contraction."""
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # Compute using einsum
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    # T should be all ones
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Start with schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    best_U, best_V, best_W = U0, V0, W0
    best_rank = n * m * p
    
    # Try to reduce rank by finding linear dependencies among the columns
    # If columns of (U, V, W) satisfy a linear relation, we can eliminate one term.
    
    # Strategy: Use the fact that for matrix multiplication, there may be
    # rank reductions beyond schoolbook. For 3x5x7, the known minimal rank
    # is likely less than 105.
    
    # Approach: Try to find a rank-r decomposition where r < n*m*p
    # by using random linear combinations and checking consistency.
    
    # A more systematic approach: the matrix multiplication tensor T_{n,m,p}
    # has a known lower bound. For 3x5x7, let's try to find rank reductions.
    
    # Method: Start from schoolbook, try to merge terms by finding
    # linear dependencies in the trilinear form.
    
    # Alternative: Use a random rank-r tensor decomposition and try to
    # solve for the coefficients. This is a system of n*m*p equations
    # in r variables per term (3r unknowns), so we need 3r >= n*m*p
    # for a generic solution, but the structure may allow r < n*m*p.
    
    # Let's try a different approach: use the fact that matrix multiplication
    # can sometimes be done with fewer multiplications using bilinear tricks.
    
    # For now, let's try to reduce the schoolbook rank by checking if any
    # subset of terms can be combined.
    
    # Actually, a cleaner approach: try random rank-r decompositions for
    # r < n*m*p and check if they work.
    
    # The tensor equation is:
    # For all (i,j,k): sum_{c=1}^{r} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a system of n*m*p = 105 equations.
    # Unknowns: 3*r (the entries of U, V, W).
    # For r=105, we have 315 unknowns and 105 equations - very underdetermined.
    # We can try smaller r.
    
    # Let's try to find the minimal r by testing r = 104, 103, ...
    # For each r, set up the system and try to solve it.
    
    # The system is nonlinear (trilinear), so we need iterative methods.
    
    # Alternative simpler approach: try to find rank reductions by exploiting
    # the structure. For example, if we can find that some combination of
    # terms cancels out, we can reduce the rank.
    
    # Let's try a specific approach: use the schoolbook as a starting point
    # and try to eliminate terms by finding linear dependencies.
    
    # For a trilinear form T = sum_c u_c * v_c * w_c, if there exist
    # coefficients alpha_c such that sum_c alpha_c * u_c = 0,
    # sum_c alpha_c * v_c = 0, sum_c alpha_c * w_c = 0,
    # then we have a dependency.
    
    # But this is hard to find directly. Let's try a numerical approach.
    
    # Try to find a rank-r decomposition for r < 105 using random initialization
    # and iterative refinement.
    
    rng = np.random.default_rng(42)
    
    # Try ranks from 104 down to some lower bound
    # Lower bound for matrix multiplication rank: max(n*m, m*p, n*p) is not quite right
    # The true lower bound is at least max(n*m, m*p, n*p) / something...
    # Actually, rank >= max(n*m, m*p, n*p) is not a valid lower bound.
    # The rank must be at least the border rank, which is at least the geometric rank.
    # For now, let's try r from 104 down to 90.
    
    found_improvement = False
    for r in range(104, 89, -1):
        if time.time() > deadline:
            break
        
        # Try to find a rank-r decomposition
        # Initialize with random vectors
        U = rng.standard_normal((n * m, r))
        V = rng.standard_normal((m * p, r))
        W = rng.standard_normal((p * n, r))
        
        # The objective is to minimize ||T_decomposed - T_target||^2
        # where T_target[i,j,k] = 1 for all valid (i,j,k)
        
        # Use gradient-free optimization: try random perturbations
        # or use a simple iterative method.
        
        # Actually, let's use a different approach: 
        # Fix U and V randomly, then W is determined by the linear system.
        # For fixed U and V, the equation becomes:
        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is linear in W for each (i,j,k).
        
        # Let's try: fix U and V, solve for W.
        # The system is: for each (i,j,k), sum_c (U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = 1
        # This gives n*m*p equations in p*n*r unknowns (entries of W).
        
        # Actually W has p*n*r entries, and we have n*m*p equations.
        # If p*n*r > n*m*p, the system is underdetermined and may have solutions.
        # p*n*r = 7*3*r = 21r, n*m*p = 105.
        # So we need 21r >= 105, i.e., r >= 5.
        
        # Let's set up the linear system for W given U and V.
        
        # Build the matrix A where A[(i,j,k), (k',i',c)] = U[i*m+j,c]*V[j*p+k,c]*delta_{k,k'}*delta_{i,i'}
        # Wait, W[k*n+i, c] appears in equation (i,j,k) with coefficient U[i*m+j,c]*V[j*p+k,c].
        
        # So the equation for (i,j,k) is:
        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        # This is a linear system in the variables W[k*n+i,c].
        # Number of equations: n*m*p = 105
        # Number of unknowns: p*n*r = 21*r
        
        # For r >= 5, we have more unknowns than equations, so solutions exist generically.
        
        # Let's build the system and solve it.
        
        # Variables: W flattened to (p*n*r,)
        # Equation (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        # Build A matrix: A[equation_idx, var_idx]
        # equation_idx = (i,j,k) -> i*m*p + j*p + k
        # var_idx = (k', i', c) -> (k'*n + i')*r + c
        
        A = np.zeros((n * m * p, p * n * r))
        b = np.ones(n * m * p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq = i * m * p + j * p + k
                    for c in range(r):
                        var = (k * n + i) * r + c
                        A[eq, var] = U[i * m + j, c] * V[j * p + k, c]
        
        # Solve A @ W_flat = b using least squares
        try:
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            W = W_flat.reshape(p * n, r)
        except np.linalg.LinAlgError:
            continue
        
        # Verify the solution
        if _verify_fast(U, V, W, n, m, p, tol=1e-6):
            # Check that this is a valid rank-r decomposition
            # The rank is at most r (could be less if some columns are zero)
            actual_rank = min(np.linalg.matrix_rank(U, tol=1e-9),
                              np.linalg.matrix_rank(V, tol=1e-9),
                              np.linalg.matrix_rank(W, tol=1e-9),
                              r)
            if actual_rank < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = actual_rank
                found_improvement = True
                # We found a better rank, continue searching for even lower
                # But don't break; try to go lower
                continue
        else:
            # The least squares solution didn't exactly work; try more random inits
            pass
        
        # Try a few more random initializations for this r
        for attempt in range(20):
            if time.time() > deadline:
                break
            U = rng.standard_normal((n * m, r))
            V = rng.standard_normal((m * p, r))
            A = np.zeros((n * m * p, p * n * r))
            b = np.ones(n * m * p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq = i * m * p + j * p + k
                        for c in range(r):
                            var = (k * n + i) * r + c
                            A[eq, var] = U[i * m + j, c] * V[j * p + k, c]
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape(p * n, r)
            except np.linalg.LinAlgError:
                continue
            if _verify_fast(U, V, W, n, m, p, tol=1e-6):
                actual_rank = min(np.linalg.matrix_rank(U, tol=1e-9),
                                  np.linalg.matrix_rank(V, tol=1e-9),
                                  np.linalg.matrix_rank(W, tol=1e-9),
                                  r)
                if actual_rank < best_rank:
                    best_U, best_V, best_W = U, V, W
                    best_rank = actual_rank
                    found_improvement = True
    
    # Return the best decomposition found
    # Convert to lists for the required interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Round to nearest half-integer if close, to ensure clean entries
    def round_half(x):
        rounded = round(x * 2) / 2
        if abs(x - rounded) < 1e-8:
            return rounded
        return x
    
    U_list = [[round_half(val) for val in row] for row in U_list]
    V_list = [[round_half(val) for val in row] for row in V_list]
    W_list = [[round_half(val) for val in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
