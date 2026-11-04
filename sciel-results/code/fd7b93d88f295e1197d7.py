"""Matrix multiplication 2x3 by 3x9 — tensor rank decomposition search."""
import time
import sys

DIMENSIONS = [2, 3, 9]
DEADLINE = 130  # seconds

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify(n, m, p, U, V, W):
    """Check that sum_r U[i,j]*V[j,k]*W[k,i] == delta.
    Index convention: U row index = i*m+j (size n*m),
                      V row index = j*p+k (size m*p),
                      W row index = k*n+i (size p*n).
    """
    rank = len(U[0]) if U else 0
    # Build the tensor as sum over r
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                # s should be 1 if this is the (i,j,k) entry
                # Actually we need s == 1 for the identity check.
                # The tensor T[i,j,k] = delta_{i,j,k} in the matmul sense? No.
                # The matmul tensor: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
                # As a 3-tensor: T[i,j,k] is the coefficient, which is always 1.
                # So sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] must equal 1 for all i,j,k.
                if abs(s - 1) > 1e-6:
                    return False
    return True


def _try_bilinear_reduction(n, m, p, max_rank):
    """Try to find a decomposition of rank <= max_rank using structured ansatz.
    
    Strategy: parameterize U with random integer/rational entries and solve
    the resulting linear system for V and W. This works when the number of
    free parameters is manageable.
    
    For small ranks we can enumerate or use random search + least squares.
    """
    import numpy as np
    
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    total_entries = n * m * p  # number of constraints (i,j,k)
    
    # For a given rank r, we have r*(dim_U + dim_V + dim_W) variables
    # and total_entries constraints (each must equal 1).
    # This is a system of cubic equations (bilinear in (U,W) and V).
    # 
    # Alternative: fix U, then the problem becomes bilinear in (V, W):
    # sum_r V[j*p+k][r] * W[k*n+i][r] = 1 / U[i*m+j][r] ... no, that's not right either.
    #
    # Actually: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
    # If we fix U, then for each pair (i, j, k), the equation is:
    #   sum_r U[i*m+j][r] * (V[j*p+k][r] * W[k*n+i][r]) = 1
    # Let X[r, j, k, i] = V[j*p+k][r] * W[k*n+i][r]. Then:
    #   sum_r U[i*m+j][r] * X[r, j, k, i] = 1 for all (i,j,k)
    #
    # This is still bilinear in V and W. 
    #
    # Simpler approach: for very small rank reductions, try known patterns.
    
    return None  # No quick win from this method in the time budget


def _structured_search(n, m, p, deadline):
    """Search for lower-rank decompositions using structured parameterizations.
    
    Key idea: exploit the structure of 2x3 by 3x9 multiplication.
    
    The 2x3 to 3x9 matmul has tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0..8}.
    The tensor shape is (2, 3, 9). The border rank gives upper bounds.
    
    Lower bound: max(n, m, p) ... actually the lower bound on tensor rank for
    multiplying an n x m by m x p matrix is at least max(n, p) * something...
    The trivial lower bound is max(n*m, m*p, n*p) / max(n, m, p) but that's not tight.
    Actually, a simple lower bound: rank >= ceil(n*m*p / max(n*m, m*p, n*p)) ... no.
    
    For 2x3 x 3x9:
      - U has 6 rows, V has 27 rows, W has 18 rows
      - The tensor has 54 nonzero entries
      - Rank must be at least... well, at least max(6, 27, 18) = 27? No, that's not a valid lower bound for tensor rank.
      
    Let me think differently. The standard lower bound is that the rank must be at least the
    maximum of the matrix ranks of all slices. For the T_{ij} slice (fixing i,j), it's a 1x9 
    all-ones vector, rank 1. For T_{jk} slice (fixing j,k), it's a 2x1 column of ones, rank 1.
    For T_{ki} slice (fixing k,i), it's a 3x1 column of ones, rank 1.
    So that bound is just 1, which is useless.
    
    Better: the rank must be at least the dimension of the span of any fiber.
    Fiber in i direction (fix j,k): vector in R^2 = (1,1), so span is 1-dim.
    Fiber in j direction (fix i,k): vector in R^3 = (1,1,1), span is 1-dim.
    Fiber in k direction (fix i,j): vector in R^9 = all ones, span is 1-dim.
    
    So the trivial lower bound is 1. The schoolbook rank is 54.
    
    Realistic approach: try to find rank r where 17 <= r <= 53 using random/structured search.
    """
    import numpy as np
    import random
    
    start_time = time.time()
    dim_U = n * m  # 6
    dim_V = m * p  # 27
    dim_W = p * n  # 18
    
    best_result = None
    
    # Try various rank values, starting from a reduced guess
    # For 2x3 x 3x9, a reasonable first guess for reduced rank might be around 20-40
    # Let's try several ranks.
    
    for r in range(max(dim_U, dim_W, dim_V), min(54, max(dim_U, dim_W, dim_V) + 20)):
        if time.time() - start_time > deadline - 30:
            break
        if time.time() > DEADLINE - 15:
            break
        
        # Try random initializations
        for attempt in range(50):
            if time.time() - start_time > deadline - 20:
                break
            
            # Random integer entries in {-1, 0, 1} with some structure
            np.random.seed(attempt + r * 1000)
            U = np.random.randint(-1, 2, size=(dim_U, r)).astype(float)
            # Ensure U has enough rank
            if np.linalg.matrix_rank(U) < min(dim_U, r):
                continue
            
            # Given U, solve for V and W:
            # sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
            #
            # Let M[j,k,i,r] = U[i*m+j][r]. Then we need:
            #   V and W such that sum_r M[j,k,i,r] * V[j*p+k, r] * W[k*n+i, r] = 1
            #
            # This is equivalent to: for each r, define A_r (m x p) and B_r (p x n) where
            #   A_r[j,k] = V[j*p+k, r], B_r[i,k] = W[k*n+i, r]
            # Then: sum_r U[i*m+j][r] * A_r[j,k] * B_r[k,i] = 1
            #
            # Let C = A_r (size 3x9) and D_r = B_r^T (size 9x2). Then:
            #   sum_r U[i*m+j][r] * (C_r @ D_r)[j,i] ... hmm, indices don't match cleanly.
            #
            # Actually C_r @ D_r has shape (3, 2), entry [j,i] = sum_k A_r[j,k]*D_r[k,i]
            # = sum_k V[j*p+k, r] * W[k*n+i, r]. That's exactly what we need!
            #
            # So the constraint is: sum_r U[i*m+j][r] * (C_r @ D_r)[j,i] = 1 for all (i,j).
            # This is a system of n*m = 6 equations, but each equation involves bilinear terms.
            #
            # The number of unknowns: r * (m*p + p*n) = r * (27 + 18) = 45r
            # The number of equations: n * m * p = 54 (one per (i,j,k) triple)
            # Wait, no. Let me re-examine.
            #
            # The constraint is: for ALL (i,j,k): sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
            # That's 54 scalar equations.
            #
            # If we fix U (6*r = 6r knowns), unknowns are V (27r) and W (18r), total 45r unknowns.
            # 54 equations in 45r unknowns. For r >= 2, 45r >= 90 > 54, so generally solvable.
            #
            # But the equations are nonlinear (products of V and W entries).
            #
            # Approach: iterate. Given U and an initial V, solve linear for W, then given U and W, solve linear for V, etc.
            # Alternating least squares (ALS) style.
            
            # Initialize V randomly
            V = np.random.randn(dim_V, r)
            W = np.random.randn(dim_W, r)
            
            # ALS iterations
            converged = False
            for it in range(200):
                if time.time() - start_time > deadline - 10:
                    break
                
                # Solve for W: for each (j, k, r), coefficient is U[i*m+j][r]*V[j*p+k][r]
                # We need: for each (i,j,k): sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
                # Rewrite: for each (i, j): sum_r (U[i*m+j][r] * V[j*p+k][r]) * W[k*n+i][r] ... 
                # Hmm, the sum is over r and the constraint is over (i,j,k) jointly.
                #
                # Let me set up the linear system for W given U and V.
                # For each (i, j, k): sum_r [U[i*m+j][r] * V[j*p+k][r]] * W[k*n+i][r] = 1
                #
                # This is: for each (i, j, k), a linear equation in W.
                # W has dim_W * r = 18r unknowns.
                # We have 54 equations.
                # Let's build the 54 x (18r) matrix.
                
                # Actually, let's be more careful with indexing.
                # W is indexed by [k*n + i, r] for k in range(p), i in range(n), r in range(r).
                
                A_W = np.zeros((n * m * p, dim_W * r))
                b_W = np.ones(n * m * p)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for r in range(r):
                                col_idx = (k * n + i) * r + r  # (k*n+i) * r + r_offset
                                # Wait, W[k*n+i][r] is at position (k*n+i, r) in the matrix
                                # Flattened: (k*n+i) * r + r
                                coeff = U[i * m + j][r] * V[j * p + k][r]
                                A_W[eq_idx, (k * n + i) * r + r] = coeff
                
                # Solve least squares for W_flat
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A_W, b_W, rcond=None)
                W = W_flat.reshape(dim_W, r)
                
                # Solve for V: for each (i, j, k): sum_r [U[i*m+j][r] * W[k*n+i][r]] * V[j*p+k][r] = 1
                A_V = np.zeros((n * m * p, dim_V * r))
                b_V = np.ones(n * m * p)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for r in range(r):
                                col_idx = (j * p + k) * r + r
                                coeff = U[i * m + j][r] * W[k * n + i][r]
                                A_V[eq_idx, col_idx] = coeff
                
                V_flat, residuals, rank_A, sv = np.linalg.lstsq(A_V, b_V, rcond=None)
                V = V_flat.reshape(dim_V, r)
            
            # After ALS, check convergence
            # Verify
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(r):
                            s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        max_err = max(max_err, abs(s - 1))
            
            if max_err < 1e-4:
                # Try to snap to rational
                U_snap = np.round(U * 100) / 100
                V_snap = np.round(V * 100) / 100
                W_snap = np.round(W * 100) / 100
                max_err_snap = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(r):
                                s += U_snap[i * m + j][r] * V_snap[j * p + k][r] * W_snap[k * n + i][r]
                            max_err_snap = max(max_err_snap, abs(s - 1))
                if max_err_snap < 1e-3:
                    best_result = {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                    return best_result
    
    return best_result


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # First, always have the schoolbook as fallback
    fallback = _schoolbook(n, m, p)
    
    # Try structured search with time budget
    result = _structured_search(n, m, p, deadline=DEADLINE - 15)
    
    if result is not None and time.time() < DEADLINE - 5:
        return result
    
    return fallback
# EVOLVE-BLOCK-END
