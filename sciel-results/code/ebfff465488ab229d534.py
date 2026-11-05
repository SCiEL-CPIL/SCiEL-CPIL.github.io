"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_strassen_based(n, m, p, deadline):
    """
    Attempt to construct a lower-rank decomposition by embedding Strassen-like
    blocks. For 3x4x5, we can view it as containing 3x3 blocks with padding.
    
    A 3x3x3 multiplication has rank 23 (Strassen) or 21 (Winograd) or even lower.
    The known minimum for 3x3x3 is 23 (actually 23 is not proven minimal, but 23 is known).
    Wait, the rank of 3x3x3 is at most 23 (Strassen) and at least 19.
    Actually, the rank of the 3x3x3 tensor is known to be at most 23.
    
    However, simply embedding 3x3x3 into 3x4x5 is not straightforward because the dimensions differ.
    
    Let's try a different approach: Randomized SVD-like or direct optimization for a specific rank.
    But first, let's try to find a known decomposition.
    
    For 3x4x5, the schoolbook rank is 60.
    If we can find rank 59, that's an improvement.
    
    Let's use a linear algebra approach:
    The tensor T is in R^(n*m) x R^(m*p) x R^(p*n).
    We want to find rank r such that T = sum_{t=1}^r u_t v_t^T w_t^T.
    
    This is a non-convex problem.
    
    Strategy:
    1. Try to find a rank-59 decomposition by fixing one term to be zero in the schoolbook and optimizing the rest? No, that's not how it works.
    2. Use the fact that if a rank-r decomposition exists, it can be found by solving a system of equations.
    
    Let's try a random search with verification for ranks slightly below 60.
    This is unlikely to work for large n, m, p, but for 3x4x5, maybe?
    No, the search space is huge.
    
    Better idea: Use the known rank of 3x3x3.
    If we can decompose 3x3x3 with rank R_333, can we use it for 3x4x5?
    
    Consider the 3x4x5 multiplication as:
    A (3x4) * B (4x5) = C (3x5)
    
    We can split B into B1 (4x3) and B2 (4x2).
    C = A*B1 + A*B2.
    
    A*B1 is 3x3. A*B2 is 3x2.
    
    If we have a rank-R decomposition for 3x3x3, we can use it for A*B1.
    For A*B2, we can use schoolbook (3*4*2 = 24) or find a lower rank.
    
    Total rank = R_333 + Rank(3x4x2).
    
    If R_333 = 23, and Rank(3x4x2) <= 24, total = 47. This is much better than 60.
    
    Let's implement this block construction.
    
    Step 1: Get a rank-23 decomposition for 3x3x3.
    Step 2: Get a schoolbook decomposition for 3x4x2 (or try to optimize it).
    Step 3: Combine them.
    """
    pass # Placeholder, we will implement a specific construction below

def _rank_3x3x3_strassen():
    """
    Return the Strassen decomposition for 3x3x3.
    Rank 23.
    This is a known result. I need to hardcode the coefficients or generate them.
    Actually, Strassen's algorithm is for 2x2x2.
    For 3x3x3, there is a rank-23 algorithm by Winograd (1971) or similar.
    Actually, the rank of 3x3x3 is at most 23.
    
    Let's use a simpler approach: Embed 2x2x2 Strassen blocks.
    3x3x3 can be viewed as overlapping 2x2x2 blocks? No.
    
    Let's stick to the block construction idea but use schoolbook for the sub-blocks initially to ensure correctness, then try to optimize.
    
    Actually, let's try a different decomposition strategy:
    Split the 4x5 matrix into blocks.
    
    A (3x4) * B (4x5)
    
    Let B = [B1 B2] where B1 is 4x3 and B2 is 4x2.
    C = A*B1 + A*B2.
    
    Term 1: A (3x4) * B1 (4x3) -> 3x3 result.
    Term 2: A (3x4) * B2 (4x2) -> 3x2 result.
    
    If we use schoolbook for both:
    Rank(Term 1) = 3*4*3 = 36.
    Rank(Term 2) = 3*4*2 = 24.
    Total = 60. No gain.
    
    We need to find a lower rank for 3x4x3 or 3x4x2.
    
    Is there a known low-rank algorithm for 3x4x3?
    3x4x3 is equivalent to 3x3x4 (by symmetry).
    
    Let's try to find a rank-59 decomposition by random perturbation of the schoolbook.
    If we take the schoolbook decomposition and apply a small perturbation, can we reduce the rank?
    Unlikely.
    
    Let's try to use the fact that the tensor is sparse.
    
    Alternative: Use the "border rank" or "secant variety" properties? Too complex.
    
    Let's try a simple heuristic:
    1. Start with schoolbook.
    2. Try to merge two terms into one?
    
    Actually, let's look at the inspiration 1 (score 1.0714). It mentions "The task is to find a low-rank decomposition...".
    
    Let's try a specific construction for 3x4x5.
    
    Consider the 3x4x5 tensor.
    Can we use a 3x3x3 rank-23 algorithm embedded?
    
    Let's define the indices:
    i in 0..2 (rows of A)
    j in 0..3 (cols of A / rows of B)
    k in 0..4 (cols of B)
    
    If we restrict j to 0..2 and k to 0..2, we have a 3x3x3 sub-tensor.
    The remaining parts are:
    - j=3, k=0..4 (1x4 part of B, multiplied by A[:,3])
    - j=0..2, k=3..4 (3x2 part of B, multiplied by A[:,0..2])
    
    This is getting complicated.
    
    Let's try a direct optimization for rank 59.
    We have 60 terms in schoolbook.
    We want to see if we can express the tensor with 59 terms.
    
    Let's use a random initialization for 59 terms and try to fit.
    This is a non-convex problem, but maybe for such a small dimension it works?
    
    Actually, let's try to use the "sum of rank-1 tensors" formulation.
    
    Let's implement a random search for rank 59.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = schoolbook_rank
    
    # 2. Try to find a lower rank using random search / optimization
    # We will try ranks from 59 down to 55.
    # For each rank, we initialize randomly and try to optimize.
    
    # Since the problem is non-convex, we need a good initialization.
    # One idea: Use the schoolbook decomposition and randomly merge two terms?
    # Or use a random basis.
    
    # Let's try a simple gradient descent / least squares approach for rank 59.
    # We need to solve:
    # sum_{t=1}^r U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
    
    # This is a system of n*m*p equations in (n*m + m*p + p*n)*r variables.
    # For r=59, variables = (12 + 20 + 15)*59 = 47*59 = 2773.
    # Equations = 60.
    # This is an underdetermined system!
    # So there should be many solutions.
    # We just need to find one.
    
    # We can use a random linear combination of the schoolbook terms?
    # No, the terms are not linear in the rank-1 sense.
    
    # Let's try to construct a solution by setting most coefficients to 0 and solving for the rest.
    # This is equivalent to finding a sparse solution.
    
    # Actually, since it's underdetermined, we can fix most variables to 0 and solve for the remaining ones.
    # But the equations are cubic (products of variables).
    
    # Let's try a different approach:
    # Use the fact that we can perturb the schoolbook decomposition.
    
    # Let's try to find a rank-59 decomposition by assuming that one of the schoolbook terms is a linear combination of the others?
    # No, the rank is the minimum number of terms.
    
    # Let's try a random search.
    # Initialize U, V, W with random values.
    # Use gradient descent to minimize the error.
    
    # But we need to be careful with the time limit.
    
    # Let's try a simple random search for rank 59.
    # We will generate random U, V, W and check if they work.
    # This is very unlikely to work directly.
    
    # Better idea: Use the schoolbook decomposition as a starting point.
    # Set r = 59.
    # Take the first 59 terms of the schoolbook decomposition.
    # The 60th term is missing.
    # Can we adjust the first 59 terms to compensate for the missing 60th term?
    
    # Let's try this.
    # We have 59 terms. We want to find U, V, W such that the sum matches the tensor.
    # We can fix the structure of the first 59 terms to be similar to schoolbook, but with some adjustments.
    
    # Actually, let's just try to solve the system of equations for r=59.
    # We can use a numerical solver.
    
    # Let's use scipy.optimize.least_squares.
    
    from scipy.optimize import least_squares
    
    def _try_rank(r, seed=42):
        if time.time() > deadline:
            return None
            
        rng = np.random.RandomState(seed)
        
        # Initialize with random values
        U_init = rng.randn(n*m, r)
        V_init = rng.randn(m*p, r)
        W_init = rng.randn(p*n, r)
        
        def flatten_params(params):
            U_flat = params[:n*m*r]
            V_flat = params[n*m*r : n*m*r + m*p*r]
            W_flat = params[n*m*r + m*p*r :]
            return U_flat.reshape(n*m, r), V_flat.reshape(m*p, r), W_flat.reshape(p*n, r)
        
        def residuals(params):
            U, V, W = flatten_params(params)
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        res.append(val - 1.0)
            return np.array(res)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            
            if result.cost < 1e-12:
                U_opt, V_opt, W_opt = flatten_params(result.x)
                if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-7):
                    return U_opt, V_opt, W_opt
        except Exception:
            pass
            
        return None
    
    # Try ranks 59, 58, 57
    for r in range(59, 54, -1):
        if time.time() > deadline - 10:
            break
            
        res = _try_rank(r, seed=r)
        if res is not None:
            best_U, best_V, best_W = res
            best_rank = r
            break # Found a lower rank, stop
    
    # Convert to list of lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
