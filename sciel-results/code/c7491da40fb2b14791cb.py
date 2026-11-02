"""Rewritten program for matrix multiplication 3x3 * 3x7 rank search."""
import time
import itertools
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 3, 7]

# Deadline guard: kill at 150s wall-clock, keep a safety margin.
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def _tensor_index(n, m, p):
    """Yield (i,j,k) indices for the multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                yield (i, j, k)

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # The multiplication tensor T[i*m+j, j*p+k, k*n+i] = delta_{i,j,k} in schoolbook
    # Actually, the tensor is: for each (i,j,k), the entry at U-row (i*m+j), V-row (j*p+k), W-row (k*n+i) must be 1
    # The decomposition says: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if this is a valid multiplication term
    # Wait, the tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for the standard decomposition
    # More precisely, the multiplication tensor has a 1 at position (i*m+j, j*p+k, k*n+i) for each (i,j,k)
    
    # Check: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1
    # And for positions where the tensor is 0, the sum should equal 0
    
    # Build the full tensor check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row_u = i * m + j
                row_v = j * p + k
                row_w = k * n + i
                val = np.sum(U[row_u, :] * V[row_v, :] * W[row_w, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check that all other entries are zero
    # The tensor is nonzero only at (i*m+j, j*p+k, k*n+i) for valid (i,j,k)
    # Actually, we need to check that for any (a, b, c) where a is a row of U, b is a row of V, c is a row of W,
    # if (a,b,c) is not of the form (i*m+j, j*p+k, k*n+i), then sum_c U[a,c]*V[b,c]*W[c,c] = 0
    
    # Let's check all possible (a, b, c) tuples
    # a ranges over n*m, b over m*p, c over p*n
    # For each (a, b, c), determine if it corresponds to a valid (i,j,k)
    # a = i*m + j => i = a//m, j = a%m
    # b = j'*p + k => j' = b//p, k = b%p
    # c = k'*n + i' => k' = c//n, i' = c%n
    # It's valid if j == j' and i == i' and k == k'
    
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue  # This can't be a valid term, should be 0
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                if i_a != i_c or k_b != k_c:
                    # Should be zero
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
                else:
                    # Should be 1
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _try_random_search(n, m, p, rank, max_iters=5000, seed=42):
    """Try to find a decomposition of the given rank using random initialization + local search."""
    rng = np.random.RandomState(seed)
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    U0, V0, W0 = _schoolbook(n, m, p)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Random initialization
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Simple gradient-free local search: perturb and keep if error decreases
        # Compute error
        def compute_error(U, V, W):
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                        err += (val - 1.0) ** 2
            # Check zeros
            for a in range(n * m):
                i_a = a // m
                j_a = a % m
                for b in range(m * p):
                    j_b = b // p
                    k_b = b % p
                    if j_a != j_b:
                        continue
                    for c in range(p * n):
                        k_c = c // n
                        i_c = c % n
                        if i_a != i_c or k_b != k_c:
                            val = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                            err += val ** 2
            return err
        
        err = compute_error(U, V, W)
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_error < 1e-12:
                break
        
        # Perturbation
        if it % 10 == 0:
            # Larger step
            U += 0.5 * rng.randn(n * m, rank)
            V += 0.5 * rng.randn(m * p, rank)
            W += 0.5 * rng.randn(p * n, rank)
        else:
            U += 0.1 * rng.randn(n * m, rank)
            V += 0.1 * rng.randn(m * p, rank)
            W += 0.1 * rng.randn(p * n, rank)
    
    if best_error < 1e-12:
        # Snap to integers/half-integers
        best_U = np.round(best_U * 2) / 2
        best_V = np.round(best_V * 2) / 2
        best_W = np.round(best_W * 2) / 2
        if _verify(best_U, best_V, best_W, n, m, p):
            return best_U, best_V, best_W
    return None

def _try_lp_rank(n, m, p, rank):
    """
    Try to find a decomposition of the given rank using linear programming.
    For a fixed rank, the equations are linear in U, V, W if we fix one factor.
    We can fix W to have orthonormal columns (or some simple structure) and solve for U, V.
    Actually, this is a 3-linear problem. Let's use a different approach.
    
    Fix V to be a simple matrix (e.g., V = I_padded or random with full column rank).
    Then the equations become bilinear in U and W.
    
    Alternatively, we can use the fact that for rank r, we need:
    sum_c U[a,c] V[b,c] W[c',c] = T[a,b,c']
    
    This is equivalent to: for each c', U * (V * W[:, c']^T)^T ... no, this is 3-tensor.
    
    Let's try: fix W to be a random matrix with full column rank.
    Then for each c' (row of W), we have:
    sum_c U[a,c] V[b,c] W[c',c] = T[a,b,c']
    
    This is: (U * diag(W[c',:]) * V^T)[a,b] = T[a,b,c']
    
    So if we let X = U * diag(W[c',:]) and Y = V^T, then X * Y = T[:, :, c']
    
    This is a matrix factorization problem: X * Y = M_c' where X is (n*m) x r, Y is r x (m*p).
    
    If we fix W, we need to find U and V such that for all c':
    U * diag(W[c',:]) * V^T = T[:, :, c']
    
    Let D_c' = diag(W[c',:]). Then U * D_c' * V^T = T_c'.
    
    If W has full column rank, we can set V = W^{-1} (if W is square, but it's not).
    
    This is getting complicated. Let's use a simpler approach: try small ranks with integer/half-integer entries.
    """
    return None

def _try_integer_search(n, m, p, rank, max_attempts=10000, seed=42):
    """
    Search for decompositions with entries in {0, 1/2, 1, 3/2, 2} or similar small sets.
    This is a combinatorial search, so limit the scope.
    """
    if rank > 15:
        return None  # Too large for brute force
    
    rng = np.random.RandomState(seed)
    candidates = [0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2]
    
    # This is infeasible for rank > 5 or so. Just try a few random ones.
    for _ in range(min(max_attempts, 100)):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break
        U = rng.choice(candidates, size=(n*m, rank))
        V = rng.choice(candidates, size=(m*p, rank))
        W = rng.choice(candidates, size=(p*n, rank))
        if _verify(U, V, W, n, m, p):
            return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # Known lower bounds: for 3x3 * 3x7, the schoolbook is 63.
    # Let's try ranks from 62 down to some minimum.
    
    # First, try a few known constructions or random search for slightly lower ranks.
    # This is a hard problem, so we'll try a limited search.
    
    for rank in range(best_rank - 1, max(best_rank - 10, 30), -1):
        if time.time() > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Try random search
        result = _try_random_search(n, m, p, rank, max_iters=2000, seed=rank)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break  # Found a lower rank, stop
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
