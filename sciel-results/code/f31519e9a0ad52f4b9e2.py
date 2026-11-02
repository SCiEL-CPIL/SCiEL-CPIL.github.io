import numpy as np
import itertools

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Known lower bound for matrix multiplication rank
    # For 3x4 * 4x7, the rank is at least max(n*m, m*p, n*p) = max(12, 28, 21) = 28
    # Actually, a tighter bound is max(n*m, m*p, n*p) but often higher.
    # The trivial rank is n*m*p = 84.
    # We will search from a reasonable lower bound up to the schoolbook rank.
    
    # Let's try ranks from 28 to 84.
    # We will use a randomized approach:
    # 1. Initialize U, V, W randomly with rank r.
    # 2. Use an iterative method to minimize the error of the tensor decomposition.
    #    The error is || T - sum_c u_c v_c w_c ||_F^2
    # 3. If the error is small enough, try to snap to integers/half-integers.
    
    # However, direct minimization is hard.
    # Let's use the "alternating least squares" style approach from the original code
    # but with a better search strategy.
    
    # The original code fixed U, V and solved for W.
    # We can cycle: fix U, V -> solve W; fix V, W -> solve U; fix U, W -> solve V.
    
    # Let's implement a function to check if a set of matrices U, V, W (float)
    # gives an exact decomposition (within tolerance) and if they can be rounded.
    
    def check_exact(U, V, W, tol=1e-10):
        n, m, p = DIMENSIONS
        # Compute the tensor T from U, V, W
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We want T[i, j, k] = 1 if (i, j, k) corresponds to the multiplication tensor?
        # Wait, the task says "exactly decompose the matrix multiplication tensor".
        # The matrix multiplication tensor M has entries M[i, j, k] = 1 if the output element (i, k)
        # is computed from input (i, j) and (j, k).
        # Actually, the standard matrix multiplication tensor T such that
        # C = A B  <=>  C_{ik} = sum_j A_{ij} B_{jk}
        # The tensor T has indices (i, j, k) and T[i, j, k] = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
        # Wait, the indices in the task description for W are W[k*n+i, r].
        # This suggests the output index is (k, i) or (i, k)?
        # "output factor follows the official trace convention W[k*n+i,r]"
        # Usually, the multiplication tensor is defined such that
        # sum_c U[i, j, c] V[j, k, c] W[k, i, c] = 1
        # This implies that the product is C_{ki} = sum_j A_{ij} B_{jk}.
        # So the output matrix is C with rows k and columns i.
        # This is consistent with W having index k*n+i.
        
        # So the condition is:
        # For all i in 0..n-1, j in 0..m-1, k in 0..p-1:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        
        # Let's verify this interpretation.
        # If A is n x m, B is m x p, then C = A B is n x p.
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor decomposition usually maps (A, B) -> C.
        # If we view A, B, C as vectors, the multiplication is a trilinear form.
        # The standard decomposition is sum_c u_c (outer) v_c (outer) w_c (outer) = T.
        # T[i, j, k] = 1 if the trilinear form picks A_{ij} B_{jk} C_{ki}.
        # Yes, this matches the condition sum_c ... = 1 for all i, j, k.
        
        # So we just need to check if the computed tensor is all 1s.
        
        # Efficient check using vectorization
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        # This is a bit tricky to vectorize directly as a single einsum because of the index permutations.
        # Let's do it step by step.
        # First, compute M[i, j, k] = sum_c U[i, j, c] V[j, k, c]
        # M[i, j, k] = sum_c U[i, j, c] V[j, k, c]
        # This is like a "partial contraction" of U and V.
        
        # We can use einsum:
        # U: (n, m, r)
        # V: (m, p, r)
        # We want M[i, j, k] = sum_r U[i, j, r] V[j, k, r]
        # Note that V's first index is j, which matches U's second index j.
        # So M = np.einsum('ijr,jkr->ijk', U_reshaped, V_reshaped)
        
        M = np.einsum('ijr,jkr->ijk', U_reshaped, V_reshaped)
        
        # Now we want T[i, j, k] = sum_r M[i, j, k] W[k, i, r] ? No.
        # The original sum is sum_c U[i,j,c] V[j,k,c] W[k,i,c].
        # Let X[i, j, k] = U[i,j,c] V[j,k,c]. This is not a matrix product, it's an element-wise product summed over c.
        # So X[i, j, k] = sum_c U[i,j,c] V[j,k,c].
        # Then T[i, j, k] = sum_c ... wait, the sum is over c for all three terms.
        # It is NOT (sum_c U V) * W. It is sum_c (U V W).
        # So we cannot separate it into M * W.
        
        # We must compute the full trilinear form.
        # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        
        # We can do this by iterating over c?
        # Or using einsum with all three tensors.
        # U: (n, m, r)
        # V: (m, p, r)
        # W: (p, n, r)
        # We want T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        
        # This is a valid einsum:
        # 'ijr,jkr,kir->ijk'
        
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        except Exception:
            return False, None
            
        # Check if T is all 1s
        if np.allclose(T, 1.0, atol=tol):
            return True, T
        else:
            return False, T

    # Search for a valid decomposition
    # We will try ranks from a lower bound to the schoolbook rank.
    # Lower bound: max(n*m, m*p, n*p) = 28.
    # Upper bound: n*m*p = 84.
    
    # To speed up, we can try to find a solution with rank r by:
    # 1. Random initialization.
    # 2. Alternating least squares (ALS) to minimize ||T - 1||_F^2.
    # 3. Check for exactness and integrality.
    
    # Since the problem is small, we can afford a few iterations.
    
    # Let's define the ALS step.
    # Given V, W, solve for U.
    # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    # For fixed i, j, U[i, j, :] is the unknown vector of size r.
    # The equation is:
    # sum_c U[i, j, c] * (V[j, k, c] W[k, i, c]) = 1 for all k.
    # Let B[i, j, k] = V[j, k, :] * W[k, i, :] (element-wise product, size r).
    # Then we have a linear system for each (i, j):
    # sum_k B[i, j, k, :]^T U[i, j, :] = 1_k ? No.
    # The equation is: for each k, sum_c U[i, j, c] * B[i, j, k, c] = 1.
    # This is a linear system A * u = b where:
    # A is a matrix of size p x r.
    # A[k, c] = B[i, j, k, c] = V[j, k, c] * W[k, i, c].
    # b is a vector of size p, all ones.
    # u is U[i, j, :].
    
    # Similarly for V and W.
    
    def solve_for_U(V, W, r):
        n, m, p = DIMENSIONS
        U = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                # Build A (p x r)
                A = np.zeros((p, r))
                for k in range(p):
                    for c in range(r):
                        A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                b = np.ones(p)
                try:
                    u, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = u
                except:
                    return None
        return U

    def solve_for_V(U, W, r):
        n, m, p = DIMENSIONS
        V = np.zeros((m * p, r))
        for j in range(m):
            for k in range(p):
                # Equation: sum_c U[i, j, c] V[j, k, c] W[k, i, c] = 1 for all i.
                # Unknown: V[j, k, :]
                # A is n x r.
                # A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                A = np.zeros((n, r))
                for i in range(n):
                    for c in range(r):
                        A[i, c] = U[i * m + j, c] * W[k * n + i, c]
                b = np.ones(n)
                try:
                    v, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = v
                except:
                    return None
        return V

    def solve_for_W(U, V, r):
        n, m, p = DIMENSIONS
        W = np.zeros((p * n, r))
        for k in range(p):
            for i in range(n):
                # Equation: sum_c U[i, j, c] V[j, k, c] W[k, i, c] = 1 for all j.
                # Unknown: W[k, i, :]
                # A is m x r.
                # A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                A = np.zeros((m, r))
                for j in range(m):
                    for c in range(r):
                        A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                b = np.ones(m)
                try:
                    w, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = w
                except:
                    return None
        return W

    # Search loop
    # We will try ranks from 28 to 84.
    # For each rank, we will try a few random initializations and run ALS for a few iterations.
    
    # To save time, we will start from a higher rank and go down?
    # Or start from the lower bound and go up?
    # The lower bound is 28. It's unlikely we'll find a solution at 28.
    # The schoolbook is 84.
    # Let's try to find a solution at rank 84 first? No, that's the baseline.
    # We want to improve on 84.
    # So we should try ranks 83, 82, ...
    
    # However, the ALS method might converge to a local minimum that is not exact.
    # And rounding to integers/half-integers is a strong constraint.
    
    # Given the 120s limit, and the complexity of finding an exact integer solution,
    # it is very likely that the optimal rank is close to 84 or that no better solution exists easily.
    # In fact, for small matrices, the rank is often equal to the schoolbook rank.
    # For 3x4 * 4x7, the rank is known to be 84?
    # Let's check if there are known results.
    # The rank of matrix multiplication tensor M_{n,m,p} is at least max(nm, mp, np).
    # For 3x4x7, max(12, 28, 21) = 28.
    # But the actual rank is often much higher.
    # For 2x2x2, rank is 7 (vs schoolbook 8).
    # For 3x3x3, rank is 23 (vs schoolbook 27).
    # For 3x4x4, rank is ?
    
    # If we cannot find a better rank quickly, we should return the schoolbook solution.
    # But we must ensure the schoolbook solution is correct.
    
    # Let's try a few ranks near 84 with a very short time limit.
    
    import time
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for overhead
    
    # Try ranks 83 down to 80
    for r in range(83, 79, -1):
        if time.time() - start_time > time_limit:
            break
            
        for trial in range(10):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(42 + trial * 100 + r)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Run ALS for a few iterations
            for it in range(20):
                if time.time() - start_time > time_limit:
                    break
                    
                U_new = solve_for_U(V, W, r)
                if U_new is None: continue
                U = U_new
                
                V_new = solve_for_V(U, W, r)
                if V_new is None: continue
                V = V_new
                
                W_new = solve_for_W(U, V, r)
                if W_new is None: continue
                W = W_new
                
                # Check for convergence
                exact, T = check_exact(U, V, W)
                if exact:
                    # Try to round
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    exact_rounded, _ = check_exact(U_int, V_int, W_int)
                    if exact_rounded:
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
    
    # Fallback to schoolbook
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
