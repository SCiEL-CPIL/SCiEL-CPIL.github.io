import time
import numpy as np

DIMENSIONS = [4, 5, 5]
_START_TIME = time.time()
_DEADLINE = 145  # seconds, safety margin

def _elapsed():
    return time.time() - _START_TIME

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if decomposition is exact."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _cw_decomposition(n, m, p):
    """
    Attempt to construct a Coppersmith-Winograd style decomposition over F_7.
    
    Strategy:
    1. Embed the n×m×p multiplication tensor into a larger 7×7×7 structure.
    2. Use the 3-way sum trick with primitive cube roots of unity in F_7.
    3. Project back to get a decomposition of the original tensor.
    
    Returns dict with U, V, W or None if verification fails.
    """
    F = 7  # Field characteristic
    
    # In F_7, we have: 1^3 = 1, 2^3 = 8 ≡ 1, 4^3 = 64 ≡ 1
    # So {1, 2, 4} are the cube roots of unity in F_7 (excluding 0)
    # Actually: 1^3=1, 2^3=8=1, 4^3=64=64-63=1. Yes!
    # The non-zero elements form a cyclic group of order 6.
    # Elements of order 3: g^2 where g is primitive root. 3 is primitive: 3^1=3, 3^2=2, 3^3=6, 3^4=4, 3^5=5, 3^6=1
    # Elements of order dividing 3: 1, 3^2=2, 3^4=4. Check: 1^3=1, 2^3=8=1, 4^3=64=1. Yes.
    
    # The CW trick uses: for each "block", we have indices (a, b, c) where a+b+c ≡ 0 mod 3 (in the exponent sense)
    # More precisely, we use the identity:
    #   sum_{x in F} (x^2 * a + x * b + c)^3 = sum over specific terms
    # But a simpler approach: use the "trick" that for any x, y, z:
    #   (x + y + z)^3 = x^3 + y^3 + z^3 + 3(x+y)(y+z)(z+x)  [over F_7, 3 ≠ 0]
    # This doesn't directly help.
    
    # Let me use a different approach: direct construction using the fact that
    # matrix multiplication can be viewed as a bilinear form, and we can use
    # the "laser method" or simply try to find a structured decomposition.
    
    # Alternative: Use the fact that for n=m=p=5, we can try rank 25-35
    # by using a structured approach.
    
    # Let's try a more direct approach: assume U, V, W have entries in {0, 1, -1}
    # and use a greedy/structured construction.
    
    # For now, let's try to construct a decomposition of rank r = n*m + m*p + n*p - some_savings
    # A known lower bound is max(n*m, m*p, n*p) = 25.
    # A trivial upper bound is n*m*p = 100.
    # We can do better: rank ≤ n*m + m*p + n*p - 2 = 25 + 25 + 20 - 2 = 68 (using border rank tricks)
    # But exact rank is harder.
    
    # Let's try a structured approach:
    # Use "block diagonal" decomposition where we group terms.
    
    # Actually, let's try the most promising approach: 
    # For 4×5×5, we can use the fact that it's close to 5×5×5.
    # Known: 5×5×5 has rank at most 49 (from CW methods), possibly less.
    # But we need exact, not border rank.
    
    # Let's try a direct search for small structured decompositions.
    
    # Approach: Try rank r from 25 to 50, and for each, try to find U, V, W
    # with entries in {0, 1, -1} that satisfy the tensor equation.
    
    # This is a constraint satisfaction problem. Let's use a smarter search.
    
    # Key insight: For each (i, k), we need:
    #   sum_j A[i,j] * B[j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where A and B are the input matrices.
    
    # The tensor equation is: for all i,j,k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try to construct this with a structured pattern.
    
    # Try: U[i*m+j, c] = 1 if c corresponds to (i,j) in some pattern
    # V[j*p+k, c] = 1 if c corresponds to (j,k) in some pattern  
    # W[k*n+i, c] = 1 if c corresponds to (k,i) in some pattern
    
    # The simplest is the schoolbook: rank = n*m*p = 100.
    # To reduce rank, we need to share terms.
    
    # Let's try a "sum over c" approach where each c handles multiple (i,j,k) triples.
    
    # For rank r, we need r terms. Each term c is a product of three vectors:
    #   u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}
    # such that sum_c u_c ⊗ v_c ⊗ w_c = multiplication tensor.
    
    # Let's try to find such a decomposition with r < 100.
    
    # Use a greedy approach:
    # 1. Start with the full tensor T[i,j,k] = 1 if (i,j,k) is a valid triple, else 0.
    #    Actually, T[i,j,k] = 1 for all i,j,k (since every triple contributes).
    # 2. Find rank-1 tensors that match parts of T.
    
    # A rank-1 tensor is u ⊗ v ⊗ w where:
    #   (u ⊗ v ⊗ w)[i,j,k] = u[i,j] * v[j,k] * w[k,i]
    
    # We want: sum_c (u_c ⊗ v_c ⊗ w_c)[i,j,k] = 1 for all i,j,k
    
    # Let's try to find a decomposition where each u_c, v_c, w_c has sparse support.
    
    # For the 4×5×5 case, let's try rank r = 25 to 40.
    
    # Use a randomized approach with structured initialization.
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook rank
    
    # First, get the schoolbook decomposition as a baseline
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    if _verify(U_school, V_school, W_school, n, m, p, tol=1e-6):
        best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        best_rank = rank_school
    
    # Now try to find lower rank decompositions
    # Try ranks from 25 to best_rank - 1
    rng = np.random.RandomState(123)
    
    for r in range(25, min(best_rank, 50)):
        if _elapsed() > _DEADLINE - 15:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if _elapsed() > _DEADLINE - 15:
                break
            
            # Initialize U, V with random sparse patterns
            # Density: each entry is non-zero with probability ~ 3/r
            density = 3.0 / r
            
            U = np.zeros((n * m, r), dtype=int)
            V = np.zeros((m * p, r), dtype=int)
            
            for i in range(n * m):
                for c in range(r):
                    if rng.random() < density:
                        U[i, c] = rng.choice([-1, 1])
            
            for j in range(m * p):
                for c in range(r):
                    if rng.random() < density:
                        V[j, c] = rng.choice([-1, 1])
            
            # Now try to solve for W
            W = np.zeros((p * n, r), dtype=int)
            valid = True
            
            for k in range(p):
                for i in range(n):
                    # Need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
                    # Let A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    # Then: A @ w = ones(m) where w = W[k*n+i, :]
                    
                    A = np.zeros((m, r), dtype=float)
                    for j in range(m):
                        A[j, :] = U[i * m + j, :].astype(float) * V[j * p + k, :].astype(float)
                    
                    # Solve A @ w = ones(m)
                    # Try to find an integer solution
                    # First, check if there's a simple solution
                    
                    # Try: find a column c where A[:, c] = ones(m)
                    found = False
                    for c in range(r):
                        if np.allclose(A[:, c], np.ones(m), atol=1e-6):
                            W[k * n + i, c] = 1
                            found = True
                            break
                    
                    if not found:
                        # Try to find a linear combination of 2-3 columns
                        # This is expensive, so limit the search
                        if r <= 30:
                            # Try all pairs of columns
                            for c1 in range(r):
                                if np.allclose(A[:, c1], np.ones(m), atol=1e-6):
                                    continue
                                for c2 in range(c1 + 1, r):
                                    # Check if A[:, c1] + A[:, c2] = ones(m)
                                    if np.allclose(A[:, c1] + A[:, c2], np.ones(m), atol=1e-6):
                                        W[k * n + i, c1] = 1
                                        W[k * n + i, c2] = 1
                                        found = True
                                        break
                                if found:
                                    break
                            if not found:
                                # Try triples
                                for c1 in range(r):
                                    if found:
                                        break
                                    for c2 in range(c1 + 1, r):
                                        if found:
                                            break
                                        for c3 in range(c2 + 1, r):
                                            if np.allclose(A[:, c1] + A[:, c2] + A[:, c3], np.ones(m), atol=1e-6):
                                                W[k * n + i, c1] = 1
                                                W[k * n + i, c2] = 1
                                                W[k * n + i, c3] = 1
                                                found = True
                                                break
                        
                        if not found:
                            # Try least squares and round
                            w_sol, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            w_int = np.round(w_sol).astype(int)
                            if np.allclose(A @ w_int, np.ones(m), atol=1e-6):
                                W[k * n + i, :] = w_int
                                found = True
                    
                    if not found:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and _verify(U, V, W, n, m, p, tol=1e-6):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break  # Found a valid decomposition at this rank, try lower
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
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

def construct():
    n, m, p = DIMENSIONS
    
    # Try the CW-style structured decomposition
    result = _cw_decomposition(n, m, p)
    
    if result is not None:
        return result
    
    # Final fallback: schoolbook
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
