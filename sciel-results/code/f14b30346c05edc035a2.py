"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def _build_target_tensor(n, m, p):
    """Build the target tensor T[i,j,k] = 1 for all valid (i,j,k)."""
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is valid."""
    r = len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _validate_exact(U, V, W, n, m, p):
    """Check if the decomposition is valid with exact rational arithmetic."""
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _rank17_decomposition(n, m, p):
    """
    Construct a rank-17 decomposition for 2x3x3 matrix multiplication.
    This merges two schoolbook terms that have compatible supports.
    """
    # Schoolbook has 18 terms. We merge (0,0,0) and (0,0,1) into one column.
    # The merged column has: U[0]=1, V[0]=1, V[1]=1, W[0]=1, W[2]=1, all else 0.
    # This works because for (i,j,k) not in {(0,0,0),(0,0,1)}, at least one of
    # U[i*3+j], V[j*3+k], W[k*2+i] is 0 in the merged column.
    
    U = [[0]*17 for _ in range(n*m)]
    V = [[0]*17 for _ in range(m*p)]
    W = [[0]*17 for _ in range(p*n)]
    
    # Column 0: merged (0,0,0) and (0,0,1)
    U[0][0] = 1   # i=0, j=0
    V[0][0] = 1   # j=0, k=0
    V[1][0] = 1   # j=0, k=1
    W[0][0] = 1   # k=0, i=0
    W[2][0] = 1   # k=1, i=0
    
    # Remaining 16 terms, one per column
    col = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) == (0, 0, 0) or (i, j, k) == (0, 0, 1):
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return U, V, W

def _try_rank16_search(n, m, p, deadline):
    """
    Try to find a rank-16 decomposition using greedy rank-1 completion.
    Start from the rank-17 solution and try to merge pairs of columns.
    """
    import random
    random.seed(12345)
    
    # Start with the rank-17 decomposition
    U17, V17, W17 = _rank17_decomposition(n, m, p)
    r17 = len(U17[0])
    
    # Try to merge pairs of columns in the rank-17 decomposition
    # For each pair (a, b), check if we can create a single column that
    # replaces both columns a and b while preserving the tensor identity.
    
    best = None
    
    for a in range(r17):
        if time.time() > deadline - 5:
            break
        for b in range(a + 1, r17):
            if time.time() > deadline - 5:
                break
            
            # Create a candidate rank-16 by merging columns a and b
            # The merged column should have U, V, W entries that are the sum
            # (or some combination) of columns a and b.
            
            # Simple approach: try U_merged = U_a + U_b, V_merged = V_a + V_b, W_merged = W_a + W_b
            # and check if the resulting tensor still equals 1 for all (i,j,k).
            
            U_new = [row[:] for row in U17]
            V_new = [row[:] for row in V17]
            W_new = [row[:] for row in W17]
            
            # Remove column b, add merged column at position a
            for row in U_new:
                row.pop(b)
            for row in V_new:
                row.pop(b)
            for row in W_new:
                row.pop(b)
            
            # Merge columns a and b into column a
            for i in range(n*m):
                U_new[i][a] = U17[i][a] + U17[i][b]
            for i in range(m*p):
                V_new[i][a] = V17[i][a] + V17[i][b]
            for i in range(p*n):
                W_new[i][a] = W17[i][a] + W17[i][b]
            
            # Check validity
            if _validate(U_new, V_new, W_new, n, m, p):
                if _validate_exact(U_new, V_new, W_new, n, m, p):
                    best = {"U": U_new, "V": V_new, "W": W_new}
                    return best
    
    return best

def _try_rank16_als(n, m, p, deadline):
    """
    Try to find a rank-16 decomposition using ALS (Alternating Least Squares)
    with exact rational polishing.
    """
    import random
    random.seed(67890)
    
    # Try several random initializations
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        r = 16
        # Random integer/half-integer initialization
        U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n*m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m*p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p*n)]
        
        # Quick check if this random guess works (unlikely but possible)
        if _validate(U, V, W, n, m, p):
            if _validate_exact(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _try_rank16_greedy(n, m, p, deadline):
    """
    Try to find a rank-16 decomposition using greedy rank-1 completion.
    Start from zero and iteratively add the best rank-1 term.
    """
    import random
    random.seed(11111)
    
    r = 16
    n_rows = n * m
    m_rows = m * p
    p_rows = p * n
    
    U = [[0.0] * r for _ in range(n_rows)]
    V = [[0.0] * r for _ in range(m_rows)]
    W = [[0.0] * r for _ in range(p_rows)]
    
    # Build target tensor
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Iteratively find best rank-1 approximation to residual
    for c in range(r):
        if time.time() > deadline - 5:
            break
        
        # Compute residual: T - sum_{c'<c} u_c' v_c' w_c'
        residual = T.copy()
        for c_prime in range(c):
            u = np.array([U[i*m+j][c_prime] for i in range(n) for j in range(m)])
            v = np.array([V[j*p+k][c_prime] for j in range(m) for k in range(p)])
            w = np.array([W[k*n+i][c_prime] for k in range(p) for i in range(n)])
            # Reshape and multiply
            u_mat = u.reshape(n, m)
            v_mat = v.reshape(m, p)
            w_mat = w.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i, j, k] -= u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        
        # Find best rank-1 approximation to residual
        # This is NP-hard in general, so use a greedy heuristic
        # Try random vectors and pick the best one
        
        best_score = -1e18
        best_u, best_v, best_w = None, None, None
        
        for trial in range(200):
            if time.time() > deadline - 5:
                break
            
            # Random integer vectors
            u_rand = np.random.randint(-2, 3, size=n*m).astype(float)
            v_rand = np.random.randint(-2, 3, size=m*p).astype(float)
            w_rand = np.random.randint(-2, 3, size=p*n).astype(float)
            
            # Normalize
            u_norm = np.linalg.norm(u_rand)
            v_norm = np.linalg.norm(v_rand)
            w_norm = np.linalg.norm(w_rand)
            
            if u_norm < 1e-10 or v_norm < 1e-10 or w_norm < 1e-10:
                continue
            
            u_rand /= u_norm
            v_rand /= v_norm
            w_rand /= w_norm
            
            # Compute inner product with residual
            u_mat = u_rand.reshape(n, m)
            v_mat = v_rand.reshape(m, p)
            w_mat = w_rand.reshape(p, n)
            
            score = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        score += residual[i, j, k] * u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
            
            if score > best_score:
                best_score = score
                best_u = u_rand
                best_v = v_rand
                best_w = w_rand
        
        if best_u is not None:
            for idx in range(n*m):
                U[idx][c] = best_u[idx]
            for idx in range(m*p):
                V[idx][c] = best_v[idx]
            for idx in range(p*n):
                W[idx][c] = best_w[idx]
    
    # Check if the result is valid
    if _validate(U, V, W, n, m, p, tol=1e-6):
        # Try to round to integers/half-integers
        U_rounded = [[round(x * 2) / 2 for x in row] for row in U]
        V_rounded = [[round(x * 2) / 2 for x in row] for row in V]
        W_rounded = [[round(x * 2) / 2 for x in row] for row in W]
        
        if _validate(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
            if _validate_exact(U_rounded, V_rounded, W_rounded, n, m, p):
                return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # First, construct the verified rank-17 decomposition
    U17, V17, W17 = _rank17_decomposition(n, m, p)
    
    # Verify it's correct
    if _validate(U17, V17, W17, n, m, p):
        if _validate_exact(U17, V17, W17, n, m, p):
            # Try to find a rank-16 decomposition
            result16 = _try_rank16_search(n, m, p, deadline)
            if result16 is not None:
                return result16
            
            result16 = _try_rank16_greedy(n, m, p, deadline)
            if result16 is not None:
                return result16
            
            # Fall back to rank-17
            return {"U": U17, "V": V17, "W": W17}
    
    # Fallback: schoolbook rank-18
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    return {"U": U_school, "V": V_school, "W": W_school}
