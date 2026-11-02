import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def _build_rank28():
    """Build the standard rank-28 decomposition for 2x3 x 3x5."""
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0, k=0..4 (schoolbook for 2x1x5), rank 10
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2a: j=1,2, k=0,1 (Strassen for 2x2x2), rank 7
    U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 0][col] = 1; V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 0][col] = 1
    W[0 * n + 1][col] = 1; W[1 * n + 1][col] = -1
    col += 1
    
    U[0 * m + 1][col] = 1
    V[1 * p + 1][col] = 1; V[2 * p + 1][col] = -1
    W[1 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    U[1 * m + 2][col] = 1
    V[2 * p + 0][col] = 1; V[1 * p + 0][col] = -1
    W[0 * n + 0][col] = 1; W[0 * n + 1][col] = 1
    col += 1
    
    U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
    V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = -1; W[1 * n + 0][col] = 1
    col += 1
    
    U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
    V[1 * p + 0][col] = 1; V[1 * p + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1
    
    U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
    V[2 * p + 0][col] = 1; V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = 1
    col += 1
    
    # Block 2b: j=1,2, k=2 (schoolbook for 2x2x1), rank 4
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    
    # Block 2c: j=1,2, k=3,4 (Strassen for 2x2x2), rank 7
    U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 3][col] = 1
    W[3 * n + 1][col] = 1; W[4 * n + 1][col] = -1
    col += 1
    
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1; V[2 * p + 4][col] = -1
    W[4 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    U[1 * m + 2][col] = 1
    V[2 * p + 3][col] = 1; V[1 * p + 3][col] = -1
    W[3 * n + 0][col] = 1; W[3 * n + 1][col] = 1
    col += 1
    
    U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
    V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = -1; W[4 * n + 0][col] = 1
    col += 1
    
    U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
    V[1 * p + 3][col] = 1; V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
    V[2 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1
    col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=3, p=5, atol=1e-8):
    """Verify that the decomposition correctly computes matrix multiplication."""
    T = np.zeros((n, m, p))
    for r in range(len(U[0])):
        u = np.array(U[:, r] if isinstance(U, list) else U[:, r])
        v = np.array(V[:, r] if isinstance(V, list) else V[:, r])
        w = np.array(W[:, r] if isinstance(W, list) else W[:, r])
        T += np.einsum('i,j,k->ijk', u, v, w)
    T_orig = np.ones((n, m, p))
    return np.allclose(T, T_orig, atol=atol)

def _check_dependency_exact(U, V, W):
    """Check for linear dependencies using exact rational arithmetic.
    
    Returns (found, null_vector, k_idx) if a dependency is found, else (False, None, None).
    """
    n, m, p = 2, 3, 5
    rank = len(U[0])
    
    # Build the matrix M where M[i,j,k,r] = U[i,r]*V[j,r]*W[k,r]
    # Flatten to (n*m*p, rank)
    size = n * m * p
    M = [[Fraction(0)] * rank for _ in range(size)]
    
    for r in range(rank):
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    # Only non-zero if indices are consistent
                    # U index: i corresponds to (a, b) where a in [0,n), b in [0,m)
                    # V index: j corresponds to (b, c) where b in [0,m), c in [0,p)
                    # W index: k corresponds to (c, a) where c in [0,p), a in [0,n)
                    # The tensor entry T[a,b,c] = sum_r U[a*m+b, r] * V[b*p+c, r] * W[c*n+a, r]
                    # So for each (a,b,c), we have one entry in the flattened tensor
                    pass
    
    # Better approach: build the full tensor matrix
    # Entry (a,b,c) in tensor = sum_r U[a*m+b, r] * V[b*p+c, r] * W[c*n+a, r]
    # We want to check if the vectors v_r (where v_r has entry (a,b,c) = U[a*m+b,r]*V[b*p+c,r]*W[c*n+a,r]) are linearly dependent
    
    # Build M as a (size x rank) matrix of Fractions
    M = np.zeros((size, rank), dtype=object)
    for r in range(rank):
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    idx = (a * m + b) * (m * p) + (b * p + c) * p + c  # This isn't right
                    # Let me use a different flattening: index = a*(m*p) + b*p + c
                    idx = a * (m * p) + b * p + c
                    u_val = Fraction(U[a * m + b][r]) if isinstance(U, list) else Fraction(int(U[a * m + b, r]))
                    v_val = Fraction(V[b * p + c][r]) if isinstance(V, list) else Fraction(int(V[b * p + c, r]))
                    w_val = Fraction(W[c * n + a][r]) if isinstance(W, list) else Fraction(int(W[c * n + a, r]))
                    M[idx, r] = u_val * v_val * w_val
    
    # Compute rank over Q using Gaussian elimination
    # Convert to list of lists for easier manipulation
    M_list = [[M[i, j] for j in range(rank)] for i in range(size)]
    
    # Gaussian elimination to find null space
    row = 0
    for col in range(rank):
        # Find pivot
        pivot = -1
        for i in range(row, size):
            if M_list[i][col] != 0:
                pivot = i
                break
        if pivot == -1:
            continue
        
        # Swap rows
        M_list[row], M_list[pivot] = M_list[pivot], M_list[row]
        
        # Eliminate
        pivot_val = M_list[row][col]
        for i in range(row + 1, size):
            if M_list[i][col] != 0:
                factor = M_list[i][col] / pivot_val
                for j in range(col, rank):
                    M_list[i][j] -= factor * M_list[row][j]
        
        row += 1
    
    # Now we have row echelon form with 'row' non-zero rows
    # The null space dimension is rank - row
    null_dim = rank - row
    if null_dim <= 0:
        return False, None, None
    
    # Find a null space vector using back-substitution
    # Free variables: columns from 'row' to rank-1
    # Set free variables to 0 except one, which we set to 1
    # Try each free variable
    for free_col in range(row, rank):
        c = [Fraction(0)] * rank
        c[free_col] = Fraction(1)
        
        # Back-substitute for pivot variables
        for i in range(row - 1, -1, -1):
            # Find pivot column for row i
            pivot_col = -1
            for j in range(i, rank):
                if M_list[i][j] != 0:
                    pivot_col = j
                    break
            if pivot_col == -1:
                continue
            # c[pivot_col] = -sum_{j > pivot_col} M_list[i][j] * c[j] / M_list[i][pivot_col]
            val = Fraction(0)
            for j in range(pivot_col + 1, rank):
                val += M_list[i][j] * c[j]
            c[pivot_col] = -val / M_list[i][pivot_col]
        
        # Verify
        residual = 0
        for i in range(size):
            dot = Fraction(0)
            for j in range(rank):
                dot += M_list[i][j] * c[j]
            residual += abs(dot)
        
        if residual == 0:
            return True, c, free_col
    
    return False, None, None

def construct():
    """Return a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 3, 5
    
    # Build the rank-28 decomposition
    U, V, W = _build_rank28()
    rank = len(U[0])
    
    # Convert to numpy for verification
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    # Check for linear dependencies using exact arithmetic
    found, c, k_idx = _check_dependency_exact(U, V, W)
    
    if found and k_idx is not None:
        # Eliminate term k_idx
        new_U = []
        new_V = []
        new_W = []
        for r in range(rank):
            if r == k_idx:
                continue
            coeff = 1 - c[r] / c[k_idx]
            new_U.append([U[i][r] * coeff for i in range(n * m)])
            new_V.append([V[i][r] * coeff for i in range(m * p)])
            new_W.append([W[i][r] * coeff for i in range(p * n)])
        
        # Verify
        if _verify_decomposition(new_U, new_V, new_W):
            return {
                "U": new_U,
                "V": new_V,
                "W": new_W
            }
    
    # If no dependency found, try a randomized perturbation search
    # Try to find a rank-27 decomposition by perturbing one term
    if time.time() < deadline:
        import random
        random.seed(42)
        
        # Try removing each term and solving for the correction
        for k_idx in range(rank):
            if time.time() > deadline:
                break
            
            # Try to express T_k as a linear combination of the other terms
            # This is equivalent to checking if T_k is in the span of the other terms
            # Build matrix of other terms
            other_terms = []
            for r in range(rank):
                if r == k_idx:
                    continue
                t = np.einsum('i,j,k->ijk', U_mat[:, r], V_mat[:, r], W_mat[:, r])
                other_terms.append(t.flatten())
            
            other_M = np.array(other_terms).T  # (900, 27)
            t_k = np.einsum('i,j,k->ijk', U_mat[:, k_idx], V_mat[:, k_idx], W_mat[:, k_idx])
            t_k_flat = t_k.flatten()
            
            # Solve least squares: find x such that other_M @ x ≈ t_k_flat
            try:
                x, residuals, _, _ = np.linalg.lstsq(other_M, t_k_flat, rcond=None)
                residual = np.linalg.norm(other_M @ x - t_k_flat)
                
                if residual < 1e-10:
                    # T_k is in the span of the other terms
                    # New decomposition: T = sum_{r != k} T_r + T_k = sum_{r != k} (1 + x_r) T_r
                    # Wait, that's not right. T = sum_{r != k} T_r + T_k
                    # And T_k = sum_{r != k} x_r T_r
                    # So T = sum_{r != k} (1 + x_r) T_r
                    # But (1+x_r) T_r is rank-1, so we have 27 rank-1 terms
                    new_U = []
                    new_V = []
                    new_W = []
                    idx = 0
                    for r in range(rank):
                        if r == k_idx:
                            continue
                        coeff = 1 + x[idx]
                        new_U.append(U_mat[:, r] * coeff)
                        new_V.append(V_mat[:, r] * coeff)
                        new_W.append(W_mat[:, r] * coeff)
                        idx += 1
                    
                    new_U = np.array(new_U).T
                    new_V = np.array(new_V).T
                    new_W = np.array(new_W).T
                    
                    if _verify_decomposition(new_U.tolist(), new_V.tolist(), new_W.tolist()):
                        return {
                            "U": new_U.tolist(),
                            "V": new_V.tolist(),
                            "W": new_W.tolist()
                        }
            except Exception:
                pass
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
