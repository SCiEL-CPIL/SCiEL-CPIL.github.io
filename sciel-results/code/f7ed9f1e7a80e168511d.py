import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def _build_factor(n, m, p, entries, rank):
    """entries: list of (i,j,k) triples; build U,V,W with given rank."""
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for t, (i, j, k) in enumerate(entries):
        if t >= rank:
            break
        U[i * m + j][t] = 1.0
        V[j * p + k][t] = 1.0
        W[k * n + i][t] = 1.0
    return U, V, W

def _build_bordered(n, m, p, rank):
    """Build bordered decomposition for n x m times m x p.
    
    The bordered method splits each dimension into core + border.
    For A (n x m), B (m x p):
    - Core: (n-1) x (m-1) x (p-1) using schoolbook
    - Border terms to handle the last row/column interactions
    
    Rank = (n-1)*(m-1)*(p-1) + n + m + p - 2
    """
    if rank < (n-1)*(m-1)*(p-1) + n + m + p - 2:
        return None, None, None
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    t = 0
    
    # Core: (n-1) x (m-1) x (p-1)
    for i in range(n-1):
        for j in range(m-1):
            for k in range(p-1):
                U[i*m + j][t] = 1.0
                V[j*p + k][t] = 1.0
                W[k*n + i][t] = 1.0
                t += 1
    
    # Border term 1: A[0..n-2, m-1] * B[m-1, 0..p-2]
    # This contributes to C[0..n-2, 0..p-2]
    for i in range(n-1):
        if t >= rank:
            return None, None, None
        U[i*m + (m-1)][t] = 1.0
        V[(m-1)*p + 0][t] = 1.0  # Need B[m-1, *]
        W[0*n + i][t] = 1.0
        t += 1
    
    # This is getting complex. Let me use a simpler but valid construction.
    # Actually, let's just use the schoolbook rank for now since constructing
    # bordered correctly is error-prone.
    
    return None, None, None

def _validate(U, V, W, n, m, p, rank):
    """Check that U,V,W exactly decompose the matrix mult tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Reconstruct tensor T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    T = np.zeros((n, m, p), dtype=float)
    for t in range(rank):
        u_col = U[:, t]
        v_col = V[:, t]
        w_col = W[:, t]
        # reshape
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        w_mat = w_col.reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k.
    expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T, expected, atol=1e-9)

def _build_bordered_simple(n, m, p):
    """Simple bordered decomposition.
    
    Uses the identity:
    AB = sum_{j} A[:,j] * B[j,:]
    
    But groups terms to reduce rank. For (n,m,p) = (3,4,6):
    
    Split j into j=0..2 (core) and j=3 (border).
    Core: sum_{j=0}^{2} A[:,j] * B[j,:] has rank 3*3*6 = 54 if done naively,
    but we can do better.
    
    Actually, let's use a different approach. The rank of matrix multiplication
    for (n,m,p) is at most n*m*p, but can be lower.
    
    For (3,4,6), let's try rank = 3*6 + 4*6 - 3 = 18 + 24 - 3 = 39
    This comes from: rank <= n*p + m*p - n
    
    Construction:
    For t = 0..n*p-1: handle A * B where we fix k and vary i
    For t = n*p..n*p+m*p-n-1: handle the remaining terms
    
    This is getting complicated. Let me just return the schoolbook for now.
    """
    return None, None, None

def _try_rank(rank, n, m, p, deadline, rng):
    """Attempt to find a valid decomposition of given rank. Returns (U,V,W) or None."""
    # Use bordered decomposition: split matrices into blocks and use Strassen-like
    # ideas on the bordered parts. For 3x4x6, we can try various block splits.
    
    # The bordered method: write A (n x m) and B (m x p) in block form.
    # If we can find a decomposition of a smaller problem, we can use it.
    
    # For 3x4x6, one approach: treat as (2+1) x (2+2) x (4+2) or similar
    # But more practically, let's try the "bordered" approach where we
    # separate one row/column and use a lower-rank decomposition for the core.
    
    # Actually, let's try a direct approach: assume U, V, W have a specific
    # sparse structure and solve for the entries.
    
    # For rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i,j,k: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    
    # This is a system of n*m*p = 72 equations in 3*r variables (per column t).
    # But the equations are bilinear in the variables of each t, coupled across t.
    
    # Let's try a greedy approach: build up the decomposition one term at a time.
    # Each term t contributes a rank-1 tensor U_t ⊗ V_t ⊗ W_t.
    # We want sum_t U_t ⊗ V_t ⊗ W_t = ones(n,m,p).
    
    # Start with zero tensor, greedily add rank-1 tensors to match remaining.
    # But this is hard to do exactly.
    
    # Alternative: use the fact that for small ranks, we can try integer/half-integer
    # entries and check if they work.
    
    # For now, let's try a few structured patterns that might give rank < 72.
    
    # Pattern 1: Use the "bordered" decomposition with a 2x2x2 core
    # If we can decompose 2x2x2 with rank 7 (Strassen), we can use that.
    # But 3x4x6 doesn't directly embed 2x2x2.
    
    # Let's try: assume the decomposition has the form where each t corresponds
    # to a (i,j,k) triple with some entries zeroed out.
    
    # For rank r, try all subsets of size r from the 72 basis tensors?
    # That's C(72, r) which is huge for r near 72.
    
    # Better: try structured patterns.
    
    # Pattern: Use rank = n*p + m*p - n = 3*6 + 4*6 - 3 = 18 + 24 - 3 = 39
    # This is the "bordered" rank for (n,m,p) when using a 1x1 border.
    
    # Actually, the bordered method gives rank <= (n-1)*(m-1)*(p-1) + n + m + p - 2
    # For 3,4,6: 2*3*5 + 3+4+6-2 = 30 + 11 = 41
    
    # Let's try to construct the bordered decomposition explicitly.
    
    # The bordered decomposition for multiplying (n x m) by (m x p):
    # Write A = [A_core | A_right; A_bottom | A_corner] where A_core is (n-1)x(m-1)
    # B = [B_top; B_bottom] where B_top is (m-1) x p
    # Then AB = A_core * B_top + A_right * B_bottom + A_bottom * B_top + A_corner * B_bottom
    # Wait, this isn't quite right. Let me think more carefully.
    
    # Standard bordered: 
    # A is n x m, B is m x p.
    # Split m = (m-1) + 1.
    # A = [A1 | a2] where A1 is n x (m-1), a2 is n x 1
    # B = [b1; B2] where b1 is (m-1) x p, B2 is 1 x p
    # Then AB = A1 * b1 + a2 * B2
    # This gives rank = n*(m-1) + n*1 + 1*p = n*(m-1) + n + p = n*m + p
    # For 3,4,6: 3*4 + 6 = 18
    
    # But we can do better by also splitting n or p.
    
    # Let's try the full bordered approach:
    # Split n = (n-1) + 1, m = (m-1) + 1, p = (p-1) + 1
    # A = [A11 | A12; A21 | A22] where A11 is (n-1)x(m-1)
    # B = [B11 B12; B21 B22] where B11 is (m-1)x(p-1)
    # AB = A11*B11 + A11*B12 + A12*B21 + A12*B22 + A21*B11 + A21*B12 + A22*B21 + A22*B22
    # But we can group these to reduce rank.
    
    # Actually, the standard bordered rank is:
    # rank <= (n-1)*(m-1)*(p-1) + n + m + p - 2
    
    # For 3,4,6: 2*3*5 + 3 + 4 + 6 - 2 = 30 + 11 = 41
    
    # Let's try to construct this. The idea is:
    # For the core (n-1)x(m-1)x(p-1) = 2x3x5 part, use schoolbook rank 30.
    # For the border terms, use additional terms.
    
    # The border terms are:
    # 1. A11 * B12: (n-1)x(m-1) * (m-1)x1 -> rank (n-1)*1 = n-1 = 2
    # 2. A12 * B21: 1x(m-1) * (m-1)x(p-1) -> rank 1*(p-1) = p-1 = 5
    # 3. A22 * B22: 1x1 * 1x1 -> rank 1
    # 4. A21 * B12: 1x(m-1) * (m-1)x1 -> rank 1
    # 5. A12 * B22: 1x1 * 1x(p-1) -> rank 1
    # 6. A21 * B11: 1x(m-1) * (m-1)x(p-1) -> rank p-1 = 5
    # 7. A11 * B22: (n-1)x(m-1) * 1x1 -> rank n-1 = 2
    
    # Wait, I'm overcomplicating this. Let me just try a simpler approach.
    
    # Try rank = n*m + m*p - n = 12 + 24 - 3 = 33
    # Or rank = n*p + m*p - n = 18 + 24 - 3 = 39
    
    # For now, let's just return None and rely on the schoolbook fallback.
    # A real improvement would require finding an actual lower-rank decomposition.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN
    rng = random.Random(42)
    
    # Best known: schoolbook
    rank = n * m * p
    U, V, W = _build_factor(n, m, p, [(i, j, k) for i in range(n) for j in range(m) for k in range(p)], rank)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try bordered decomposition
    bordered_rank = (n-1)*(m-1)*(p-1) + n + m + p - 2
    if bordered_rank < best_rank:
        U_b, V_b, W_b = _build_bordered_simple(n, m, p)
        if U_b is not None:
            if _validate(U_b, V_b, W_b, n, m, p, len(U_b[0])):
                best_result = {"U": U_b, "V": V_b, "W": W_b}
                best_rank = len(U_b[0])
    
    # Try to find lower rank decompositions using bordered method
    # The bordered decomposition for (n,m,p) has rank at most:
    # (n-1)*(m-1)*(p-1) + n + m + p - 2
    
    bordered_rank = (n-1)*(m-1)*(p-1) + n + m + p - 2
    if bordered_rank < best_rank:
        # Try to construct the bordered decomposition
        U_b, V_b, W_b = _build_bordered(n, m, p, bordered_rank)
        if U_b is not None and _validate(U_b, V_b, W_b, n, m, p, bordered_rank):
            best_result = {"U": U_b, "V": V_b, "W": W_b}
            best_rank = bordered_rank
    
    return best_result

def _build_bordered(n, m, p, rank):
    """Build bordered decomposition for n x m times m x p."""
    # Core: (n-1) x (m-1) x (p-1) with schoolbook rank
    core_rank = (n-1)*(m-1)*(p-1)
    
    # Border terms:
    # 1. (n-1) terms for A11 * B12 type
    # 2. (p-1) terms for A12 * B21 type  
    # 3. 1 term for A22 * B22
    # 4. Additional terms for other border interactions
    
    # Total: core_rank + (n-1) + (p-1) + 1 + ... 
    # This should equal (n-1)*(m-1)*(p-1) + n + m + p - 2
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    t = 0
    
    # Core part: (n-1) x (m-1) x (p-1)
    for i in range(n-1):
        for j in range(m-1):
            for k in range(p-1):
                if t >= rank:
                    return None, None, None
                U[i*m + j][t] = 1.0
                V[j*p + k][t] = 1.0
                W[k*n + i][t] = 1.0
                t += 1
    
    # Border: row n-1 of A with columns 0..m-2 of B
    for j in range(m-1):
        if t >= rank:
            return None, None, None
        U[(n-1)*m + j][t] = 1.0
        V[j*p + (p-1)][t] = 1.0  # This is wrong, need to think more carefully
        W[(p-1)*n + (n-1)][t] = 1.0
        t += 1
    
    # This approach is getting complicated. Let me just return None for now.
    return None, None, None
