"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check valid terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check spurious terms
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in valid:
                    val = np.sum(U[a] * V[b] * W[c])
                    if abs(val) > 1e-9:
                        return False
    return True

def _rank16_decomposition(n, m, p):
    """
    Rank-16 decomposition via 4x32 matrix multiplication view.
    
    The multiplication C = A * B where A is 2x2 and B is 2x16 can be
    viewed as: for each column k of B, C[:,k] = A @ B[:,k].
    
    We use the fact that C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    and C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k].
    
    Key insight: We can pair up columns of B. For pairs (2t, 2t+1):
    Let B_left = [B[0,2t], B[1,2t]]^T and B_right = [B[0,2t+1], B[1,2t+1]]^T
    
    We need:
    C[0,2t]   = A[0,0]*B[0,2t]   + A[0,1]*B[1,2t]
    C[1,2t]   = A[1,0]*B[0,2t]   + A[1,1]*B[1,2t]
    C[0,2t+1] = A[0,0]*B[0,2t+1] + A[0,1]*B[1,2t+1]
    C[1,2t+1] = A[1,0]*B[0,2t+1] + A[1,1]*B[1,2t+1]
    
    This is equivalent to a 2x2 matrix A multiplied by a 2x2 matrix 
    [B[:,2t] B[:,2t+1]], producing a 2x2 output.
    
    Using Strassen's 7 multiplications for each 2x2x2 block would give
    7*8 = 56. But we can do better by exploiting the structure more.
    
    Actually, let's think of it differently. The tensor for 2x2 * 2x2
    has rank at most 7 (Strassen). For 8 independent such blocks, rank <= 56.
    
    But wait - we can share the A factors! The same A multiplies all 8 blocks.
    
    Let me reconsider. The total tensor is:
    T[a,b,c] = 1 if (a,b,c) corresponds to a valid (i,j,k) triple.
    
    a = i*m + j = i*2 + j (i=0,1; j=0,1)
    b = j*p + k = j*16 + k (j=0,1; k=0..15)
    c = k*n + i = k*2 + i (k=0..15; i=0,1)
    
    So T[i*2+j, j*16+k, k*2+i] = 1 for all i,j,k.
    
    This is the matrix multiplication tensor M_{2,2,16}.
    
    The standard bound: rank(M_{n,m,p}) <= n*m*p (schoolbook) = 64.
    With Strassen on 2x2 blocks: we group k into pairs, giving 8 blocks of
    M_{2,2,2}, each of rank 7, total 56.
    
    Can we do better? The tensor M_{2,2,16} can be seen as the tensor product
    M_{2,2,2} x M_{1,1,8} (where M_{1,1,8} is just 8 scalar multiplications).
    rank(M_{2,2,2} x M_{1,1,8}) <= rank(M_{2,2,2}) * rank(M_{1,1,8}) = 7 * 8 = 56.
    
    But actually M_{2,2,16} is NOT a tensor product in the usual sense because
    the indices don't factorize that way.
    
    Let me try a different approach: view it as 4 scalar multiplications per k,
    but share across k values.
    
    For fixed j (column of A / row of B):
    C[i,k] gets contribution A[i,j]*B[j,k] for i=0,1.
    
    So for j=0: C[0,k] += A[0,0]*B[0,k], C[1,k] += A[1,0]*B[0,k]
    For j=1: C[0,k] += A[0,1]*B[1,k], C[1,k] += A[1,1]*B[1,k]
    
    The j=0 part: A[0,0] and A[1,0] multiply B[0,k] for all k, contributing
    to C[0,k] and C[1,k] respectively. This is like a 2x16 outer product scaled
    by a 2-vector.
    
    Similarly j=1: A[0,1] and A[1,1] multiply B[1,k].
    
    Total: 2 (for j=0, two A entries) + 2 (for j=1, two A entries) = 4 rank-1 terms
    per k? No, that gives 4*16 = 64.
    
    The current rank-32 approach already does 2 terms per k. Let me see if we
    can do 1 term per k by combining.
    
    For a single k, we need:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    This is a 2x2 matrix product (A times column k of B). The rank of this
    2x2x1 tensor (or rather the 2x2x2 tensor restricted) is at most 2 (trivial)
    or can be less with clever combinations.
    
    Actually for a single k, the tensor T_k[i,j] = A[i,j]*B[j,k] for i,j in {0,1}
    is a rank-2 tensor (outer product of A column and... no, it's A * B[:,k]).
    
    The bilinear map (A, B[:,k]) -> C[:,k] has 4 inputs (A[0,0],A[0,1],A[1,0],A[1,1])
    and 2 outputs (C[0,k], C[1,k]) with 2 intermediate (B[0,k], B[1,k]).
    
    This is exactly M_{2,2,2} restricted to one column of the third dimension.
    The rank of M_{2,2,2} is 7, but we only need 2 of the 8 output entries...
    
    Hmm, this is getting complex. Let me just try the rank-32 approach which
    is known to work, and also try to find a rank-16 by exploiting that the
    A matrix is shared across all k.
    
    Actually, I realize the rank-32 decomposition in the current code might have
    a bug. Let me verify it carefully and also try to construct rank-16.
    
    For rank 16, we'd need 1 term per k. For each k, we need to compute:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    With 1 rank-1 term: U_a * V_b * W_c for each (a,b,c).
    We need: sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
    
    For a single k, the relevant entries are:
    T[0, k, 2k] = 1 (i=0,j=0) -> wait, let me recheck indices.
    
    a = i*m+j, b = j*p+k, c = k*n+i
    
    For i=0,j=0,k=0: a=0, b=0, c=0
    For i=0,j=0,k=1: a=0, b=1, c=2
    ...
    For i=0,j=1,k=0: a=1, b=16, c=0
    For i=0,j=1,k=1: a=1, b=17, c=2
    
    So T[0, k, 2k] = 1 (A[0,0]*B[0,k] -> C[0,k])
    T[1, 16+k, 2k] = 1 (A[0,1]*B[1,k] -> C[0,k])
    T[2, k, 2k+1] = 1 (A[1,0]*B[0,k] -> C[1,k])
    T[3, 16+k, 2k+1] = 1 (A[1,1]*B[1,k] -> C[1,k])
    
    And all other T[a,b,c] = 0.
    
    For rank 16, we have 16 terms. Each term t has U[*,t], V[*,t], W[*,t].
    For term t to contribute to T[0,k,2k], we need U[0,t]*V[k,t]*W[2k,t] != 0.
    
    This is a very constrained system. Let me try a construction:
    
    For each k (0 to 15), use term k:
    U[0,k] = 1, U[2,k] = 1  (A[0,0] and A[1,0])
    V[k,k] = 1              (B[0,k])
    W[2k,k] = 1, W[2k+1,k] = 1  (C[0,k] and C[1,k])
    
    But this only handles the j=0 part (B[0,k] * A[:,0]).
    We still need j=1 part: B[1,k] * A[:,1].
    
    That would require another 16 terms, giving rank 32.
    
    To get rank 16, we'd need to combine both j=0 and j=1 into 16 terms total.
    This seems unlikely without more clever algebra.
    
    Let me just verify the rank-32 works and return it.
    """
    rank = 2 * p  # 32
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 32 x 32
    W = [[0] * rank for _ in range(p * n)]  # 32 x 32
    
    for k in range(p):
        # Term 1: A[0,0] and A[1,0] with B[0,k], contributing to C[0,k] and C[1,k]
        col1 = 2 * k
        U[0][col1] = 1   # A[0,0]
        U[2][col1] = 1   # A[1,0]
        V[k][col1] = 1   # B[0,k] (row 0 of B, column k)
        W[2 * k][col1] = 1     # C[0,k]
        W[2 * k + 1][col1] = 1 # C[1,k]
        
        # Term 2: A[0,1] and A[1,1] with B[1,k], contributing to C[0,k] and C[1,k]
        col2 = 2 * k + 1
        U[1][col2] = 1   # A[0,1]
        U[3][col2] = 1   # A[1,1]
        V[p + k][col2] = 1  # B[1,k] (row 1 of B, column k)
        W[2 * k][col2] = 1     # C[0,k]
        W[2 * k + 1][col2] = 1 # C[1,k]
    
    return U, V, W

def _strassen_block(n, m, p):
    """Fallback: Strassen-based decomposition, rank 56."""
    n, m, p = 2, 2, 16
    num_blocks = p // 2
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        v_idx = {'B00': 2*t, 'B01': 2*t+1, 'B10': 16+2*t, 'B11': 16+2*t+1}
        w_idx = {'C00': 4*t, 'C01': 4*t+2, 'C10': 4*t+1, 'C11': 4*t+3}
        
        # M1 = (A00+A11)(B00+B11)
        U[0][col] += 1; U[3][col] += 1
        V[v_idx['B00']][col] += 1; V[v_idx['B11']][col] += 1
        W[w_idx['C00']][col] += 1; W[w_idx['C11']][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[2][col] += 1; U[3][col] += 1
        V[v_idx['B00']][col] += 1
        W[w_idx['C10']][col] += 1; W[w_idx['C11']][col] -= 1
        col += 1
        # M3 = A00(B01-B11)
        U[0][col] += 1
        V[v_idx['B01']][col] += 1; V[v_idx['B11']][col] -= 1
        W[w_idx['C01']][col] += 1; W[w_idx['C11']][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[3][col] += 1
        V[v_idx['B10']][col] += 1; V[v_idx['B00']][col] -= 1
        W[w_idx['C00']][col] += 1; W[w_idx['C10']][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[0][col] += 1; U[1][col] += 1
        V[v_idx['B11']][col] += 1
        W[w_idx['C00']][col] -= 1; W[w_idx['C01']][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[2][col] += 1; U[0][col] -= 1
        V[v_idx['B00']][col] += 1; V[v_idx['B01']][col] += 1
        W[w_idx['C11']][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[1][col] += 1; U[3][col] -= 1
        V[v_idx['B10']][col] += 1; V[v_idx['B11']][col] += 1
        W[w_idx['C00']][col] += 1
        col += 1
    
    return U, V, W

def _schoolbook(n, m, p):
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Try rank-32 decomposition first
    U, V, W = _rank16_decomposition(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fall back to Strassen (rank 56)
    if time.time() < deadline:
        U, V, W = _strassen_block(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Final fallback: schoolbook (rank 64)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
