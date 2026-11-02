"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_exceeded():
    return (time.time() - _START_TIME) > _TIME_LIMIT

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication.
    
    The tensor T[i,j,k; a,b,c] = delta[i,a]*delta[j,b]*delta[k,c] for (C = A*B)[a,c] = sum_b A[a,b]*B[b,c]
    where A is n x m, B is m x p, C is n x p.
    
    In the standard tensor decomposition:
    T = sum_r U_r * V_r * W_r
    where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    The tensor entry T[i*m+j, j'*p+k, k'*n+i'] should equal delta[i,i']*delta[j,j']*delta[k,k]
    Wait, let me reconsider the indexing.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The multiplication tensor has indices (i, j, k) where:
    - i: row of A (and C), range [0, n)
    - j: column of A / row of B, range [0, m)  
    - k: column of B (and C), range [0, p)
    
    T[i,j,k] = 1 if this is a valid multiplication term, else 0
    Actually T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k} ... no.
    
    The tensor for matrix multiplication M_{n,m,p} is defined as:
    T = sum_{i,j,k} e_i^A * e_j^B * e_k^C * delta_{i,i}delta_{j,j}delta_{k,k}
    
    More precisely, the bilinear map (A, B) -> C = AB has the property that:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor is T[i,j,k] = 1 for all valid (i,j,k) in the sum, meaning:
    T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i * e_j * e_k
    
    In the decomposition U * V * W:
    (U V W)(i*m+j, j*p+k, k*n+i') = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i', r]
    
    This should equal delta_{i,i'} * delta_{j,j} * delta_{k,k} = 1 if i=i' and j=j and k=k
    
    Wait, the W index is k*n+i'. So W[k*n+i', r] corresponds to output element (i', k).
    The input indices are: U index = i*m+j (A element at row i, col j), V index = j*p+k (B element at row j, col k).
    
    For the decomposition to work:
    sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    Let me verify with the schoolbook: each (i,j,k) gets its own column r.
    U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    Then sum_r U[i*m+j,r]*V[j'*p+k',r]*W[k''*n+i',r] = 1 only if there's a column where i*m+j matches, j*p+k matches j'*p+k', and k''*n+i' matches k*n+i.
    That means i=i', j=j', k=k', k''=k. So yes, it works.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check a sample of tensor entries
    # T[i*m+j, j'*p+k, k''*n+i'] should be delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
    # But we can't check all n*m * m*p * p*n = 4*32*32 = 4096 entries easily in a loop
    # Instead, let's check the full tensor via the decomposition
    
    # The tensor T has shape (n*m, m*p, p*n)
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k''*n+i'
    # T[a,b,c] = 1 iff i=i' and j=j' and k=k''
    
    # Let's verify by checking that for each (i,j), (j',k), (k'',i'):
    # sum_r U[i*m+j,r] * V[j'*p+k,r] * W[k''*n+i',r] = 1 iff i=i', j=j', k=k''
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for jp in range(m):  # j'
                for k in range(p):
                    for kp in range(p):  # k''
                        for ip in range(n):  # i'
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            # Compute sum_r U[i*m+j,r] * V[jp*p+k,r] * W[kp*n+ip,r]
                            val = 0.0
                            for r in range(rank):
                                val += U[i*m+j][r] * V[jp*p+k][r] * W[kp*n+ip][r]
                            if abs(val - expected) > 1e-9:
                                errors += 1
                                if errors > 10:
                                    return False
    return errors == 0

def _strassen_2x2():
    """Return Laderman's rank-7 decomposition for 2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) where:
    - U index: i*m+j = i*2+j for A[i,j]
    - V index: j*p+k = j*2+k for B[j,k]  
    - W index: k*n+i = k*2+i for C[i,k]
    
    Strassen's 7 products for 2x2x2:
    C[0,0] = M1 + M4 - M5 - M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M3 + M6 + M7
    
    where:
    M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    M2 = (A[1,0]+A[1,1])*B[0,0]
    M3 = A[0,0]*(B[0,1]-B[1,1])
    M4 = A[1,1]*(B[1,0]-B[0,0])
    M5 = (A[0,0]+A[0,1])*B[1,1]
    M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    M7 = (A[1,1]-A[0,1])*(B[1,0]+B[1,1])
    
    So U rows (for A indices 0,1,2,3 = [0,0],[0,1],[1,0],[1,1]):
    U[0] = [1, 0, 1, 0, 1, -1, 0]  (A[0,0] appears in M1, M3, M5, M6)
    U[1] = [0, 0, 0, 0, 1, 0, -1]  (A[0,1] appears in M5, M7)
    U[2] = [0, 1, 0, 0, 0, 1, 0]   (A[1,0] appears in M2, M6)
    U[3] = [1, 1, 0, 1, 0, 0, 1]   (A[1,1] appears in M1, M2, M4, M7)
    
    V rows (for B indices 0,1,2,3 = [0,0],[0,1],[1,0],[1,1]):
    V[0] = [1, 1, 0, -1, 0, 1, 0]  (B[0,0] appears in M1, M2, M4, M6)
    V[1] = [0, 0, 1, 0, 0, 1, 0]   (B[0,1] appears in M3, M6)
    V[2] = [0, 0, 0, 1, 0, 0, 1]   (B[1,0] appears in M4, M7)
    V[3] = [1, 0, -1, 0, 1, 0, 1]  (B[1,1] appears in M1, M3, M5, M7)
    
    W rows (for C indices 0,1,2,3 = [0,0],[0,1],[1,0],[1,1] since W[k*n+i] = W[k*2+i]):
    C[0,0] = M1+M4-M5-M7 -> W[0] = [1, 0, 0, 1, -1, 0, -1]
    C[0,1] = M3+M5 -> W[1] = [0, 0, 1, 0, 1, 0, 0]
    C[1,0] = M2+M4 -> W[2] = [0, 1, 0, 1, 0, 0, 0]
    C[1,1] = M1-M3+M6+M7 -> W[3] = [1, 0, -1, 0, 0, 1, 1]
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, -1, 0, 0, 1, 1]
    ]
    return U, V, W

def _build_strassen_2x16():
    """Build a rank-56 decomposition for 2x2 by 2x16 using Strassen on 2x2 blocks.
    
    Group the 16 columns of B into 8 blocks of 2 columns each.
    For each block, use Strassen's 7 products.
    Total rank = 8 * 7 = 56.
    """
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    U_s, V_s, W_s = _strassen_2x2()
    
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    for block in range(num_blocks):
        base = block * 7
        for r in range(7):
            col = base + r
            # U: A is 2x2, so U has 4 rows
            for i in range(n * m):
                U[i][col] = U_s[i][r]
            # V: B block is 2x2, rows are j*p + k where j in [0,2), k in [block*2, block*2+2)
            for j in range(m):
                for k in range(2):
                    v_row = j * p + (block * 2 + k)
                    V[v_row][col] = V_s[j * 2 + k][r]
            # W: C block is 2x2, rows are k*n + i where k in [block*2, block*2+2), i in [0,2)
            for k in range(2):
                for i in range(n):
                    w_row = (block * 2 + k) * n + i
                    W[w_row][col] = W_s[k * 2 + i][r]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Start with the Strassen-based decomposition (rank 56)
    U, V, W = _build_strassen_2x16()
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    # Verify the Strassen decomposition
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook if Strassen is wrong
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
        best_U, best_V, best_W = U, V, W
        best_rank = rank
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to find a lower-rank decomposition via search
    # The known lower bound for 2x2x16 is not well-established, but 56 is a good upper bound.
    # Let's try a random search / local search approach to see if we can do better.
    
    # For now, return the Strassen decomposition which gives rank 56 (score 64/56 ≈ 1.143)
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
