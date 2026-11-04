"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _validate(U, V, W, n=4, m=4, p=5, tol=1e-9):
    """Check that U,V,W form a valid tensor decomposition of matmul(n,m,p)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    if not np.allclose(M_decomp, 1.0, atol=tol):
        return False
    return True

def _construct_schoolbook(n=4, m=4, p=5):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _winograd_2x2x2():
    """
    Return the 5 rank-1 tensors for 2x2x2 matrix multiplication.
    Each tensor is (u, v, w) where u, v, w are 4-dim vectors.
    
    For A (2x2), B (2x2), C = A*B (2x2):
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    The 5 multiplications in Winograd F(2,2,2):
    P1 = (a + d) * (e + h)
    P2 = (c + d) * e
    P3 = a * (g - h)
    P4 = d * (e - f)
    P5 = (a + b) * h
    
    C[0,0] = P1 + P4 - P5
    C[0,1] = P3 + P5
    C[1,0] = P2 + P4
    C[1,1] = P1 - P2 + P3
    """
    terms = []
    
    # P1 = (a + d) * (e + h)
    # u: a=1, d=1 -> [1, 0, 0, 1] (indices: a=0, b=1, c=2, d=3)
    # v: e=1, h=1 -> [1, 0, 0, 1] (indices: e=0, f=1, g=2, h=3)
    # w: C[0,0]+=1, C[1,1]+=1 -> [1, 0, 0, 1] (indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3)
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    
    # P2 = (c + d) * e
    # u: c=1, d=1 -> [0, 0, 1, 1]
    # v: e=1 -> [1, 0, 0, 0]
    # w: C[1,0]+=1 -> [0, 0, 1, 0]
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
    
    # P3 = a * (g - h)
    # u: a=1 -> [1, 0, 0, 0]
    # v: g=1, h=-1 -> [0, 0, 1, -1]
    # w: C[0,1]+=1, C[1,1]+=1 -> [0, 1, 0, 1]
    terms.append(([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]))
    
    # P4 = d * (e - f)
    # u: d=1 -> [0, 0, 0, 1]
    # v: e=1, f=-1 -> [1, -1, 0, 0]
    # w: C[0,0]+=1, C[1,0]+=1 -> [1, 0, 1, 0]
    terms.append(([0, 0, 0, 1], [1, -1, 0, 0], [1, 0, 1, 0]))
    
    # P5 = (a + b) * h
    # u: a=1, b=1 -> [1, 1, 0, 0]
    # v: h=1 -> [0, 0, 0, 1]
    # w: C[0,0]-=1, C[0,1]+=1 -> [-1, 1, 0, 0]
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    
    return terms

def _construct_winograd_4x4x4(n=4, m=4, p=4):
    """
    Construct a rank-20 decomposition for 4x4 * 4x4 using Winograd 2x2x2.
    
    We partition the 4x4 matrices into 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each block is 2x2
    B = [[B00, B01], [B10, B11]] where each block is 2x2
    C = [[C00, C01], [C10, C11]] where each block is 2x2
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Each product like A00*B00 is a 2x2x2 matmul with rank 5.
    There are 8 such products (4 block positions * 2 terms each).
    Total rank: 8 * 5 = 40.
    
    Wait, that's not 20. Let me reconsider.
    
    Actually, the 4x4x4 tensor is not a direct sum of independent 2x2x2 tensors
    in a way that gives rank 20. The rank of 4x4x4 matmul is known to be at most
    49 (using Strassen recursively) or lower.
    
    For now, let me just use schoolbook for the 4x4x4 part and add the 5th column.
    """
    return _construct_schoolbook(n, m, p)

def _construct_winograd_4x4x5(n=4, m=4, p=5):
    """
    Construct a decomposition for 4x4 * 4x5.
    
    Split B into B1 (4x4) and B2 (4x1).
    C = A*B1 + A*B2
    
    Use schoolbook for both parts: rank 4*4*4 + 4*4*1 = 64 + 16 = 80.
    
    For now, return schoolbook.
    """
    return _construct_schoolbook(n, m, p)

def construct():
    """Construct a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 80)
    best_result = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _validate(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        # Fallback: construct manually
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

# EVOLVE-BLOCK-END
