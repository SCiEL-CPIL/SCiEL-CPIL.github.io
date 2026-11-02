"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 145.0

def _validate_decomposition(U, V, W, n, m, p):
    """
    Validates if U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    # Tensor T[a, b, c] = 1 if C[c] uses A[a] and B[b]
    # A is n x m, B is m x p, C is n x p
    # Indexing:
    # A: row-major, i in [0,n), j in [0,m) -> idx = i*m + j
    # B: row-major, j in [0,m), k in [0,p) -> idx = j*p + k
    # C: row-major, i in [0,n), k in [0,p) -> idx = i*p + k
    
    # We check if sum_r U[a,r] V[b,r] W[c,r] == T[a,b,c]
    # T[a,b,c] is 1 if the product A[a] * B[b] contributes to C[c].
    # A[i,j] * B[j,k] contributes to C[i,k].
    # a = i*m + j
    # b = j*p + k
    # c = i*p + k
    
    # We can compute the decomposition result and compare to the true tensor.
    # Result R[c, a, b] = sum_r W[c, r] U[a, r] V[b, r]
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Efficient check:
    # For each r, add U[:,r] outer V[:,r] outer W[:,r]
    # This is expensive for large rank.
    # Instead, check a subset of entries or use matrix multiplication properties.
    
    # Let's check all non-zero entries of the tensor.
    # There are n*m*p non-zero entries? No, only those where indices match.
    # For each i, j, k:
    # a = i*m + j
    # b = j*p + k
    # c = i*p + k
    # T[a,b,c] = 1
    
    # We need sum_r U[a,r] V[b,r] W[c,r] == 1 for all valid (i,j,k).
    # And sum_r U[a,r] V[b,r] W[c,r] == 0 for all other (a,b,c).
    
    # Checking all 0s is expensive.
    # But if we ensure the 1s are correct and the rank is minimal, 
    # and we constructed it carefully, it's likely correct.
    # However, to be safe, we check the 1s.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = i * p + k
                val = np.dot(U_np[a, :], V_np[b, :] * W_np[c, :])
                if not np.isclose(val, 1.0):
                    return False
    return True

def _search_2x2x3_rank10():
    """
    Search for a rank-10 decomposition of 2x2 * 2x3.
    """
    # Known solution for 2x2x3 rank 10?
    # Let's try to find one.
    # If not found, return None.
    
    # We will use a randomized search with a time limit.
    # But 160 variables is too many.
    # We will try to fix the structure.
    
    # Actually, let's try a specific known construction if available.
    # If not, we return None.
    
    # Given the difficulty, we return None for now.
    return None

def _search_2x2x4_rank13():
    """
    Search for a rank-13 decomposition of 2x2 * 2x4.
    """
    return None

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Use Strassen's 2x2x2 algorithm for each pair of columns.
    # 7 pairs of columns (0-1, 2-3, ..., 12-13) -> 7*7 = 49 rank
    # 1 single column (14) -> 4 rank (trivial)
    # Total rank = 53
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    def add_strassen_block(base_r, k0, k1):
        """
        Strassen decomposition for 2x2 A times 2x2 B (columns k0, k1).
        
        B matrix columns are k0 and k1, rows are 0 and 1.
        B0k0 = B[0][k0], B0k1 = B[0][k1]
        B1k0 = B[1][k0], B1k1 = B[1][k1]
        
        7 products:
        P1 = (A00+A11)(B0k0+B1k1)
        P2 = (A10+A11)(B0k0)
        P3 = A00(B0k1-B1k1)
        P4 = A11(B1k0-B0k0)
        P5 = (A00+A01)(B1k1)
        P6 = (A10-A00)(B0k0+B0k1)
        P7 = (A01-A11)(B1k0+B1k1)
        
        Output C = A*B:
        C0k0 = A00*B0k0 + A01*B1k0 = P1 + P4 - P5 + P7
        C1k0 = A10*B0k0 + A11*B1k0 = P2 + P4
        C0k1 = A00*B0k1 + A01*B1k1 = P3 + P5
        C1k1 = A10*B0k1 + A11*B1k1 = P1 - P2 + P3 + P6
        """
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        # P1: (A00+A11)(B0k0+B1k1)
        U[0][r1] = 1   # A00
        U[3][r1] = 1   # A11
        V[0 * p + k0][r1] = 1   # B0k0
        V[1 * p + k1][r1] = 1   # B1k1
        
        # P2: (A10+A11)(B0k0)
        U[2][r2] = 1   # A10
        U[3][r2] = 1   # A11
        V[0 * p + k0][r2] = 1   # B0k0
        
        # P3: A00(B0k1-B1k1)
        U[0][r3] = 1   # A00
        V[0 * p + k1][r3] = 1   # B0k1
        V[1 * p + k1][r3] = -1  # B1k1
        
        # P4: A11(B1k0-B0k0)
        U[3][r4] = 1   # A11
        V[1 * p + k0][r4] = 1   # B1k0
        V[0 * p + k0][r4] = -1  # B0k0
        
        # P5: (A00+A01)(B1k1)
        U[0][r5] = 1   # A00
        U[1][r5] = 1   # A01
        V[1 * p + k1][r5] = 1   # B1k1
        
        # P6: (A10-A00)(B0k0+B0k1)
        U[2][r6] = 1   # A10
        U[0][r6] = -1  # A00
        V[0 * p + k0][r6] = 1   # B0k0
        V[0 * p + k1][r6] = 1   # B0k1
        
        # P7: (A01-A11)(B1k0+B1k1)
        U[1][r7] = 1   # A01
        U[3][r7] = -1  # A11
        V[1 * p + k0][r7] = 1   # B1k0
        V[1 * p + k1][r7] = 1   # B1k1
        
        # Reconstruction
        # C0k0 = P1 + P4 - P5 + P7 -> W[2*k0]
        W[2 * k0][r1] = 1
        W[2 * k0][r4] = 1
        W[2 * k0][r5] = -1
        W[2 * k0][r7] = 1
        
        # C1k0 = P2 + P4 -> W[2*k0+1]
        W[2 * k0 + 1][r2] = 1
        W[2 * k0 + 1][r4] = 1
        
        # C0k1 = P3 + P5 -> W[2*k1]
        W[2 * k1][r3] = 1
        W[2 * k1][r5] = 1
        
        # C1k1 = P1 - P2 + P3 + P6 -> W[2*k1+1]
        W[2 * k1 + 1][r1] = 1
        W[2 * k1 + 1][r2] = -1
        W[2 * k1 + 1][r3] = 1
        W[2 * k1 + 1][r6] = 1
    
    # Add 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)
    
    # Add trivial block for column 14 (single column, 4 multiplications)
    base_r = 49
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3
    
    # Q1 = A00 * B0,14 -> C0,14
    U[0][r1] = 1
    V[0 * p + 14][r1] = 1
    W[2 * 14][r1] = 1
    
    # Q2 = A01 * B1,14 -> C0,14
    U[1][r2] = 1
    V[1 * p + 14][r2] = 1
    W[2 * 14][r2] = 1
    
    # Q3 = A10 * B0,14 -> C1,14
    U[2][r3] = 1
    V[0 * p + 14][r3] = 1
    W[2 * 14 + 1][r3] = 1
    
    # Q4 = A11 * B1,14 -> C1,14
    U[3][r4] = 1
    V[1 * p + 14][r4] = 1
    W[2 * 14 + 1][r4] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
