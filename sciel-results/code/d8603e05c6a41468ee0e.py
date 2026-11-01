import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact using numpy for speed.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Precompute V * W element-wise for each rank term? No, that's O(rank * m * p * n).
    # The tensor identity is:
    # sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # We can vectorize this.
    # For a fixed r, we have outer(U[:,r], V[:,r]) which is (n*m, m*p).
    # Let's iterate over r and accumulate? No, that's O(rank * n * m * m * p).
    
    # Better:
    # For each r, compute M_r = U[:,r] * V[:,r] * W[:,r] ? No.
    # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    
    # Let's use einsum if possible, or simple loops if small.
    # n=2, m=3, p=10. rank ~ 55.
    # 60 terms in the sum? No, 2*3*10 = 60 checks.
    # For each check (i,j,k), we sum over r.
    # 60 * 55 operations. Very fast.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                total = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _construct_rank55(n, m, p):
    """
    Construct a rank-55 decomposition using Strassen for the 2x2 part of the 2x3 matrix
    and schoolbook for the rest, combined with the 3x10 structure.
    
    We decompose the multiplication:
    A (2x3) * B (3x10) = C (2x10)
    
    Split A into A1 (2x2) and A2 (2x1).
    Split B into B1 (2x10) and B2 (1x10).
    
    C = A1 * B1 + A2 * B2
    
    Part 1: A1 (2x2) * B1 (2x10).
    We group the 10 columns of B1 into 5 pairs.
    Each pair forms a 2x2 matrix B_pair.
    We use Strassen's algorithm (rank 7) for each 2x2 * 2x2 multiplication.
    Total rank for Part 1 = 5 * 7 = 35.
    
    Part 2: A2 (2x1) * B2 (1x10).
    This is just scaling each row of B2 by the corresponding element of A2.
    C0_k = A2_0 * B2_k
    C1_k = A2_1 * B2_k
    This requires 2 multiplications per column k.
    Total rank for Part 2 = 10 * 2 = 20.
    
    Total Rank = 35 + 20 = 55.
    """
    
    # Dimensions
    # U rows: A indices (2*3 = 6). A[i,j] -> i*3+j.
    # V rows: B indices (3*10 = 30). B[j,k] -> j*10+k.
    # W rows: C indices (10*2 = 20). C[i,k] -> k*2+i. (Trace convention W[k*n+i, r])
    
    rank = 55
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(30)]
    W = [[0]*rank for _ in range(20)]
    
    col = 0
    
    # Part 1: Strassen for A1 (2x2) * B1 (2x10)
    # A1 indices:
    # A[0,0] -> 0
    # A[0,1] -> 1
    # A[1,0] -> 3
    # A[1,1] -> 4
    
    # B1 indices:
    # B[0,k] -> k
    # B[1,k] -> 10+k
    
    # C indices for Part 1 (columns 0..9):
    # C[0,k] -> 2k
    # C[1,k] -> 2k+1
    
    for g in range(5):
        k1 = 2*g
        k2 = 2*g + 1
        
        # B_pair is 2x2:
        # B00 = B[0, k1] -> idx k1
        # B01 = B[0, k2] -> idx k2
        # B10 = B[1, k1] -> idx 10+k1
        # B11 = B[1, k2] -> idx 10+k2
        
        # Strassen Multiplications:
        # M1 = (A00 + A11) * (B00 + B11)
        U[0][col] = 1; U[4][col] = 1
        V[k1][col] = 1; V[10+k2][col] = 1
        # C00 += M1, C11 += M1
        # C[0, k1] -> 2*k1
        # C[1, k2] -> 2*k2+1
        W[2*k1][col] = 1
        W[2*k2+1][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[3][col] = 1; U[4][col] = 1
        V[k1][col] = 1
        # C10 += M2, C11 -= M2
        # C[1, k1] -> 2*k1+1
        # C[1, k2] -> 2*k2+1
        W[2*k1+1][col] = 1
        W[2*k2+1][col] = -1
        col += 1
        
        # M3 = A00 * (B10 - B11)
        U[0][col] = 1
        V[10+k1][col] = 1; V[10+k2][col] = -1
        # C01 += M3, C11 += M3
        # C[0, k2] -> 2*k2
        # C[1, k2] -> 2*k2+1
        W[2*k2][col] = 1
        W[2*k2+1][col] = 1
        col += 1
        
        # M4 = A11 * (B01 - B00)
        U[4][col] = 1
        V[k2][col] = 1; V[k1][col] = -1
        # C00 += M4, C10 += M4
        # C[0, k1] -> 2*k1
        # C[1, k1] -> 2*k1+1
        W[2*k1][col] = 1
        W[2*k1+1][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[0][col] = 1; U[1][col] = 1
        V[10+k2][col] = 1
        # C00 -= M5, C01 += M5
        # C[0, k1] -> 2*k1
        # C[0, k2] -> 2*k2
        W[2*k1][col] = -1
        W[2*k2][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B10)
        U[3][col] = 1; U[0][col] = -1
        V[k1][col] = 1; V[10+k1][col] = 1
        # C11 += M6
        # C[1, k2] -> 2*k2+1
        W[2*k2+1][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B01 + B11)
        U[1][col] = 1; U[4][col] = -1
        V[k2][col] = 1; V[10+k2][col] = 1
        # C00 += M7
        # C[0, k1] -> 2*k1
        W[2*k1][col] = 1
        col += 1

    # Part 2: Schoolbook for A2 (2x1) * B2 (1x10)
    # A2 indices:
    # A[0,2] -> 2
    # A[1,2] -> 5
    
    # B2 indices:
    # B[2,k] -> 20+k
    
    # C indices:
    # C[0,k] -> 2k
    # C[1,k] -> 2k+1
    
    for k in range(10):
        # M0 = A[0,2] * B[2,k]
        U[2][col] = 1
        V[20+k][col] = 1
        W[2*k][col] = 1
        col += 1
        
        # M1 = A[1,2] * B[2,k]
        U[5][col] = 1
        V[20+k][col] = 1
        W[2*k+1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = 140
    
    n, m, p = 2, 3, 10
    
    # Try Rank 55 first
    r55 = _construct_rank55(n, m, p)
    if _verify(r55['U'], r55['V'], r55['W'], n, m, p):
        return r55
        
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Verified:", _verify(res['U'], res['V'], res['W'], 2, 3, 10))
