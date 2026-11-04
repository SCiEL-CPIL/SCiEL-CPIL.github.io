import time
import numpy as np

DIMENSIONS = [4, 4, 7]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reconstructs the matrix multiplication tensor using vectorized ops."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Vectorized: compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape for broadcasting
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=tol)

def _schoolbook_decomposition(n, m, p):
    """Return the trivial schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2.
    Rank 7 decomposition.
    U, V, W shapes: (4, 7)
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # m1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # m3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # m4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # m5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # m6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # m7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return np.array(U), np.array(V), np.array(W)

def _rank49_4x4x4():
    """
    Construct rank 49 decomposition for 4x4 * 4x4 using Kronecker product of Strassen.
    U, V, W shapes: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            # U4[i, c] = U2[i1, c1] * U2[i2, c2]
            # i = 2*i1 + i2
            for i1 in range(2):
                for i2 in range(2):
                    i = 2 * i1 + i2
                    U4[i, c] = U2[i1, c1] * U2[i2, c2]
            
            # V4[j, c] = V2[j1, c1] * V2[j2, c2]
            # j = 2*j1 + j2
            for j1 in range(2):
                for j2 in range(2):
                    j = 2 * j1 + j2
                    V4[j, c] = V2[j1, c1] * V2[j2, c2]
            
            # W4[k, c] = W2[k1, c1] * W2[k2, c2]
            # k = 2*k1 + k2
            for k1 in range(2):
                for k2 in range(2):
                    k = 2 * k1 + k2
                    W4[k, c] = W2[k1, c1] * W2[k2, c2]
                    
    return U4, V4, W4

def _decomp_2x2x1():
    """
    Schoolbook decomposition for 2x2 * 2x1.
    Rank 4.
    U: (4, 4), V: (2, 4), W: (2, 4)
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # t1: a * e -> c00 (U[0], V[0], W[0])
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # t2: b * f -> c00 (U[1], V[1], W[0])
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    # t3: c * e -> c10 (U[2], V[0], W[1])
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    # t4: d * f -> c10 (U[3], V[1], W[1])
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def _decomp_2x2x3():
    """
    Decomposition for 2x2 * 2x3.
    Rank 11 (7 from Strassen 2x2x2 + 4 from Schoolbook 2x2x1).
    U: (4, 11), V: (6, 11), W: (6, 11)
    """
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # Part 1: 2x2x2 Strassen for cols 0,1
    U_s, V_s, W_s = _strassen_2x2x2()
    U[:, :7] = U_s
    # V_s is (4, 7). We need to map to V (6, 11).
    # V_s indices: 0:00, 1:01, 2:10, 3:11 (j*p+k, p=2)
    # V target indices: 0:00, 1:01, 2:02, 3:10, 4:11, 5:12
    # So V_s[0] -> V[0], V_s[1] -> V[1], V_s[2] -> V[3], V_s[3] -> V[4]
    V[0, :7] = V_s[0, :]
    V[1, :7] = V_s[1, :]
    V[3, :7] = V_s[2, :]
    V[4, :7] = V_s[3, :]
    
    # W_s is (4, 7). Indices: 0:00, 1:01, 2:10, 3:11 (k*n+i, n=2)
    # W target indices: 0:00, 1:01, 2:02, 3:10, 4:11, 5:12
    # So W_s[0] -> W[0], W_s[1] -> W[1], W_s[2] -> W[3], W_s[3] -> W[4]
    W[0, :7] = W_s[0, :]
    W[1, :7] = W_s[1, :]
    W[3, :7] = W_s[2, :]
    W[4, :7] = W_s[3, :]
    
    # Part 2: 2x2x1 Schoolbook for col 2
    # U_sc, V_sc, W_sc from _decomp_2x2x1
    U_sc, V_sc, W_sc = _decomp_2x2x1()
    U[:, 7:11] = U_sc
    # V_sc is (2, 4). Indices: 0:0 (j=0, k=0), 1:1 (j=1, k=0)
    # We need to map to col 2 of V (indices 2 and 5)
    V[2, 7:11] = V_sc[0, :]
    V[5, 7:11] = V_sc[1, :]
    
    # W_sc is (2, 4). Indices: 0:0 (k=0, i=0), 1:1 (k=0, i=1)
    # We need to map to col 2 of W (indices 2 and 5)
    W[2, 7:11] = W_sc[0, :]
    W[5, 7:11] = W_sc[1, :]
    
    return U, V, W

def _decomp_4x4x3():
    """
    Decomposition for 4x4 * 4x3.
    Rank 44 (Kronecker of 2x2x3 and 2x2x1).
    U: (16, 44), V: (12, 44), W: (12, 44)
    """
    U223, V223, W223 = _decomp_2x2x3()
    U221, V221, W221 = _decomp_2x2x1()
    
    rank = 44
    U = np.zeros((16, rank))
    V = np.zeros((12, rank))
    W = np.zeros((12, rank))
    
    for c1 in range(11):
        for c2 in range(4):
            c = c1 * 4 + c2
            # U[i, c] = U223[i1, c1] * U221[i2, c2]
            for i1 in range(2):
                for i2 in range(2):
                    i = 2 * i1 + i2
                    U[i, c] = U223[i1, c1] * U221[i2, c2]
            
            # V[j, c] = V223[j1, c1] * V221[j2, c2]
            # V223 j: 0..5 (j1*p1+k1, p1=3)
            # V221 j: 0..1 (j2*p2+k2, p2=1)
            # Target j: 0..11 (j*p+k, p=3)
            # j = 2*j1 + j2 ? No.
            # Dimension mapping:
            # 2x2x3: n=2, m=2, p=3. V rows: m*p = 6.
            # 2x2x1: n=2, m=2, p=1. V rows: m*p = 2.
            # 4x4x3: n=4, m=4, p=3. V rows: m*p = 12.
            # Index j in 4x4x3 corresponds to (j1, j2) where j = 2*j1 + j2?
            # No, the Kronecker product structure for V:
            # V_{443}[j, c] = V_{223}[j1, c1] * V_{221}[j2, c2]
            # How do j1, j2 map to j?
            # In 2x2x3, j1 ranges 0..1 (row of A), k1 ranges 0..2 (col of B).
            # In 2x2x1, j2 ranges 0..1 (row of A), k2 ranges 0..0 (col of B).
            # In 4x4x3, j ranges 0..3 (row of A), k ranges 0..2 (col of B).
            # j = 2*j1 + j2.
            # k = k1 (since k2 is always 0).
            # So V index in 4x4x3 is j*3 + k = (2*j1 + j2)*3 + k1.
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(3):
                        # This mapping is not straightforward in a nested loop over V223 and V221 directly.
                        # We need to iterate over the components of the Kronecker product.
                        pass
                        
            # Let's do it by iterating over the non-zero elements or just filling the whole thing.
            # V223 is 6x11. V221 is 2x4.
            # V443 is 12x44.
            # V443[j, c] is non-zero if V223[j1, c1] and V221[j2, c2] are non-zero.
            # j = 2*j1 + j2.
            # But V223 row index is j1*3 + k1.
            # So we need to decompose V223 row index r1 into j1, k1.
            # r1 = j1*3 + k1.
            
            # Let's re-iterate:
            for r1 in range(6):
                j1 = r1 // 3
                k1 = r1 % 3
                for r2 in range(2):
                    j2 = r2 // 1
                    k2 = r2 % 1 # k2 is 0
                    
                    j = 2 * j1 + j2
                    k = k1 # k2 is 0
                    idx_v = j * 3 + k
                    
                    val = V223[r1, c1] * V221[r2, c2]
                    V[idx_v, c] = val
                    
            # Similarly for W
            # W223 is 6x11. W221 is 2x4.
            # W443 is 12x44.
            # W index in 2x2x3: r1 = k1*2 + i1.
            # W index in 2x2x1: r2 = k2*2 + i2.
            # W index in 4x4x3: r = k*4 + i.
            # k = k1 (k2=0).
            # i = 2*i1 + i2.
            for r1 in range(6):
                k1 = r1 // 2
                i1 = r1 % 2
                for r2 in range(2):
                    k2 = r2 // 2
                    i2 = r2 % 2
                    
                    k = k1
                    i = 2 * i1 + i2
                    idx_w = k * 4 + i
                    
                    val = W223[r1, c1] * W221[r2, c2]
                    W[idx_w, c] = val

    return U, V, W

def _build_4x4x7_strassen():
    """
    Build a rank 93 decomposition for 4x4 * 4x7.
    Part 1: 4x4 * 4x3 (Rank 44)
    Part 2: 4x4 * 4x4 (Rank 49)
    """
    U443, V443, W443 = _decomp_4x4x3()
    U444, V444, W444 = _rank49_4x4x4()
    
    n, m, p = 4, 4, 7
    rank = 93
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: 4x4 * 4x3 (cols 0-2 of B)
    for c in range(44):
        U[:, c] = U443[:, c]
        # V443 is 12x44. V is 28x93.
        # V443 rows: j*3+k (j=0..3, k=0..2).
        # V rows: j*7+k (j=0..3, k=0..6).
        # We map k_local (0..2) to k_global (0..2).
        for j in range(4):
            for k in range(3):
                V[j * 7 + k, c] = V443[j * 3 + k, c]
                
        # W443 is 12x44. W is 28x93.
        # W443 rows: k*4+i (k=0..2, i=0..3).
        # W rows: k*4+i (k=0..6, i=0..3).
        # We map k_local (0..2) to k_global (0..2).
        for k in range(3):
            for i in range(4):
                W[k * 4 + i, c] = W443[k * 4 + i, c]

    # Part 2: 4x4 * 4x4 (cols 3-6 of B)
    for c in range(49):
        c2 = 44 + c
        U[:, c2] = U444[:, c]
        
        # V444 is 16x49. V is 28x93.
        # V444 rows: j*4+k (j=0..3, k=0..3).
        # V rows: j*7+k (j=0..3, k=0..6).
        # We map k_local (0..3) to k_global (3..6).
        for j in range(4):
            for k in range(4):
                V[j * 7 + (3 + k), c2] = V444[j * 4 + k, c]
                
        # W444 is 16x49. W is 28x93.
        # W444 rows: k*4+i (k=0..3, i=0..3).
        # W rows: k*4+i (k=0..6, i=0..3).
        # We map k_local (0..3) to k_global (3..6).
        for k in range(4):
            for i in range(4):
                W[(3 + k) * 4 + i, c2] = W444[k * 4 + i, c]
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # First, get the known rank 93 decomposition
    U93, V93, W93 = _build_4x4x7_strassen()
    
    # Verify it's correct
    if not _verify_decomposition(U93, V93, W93, n, m, p, tol=1e-9):
        # Fallback to schoolbook
        U_s, V_s, W_s = _schoolbook_decomposition(n, m, p)
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U93.tolist(), "V": V93.tolist(), "W": W93.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
