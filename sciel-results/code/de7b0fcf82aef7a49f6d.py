"""Strassen-based rank-99 decomposition for 3x5 by 5x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction as F

DIMENSIONS = [3, 5, 7]

def _schoolbook(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
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

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms of Strassen's 2x2x2 matrix multiplication.
    Each term is a triple (u, v, w) where:
      u is a vector of length 4 (indexed by (i,a) for i,a in {0,1})
      v is a vector of length 4 (indexed by (b,k) for b,k in {0,1})
      w is a vector of length 4 (indexed by (k,i) for k,i in {0,1})
    
    The constraint is: for all (i,a,b,k):
      sum_t w[k*2+i, t] * u[i*2+a, t] * v[b*2+k, t] = 1 if a==b else 0
    
    Strassen's 7 products:
      M0 = (A00+A11)(B00+B11)
      M1 = (A10+A11)B00
      M2 = A00(B01-B11)
      M3 = A11(B10-B00)
      M4 = (A00+A01)B11
      M5 = (A10-A00)(B00+B01)
      M6 = (A01-A11)(B10+B11)
    
    C00 = M0 - M3 + M5
    C01 = M2 + M4
    C10 = M1 + M3
    C11 = M0 - M2 + M6
    
    For each Mt, u_t[i,a] is the coefficient of A[i,a] in the first factor,
    v_t[b,k] is the coefficient of B[b,k] in the second factor,
    w_t[k,i] is the coefficient of Mt in C[k,i].
    """
    terms = []
    
    # M0 = (A00+A11)(B00+B11)
    # u: A00+A11 -> [1, 0, 0, 1] (indices: 00,01,10,11)
    # v: B00+B11 -> [1, 0, 0, 1] (indices: 00,01,10,11)
    # w: contributes +1 to C00 (k=0,i=0) and +1 to C11 (k=1,i=1)
    # w indices: (k,i): 00,01,10,11 -> w[0]=1, w[3]=1
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M1 = (A10+A11)B00
    # u: A10+A11 -> [0, 0, 1, 1]
    # v: B00 -> [1, 0, 0, 0]
    # w: contributes +1 to C10 (k=1,i=0)
    # w indices: w[2]=1 (k=1,i=0)
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    
    # M2 = A00(B01-B11)
    # u: A00 -> [1, 0, 0, 0]
    # v: B01-B11 -> [0, 1, 0, -1]
    # w: contributes +1 to C01 (k=0,i=1) and -1 to C11 (k=1,i=1)
    # w indices: w[1]=1 (k=0,i=1), w[3]=-1 (k=1,i=1)
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,-1]))
    
    # M3 = A11(B10-B00)
    # u: A11 -> [0, 0, 0, 1]
    # v: B10-B00 -> [-1, 0, 1, 0]
    # w: contributes -1 to C00 (k=0,i=0) and +1 to C10 (k=1,i=0)
    # w indices: w[0]=-1 (k=0,i=0), w[2]=1 (k=1,i=0)
    terms.append(([0,0,0,1], [-1,0,1,0], [-1,0,1,0]))
    
    # M4 = (A00+A01)B11
    # u: A00+A01 -> [1, 1, 0, 0]
    # v: B11 -> [0, 0, 0, 1]
    # w: contributes +1 to C01 (k=0,i=1)
    # w indices: w[1]=1 (k=0,i=1)
    terms.append(([1,1,0,0], [0,0,0,1], [0,1,0,0]))
    
    # M5 = (A10-A00)(B00+B01)
    # u: A10-A00 -> [-1, 0, 1, 0]
    # v: B00+B01 -> [1, 1, 0, 0]
    # w: contributes +1 to C00 (k=0,i=0)
    # w indices: w[0]=1 (k=0,i=0)
    terms.append(([-1,0,1,0], [1,1,0,0], [1,0,0,0]))
    
    # M6 = (A01-A11)(B10+B11)
    # u: A01-A11 -> [0, 1, 0, -1]
    # v: B10+B11 -> [0, 0, 1, 1]
    # w: contributes +1 to C11 (k=1,i=1)
    # w indices: w[3]=1 (k=1,i=1)
    terms.append(([0,1,0,-1], [0,0,1,1], [0,0,0,1]))
    
    return terms

def _build_block_decomposition(n, m, p):
    """
    Build a decomposition using Strassen for 2x2x2 sub-blocks and schoolbook elsewhere.
    
    Partition:
    - I (size n=3): [2, 1]
    - J (size m=5): [2, 2, 1]
    - K (size p=7): [2, 2, 2, 1]
    
    For each combination of blocks (I_x, J_y, K_z):
    - If all have size 2: use Strassen (rank 7)
    - Otherwise: use schoolbook (rank I_x_size * J_y_size * K_z_size)
    """
    I_blocks = [(0, 2), (2, 1)]
    J_blocks = [(0, 2), (2, 2), (4, 1)]
    K_blocks = [(0, 2), (2, 2), (4, 2), (6, 1)]
    
    strassen_terms = _strassen_2x2x2_terms()
    
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    
    col = 0
    for i_start, i_size in I_blocks:
        for j_start, j_size in J_blocks:
            for k_start, k_size in K_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Use Strassen for this 2x2x2 block
                    for (u, v, w) in strassen_terms:
                        # u is indexed by (i,a) where i in {0,1}, a in {0,1}
                        # v is indexed by (b,k) where b in {0,1}, k in {0,1}
                        # w is indexed by (k,i) where k in {0,1}, i in {0,1}
                        for li in range(2):
                            for la in range(2):
                                gi = i_start + li
                                ga = j_start + la
                                if abs(u[li * 2 + la]) > 0:
                                    U[gi * m + ga][col] = u[li * 2 + la]
                        for lb in range(2):
                            for lk in range(2):
                                gb = j_start + lb
                                gk = k_start + lk
                                if abs(v[lb * 2 + lk]) > 0:
                                    V[gb * p + gk][col] = v[lb * 2 + lk]
                        for lk in range(2):
                            for li in range(2):
                                gi = i_start + li
                                gk = k_start + lk
                                if abs(w[lk * 2 + li]) > 0:
                                    W[gk * n + gi][col] = w[lk * 2 + li]
                        col += 1
                else:
                    # Use schoolbook for this block
                    for li in range(i_size):
                        for lj in range(j_size):
                            for lk in range(k_size):
                                gi = i_start + li
                                gj = j_start + lj
                                gk = k_start + lk
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
    
    # Trim to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}

def _validate_decomposition(U, V, W, n, m, p, tol=1e-7):
    """Check that the factor matrices define the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for a in range(m):
            for b in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += W[k * n + i][c] * U[i * m + a][c] * V[b * p + k][c]
                    expected = 1.0 if a == b else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True

def _validate_with_numpy(U, V, W, n, m, p):
    """Validate using numpy with random matrices."""
    np.random.seed(12345)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C using the decomposition
    # z_c = (sum_{i,j} U[i*m+j, c] A[i,j]) * (sum_{j,k} V[j*p+k, c] B[j,k])
    # C[i,k] = sum_c W[k*n+i, c] z_c
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # z_c for each c
    # U_flat: (n*m, rank), A_flat: (n*m,)
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    U_A = U_np @ A_flat  # (rank,)
    V_B = V_np @ B_flat  # (rank,)
    z = U_A * V_B  # (rank,)
    
    C_computed = (W_np @ z).reshape(p, n).T  # (n, p)
    
    return np.allclose(C_computed, C_expected, atol=1e-10)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, leaving 20s margin
    
    n, m, p = 3, 5, 7
    
    # Try the block Strassen decomposition (rank 99)
    result = _build_block_decomposition(n, m, p)
    rank = len(result["U"][0])
    
    # Validate with numpy
    if _validate_with_numpy(result["U"], result["V"], result["W"], n, m, p):
        if time.time() < deadline:
            # Also validate with exact check
            if _validate_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
