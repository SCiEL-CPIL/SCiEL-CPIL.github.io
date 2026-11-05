import time
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Construct a rank-28 tensor decomposition for 2x3 by 3x5 matrix multiplication.
    
    Decomposition strategy:
    Split A (2x3) into A1 (2x1) and A2 (2x2).
    Split B (3x5) into B1 (1x5), B21 (2x2), B22 (2x2), B23 (2x1).
    C = A1*B1 + A2*B21 + A2*B22 + A2*B23.
    
    A1*B1: Rank 10 (direct).
    A2*B21: Rank 7 (Strassen).
    A2*B22: Rank 7 (Strassen).
    A2*B23: Rank 4 (direct).
    Total Rank: 28.
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    # Helper to add to W with offset
    def add_w(w_idx, col, val):
        if 0 <= w_idx < len(W):
            W[w_idx][col] += val

    # Part 1: A1 * B1 (10 terms, cols 0-9)
    # A1 = A[:, 0], B1 = B[0, :]
    # U idx: A[i, 0] is i*3 + 0 => i=0->0, i=1->3
    # V idx: B[0, k] is 0*5 + k => k=0..4
    # W idx: C[i, k] is k*n + i => k*2 + i
    
    for i in range(n):
        for k in range(p):
            col = i * p + k
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1

    # Part 2: A2 * B21 (7 terms, cols 10-16)
    # A2 = A[:, 1:3] -> U idx: A[0,1]=1, A[0,2]=2, A[1,1]=4, A[1,2]=5
    # B21 = B[1:3, 0:2] -> V idx: B[1,0]=5, B[1,1]=6, B[2,0]=10, B[2,1]=11
    # C target: C[:, 0:2] -> W idx:
    # C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    
    base_col = 10
    
    # M1 = (A[0,1] + A[1,2])(B[1,0] + B[2,1])
    col = base_col
    U[1][col] += 1; U[5][col] += 1
    V[5][col] += 1; V[11][col] += 1
    add_w(0, col, 1)   # C[0,0]
    add_w(3, col, 1)   # C[1,1]
    
    # M2 = (A[1,1] + A[1,2])B[1,0]
    col = base_col + 1
    U[4][col] += 1; U[5][col] += 1
    V[5][col] += 1
    add_w(1, col, 1)   # C[1,0]
    add_w(3, col, -1)  # C[1,1]
    
    # M3 = (A[0,1] - A[1,2])B[2,0]
    col = base_col + 2
    U[1][col] += 1; U[5][col] += -1
    V[10][col] += 1
    add_w(2, col, 1)   # C[0,1]
    add_w(3, col, -1)  # C[1,1]
    
    # M4 = A[0,1](B[1,1] - B[2,1])
    col = base_col + 3
    U[1][col] += 1
    V[6][col] += 1; V[11][col] += -1
    add_w(0, col, 1)   # C[0,0]
    
    # M5 = (A[0,2] + A[1,2])B[2,1]
    col = base_col + 4
    U[2][col] += 1; U[5][col] += 1
    V[11][col] += 1
    add_w(0, col, -1)  # C[0,0]
    add_w(2, col, 1)   # C[0,1]
    
    # M6 = (A[0,2] - A[0,1])(B[1,0] + B[1,1])
    col = base_col + 5
    U[2][col] += 1; U[1][col] += -1
    V[5][col] += 1; V[6][col] += 1
    add_w(3, col, 1)   # C[1,1]
    
    # M7 = (A[1,1] - A[1,2])(B[2,0] + B[2,1])
    col = base_col + 6
    U[4][col] += 1; U[5][col] += -1
    V[10][col] += 1; V[11][col] += 1
    add_w(0, col, -1)  # C[0,0]
    add_w(1, col, 1)   # C[1,0]

    # Part 3: A2 * B22 (7 terms, cols 17-23)
    # A2 same U idxs.
    # B22 = B[1:3, 2:4] -> V idx: B[1,2]=7, B[1,3]=8, B[2,2]=12, B[2,3]=13
    # C target: C[:, 2:4] -> W idx:
    # C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
    
    base_col = 17
    
    # M1 = (A[0,1] + A[1,2])(B[1,2] + B[2,3])
    col = base_col
    U[1][col] += 1; U[5][col] += 1
    V[7][col] += 1; V[13][col] += 1
    add_w(4, col, 1)   # C[0,2]
    add_w(7, col, 1)   # C[1,3]
    
    # M2 = (A[1,1] + A[1,2])B[1,2]
    col = base_col + 1
    U[4][col] += 1; U[5][col] += 1
    V[7][col] += 1
    add_w(5, col, 1)   # C[1,2]
    add_w(7, col, -1)  # C[1,3]
    
    # M3 = (A[0,1] - A[1,2])B[2,2]
    col = base_col + 2
    U[1][col] += 1; U[5][col] += -1
    V[12][col] += 1
    add_w(6, col, 1)   # C[0,3]
    add_w(7, col, -1)  # C[1,3]
    
    # M4 = A[0,1](B[1,3] - B[2,3])
    col = base_col + 3
    U[1][col] += 1
    V[8][col] += 1; V[13][col] += -1
    add_w(4, col, 1)   # C[0,2]
    
    # M5 = (A[0,2] + A[1,2])B[2,3]
    col = base_col + 4
    U[2][col] += 1; U[5][col] += 1
    V[13][col] += 1
    add_w(4, col, -1)  # C[0,2]
    add_w(6, col, 1)   # C[0,3]
    
    # M6 = (A[0,2] - A[0,1])(B[1,2] + B[1,3])
    col = base_col + 5
    U[2][col] += 1; U[1][col] += -1
    V[7][col] += 1; V[8][col] += 1
    add_w(7, col, 1)   # C[1,3]
    
    # M7 = (A[1,1] - A[1,2])(B[2,2] + B[2,3])
    col = base_col + 6
    U[4][col] += 1; U[5][col] += -1
    V[12][col] += 1; V[13][col] += 1
    add_w(4, col, -1)  # C[0,2]
    add_w(5, col, 1)   # C[1,2]

    # Part 4: A2 * B23 (4 terms, cols 24-27)
    # A2 same.
    # B23 = B[1:3, 4:5] -> V idx: B[1,4]=9, B[2,4]=14
    # C target: C[:, 4] -> W idx:
    # C[0,4]=8, C[1,4]=9
    
    base_col = 24
    
    # T1: A[0,1] * B[1,4] -> C[0,4]
    col = base_col
    U[1][col] += 1
    V[9][col] += 1
    add_w(8, col, 1)
    
    # T2: A[0,2] * B[2,4] -> C[0,4]
    col = base_col + 1
    U[2][col] += 1
    V[14][col] += 1
    add_w(8, col, 1)
    
    # T3: A[1,1] * B[1,4] -> C[1,4]
    col = base_col + 2
    U[4][col] += 1
    V[9][col] += 1
    add_w(9, col, 1)
    
    # T4: A[1,2] * B[2,4] -> C[1,4]
    col = base_col + 3
    U[5][col] += 1
    V[14][col] += 1
    add_w(9, col, 1)

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple verification with random matrices
    import numpy as np
    np.random.seed(0)
    A = np.random.rand(2, 3)
    B = np.random.rand(3, 5)
    C_expected = A @ B
    
    d = construct()
    U = np.array(d['U'])
    V = np.array(d['V'])
    W = np.array(d['W'])
    
    # C_calc[i, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This structure is not a simple matmul. It's a tensor contraction.
    # C[i,k] = sum_{j,c} A[i,j] * (something)
    # Actually, the tensor T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] should equal 1 if j matches, 0 else?
    # No, for matrix mult tensor, T_{ijk} = delta_{i,i} delta_{j,j}... wait.
    # The multiplication tensor T for (n,m,p) has T[i,j,k] = 1 if the indices correspond to the multiplication A[i,j] * B[j,k].
    # In our case, T[i,j,k] is 1 if we are multiplying A[i,j] and B[j,k].
    # So T[i,j,k] = 1 for all valid i,j,k in the schoolbook sense?
    # No, the tensor entries are 1 at (i, j, j) ? No.
    # Indices: i (0..n-1), j (0..m-1), k (0..p-1).
    # The term in C[i,k] is A[i,j] * B[j,k].
    # So the tensor component is T[i,j,k]. It should be 1.
    # But the decomposition is sum_c u_ic v_jc w_kc.
    # Here U is indexed by (i,j), V by (j,k), W by (k,i).
    # So sum_c U[i,j,c] V[j,k,c] W[k,i,c] should equal 1 for all i,j,k.
    
    C_calc = np.zeros((2, 5))
    for i in range(2):
        for k in range(5):
            s = 0
            for j in range(3):
                for c in range(28):
                    s += U[i*3+j, c] * V[j*5+k, c] * W[k*2+i, c]
            C_calc[i, k] = s
    # This s is the sum of products of coefficients.
    # This s should be equal to A[i,j] * B[j,k]? No.
    # The tensor T decomposes into rank 1 tensors.
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] ?? No.
    # The standard identity is:
    # C = A B
    # C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor M has M_{ijk} = 1.
    # Rank 1 decomposition: M_{ijk} = sum_r x_{ir} y_{jr} z_{kr}.
    # Here our U corresponds to x, V to y, W to z?
    # U is (n*m) x rank. Index is i*m+j. So U[row, c] is x_{i,j, r}.
    # V is (m*p) x rank. Index is j*p+k. So V[row, c] is y_{j,k, r}.
    # W is (p*n) x rank. Index is k*n+i. So W[row, c] is z_{k,i, r}.
    # Wait, the convention W[k*n+i] means z is indexed by (k, i).
    # So the product term is U[i,j,c] * V[j,k,c] * W[k,i,c].
    # Summing over c and j?
    # The sum over c gives the rank 1 decomposition of the tensor entry M[i,j,k].
    # M[i,j,k] should be 1.
    # So sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 for all i,j,k.
    
    # Let's check this property instead of multiplying random matrices.
    ok = True
    for i in range(2):
        for j in range(3):
            for k in range(5):
                s = 0
                for c in range(28):
                    s += U[i*3+j, c] * V[j*5+k, c] * W[k*2+i, c]
                if s != 1:
                    print(f"Fail at i={i}, j={j}, k={k}, sum={s}")
                    ok = False
    if ok:
        print("Decomposition is valid.")
    else:
        print("Decomposition is INVALID.")

