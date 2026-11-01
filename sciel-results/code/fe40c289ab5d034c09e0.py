"""Matrix multiplication 2x3x4 with a rank-18 decomposition built from Strassen (2x2x2) and rank-2 (2x1x2) blocks."""
import numpy as np

DIMENSIONS = [2, 3, 4]

def _block_strassen_2x2x2():
    # Returns U (4x7), V (4x7), W (4x7) for 2x2x2 multiplication
    # Convention: U indexed by (i,j), V by (j,k), W by (k,i)
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11) B00
    # M3 = A00 (B01-B11)
    # M4 = A11 (B10-B00)
    # M5 = (A00+A01) B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indexing: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    
    # M1
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[0, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # M3
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01
    
    # M6
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11
    
    # M7
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    W[2, 6] = -1 # C10
    
    return U, V, W

def _block_rank2_2x1x2():
    # Returns U (2x2), V (1x2), W (2x2) for 2x1x2 multiplication
    # A is 2x1, B is 1x2, C is 2x2
    # C[i,k] = A[i] * B[k]
    # Rank 2: c=0: A0*B0, c=1: A1*B1? No.
    # C = [A0*B0, A0*B1; A1*B0, A1*B1]
    # Term 1: u=[1,0], v=[1], w=[1,0] -> gives A0*B0 at C00
    # Term 2: u=[1,0], v=[1], w=[0,1] -> gives A0*B1 at C01
    # Term 3: u=[0,1], v=[1], w=[1,0] -> gives A1*B0 at C10
    # Term 4: u=[0,1], v=[1], w=[0,1] -> gives A1*B1 at C11
    # That's rank 4. But we can do better.
    # C = [A0; A1] * [B0, B1]
    # This is an outer product of column vector [A0,A1] and row vector [B0,B1].
    # Rank of outer product is 1? No, the tensor is T_{i,k} = A_i B_k.
    # Tensor T has rank 2: T = A0*B0 * (e0 e0^T) + A0*B1 * (e0 e1^T) + A1*B0 * (e1 e0^T) + A1*B1 * (e1 e1^T)
    # Wait, the tensor for 2x1x2 is T_{i,0,k} = delta(i,0)*delta(0,0)*... no.
    # The multiplication tensor T_{i,j,k} where j is the only index in the middle (size 1).
    # T_{i,0,k} = 1 if we're computing C_{i,k} = sum_j A_{i,j} B_{j,k} = A_{i,0} B_{0,k}.
    # So the tensor is T = e_i e_0^T e_k^T summed? No.
    # T_{i,j,k} = 1 if j=0, else 0. So T = (sum_i e_i) e_0^T (sum_k e_k) = (e_0+e_1) e_0^T (e_0+e_1).
    # This is rank 1!
    # U = [1, 1]^T (2x1), V = [1] (1x1), W = [1, 1]^T (2x1)
    # Check: sum_c U_i,c V_j,c W_k,c = U_i0 V_00 W_k0 = 1*1*1 = 1 for all i,k.
    # Yes! Rank 1 for 2x1x2.
    
    U = np.ones((2, 1), dtype=float)
    V = np.ones((1, 1), dtype=float)
    W = np.ones((2, 1), dtype=float)
    return U, V, W

def construct():
    # 2x3x4 decomposition:
    # Split the 3 middle indices into j1 in {0,1} and j2 = 2.
    # Split the 4 last indices into k1 in {0,1} and k2 in {2,3}.
    # This gives 4 blocks:
    # Block (j1, k1): 2x2x2 -> Strassen rank 7
    # Block (j1, k2): 2x2x2 -> Strassen rank 7
    # Block (j2, k1): 2x1x2 -> rank 1
    # Block (j2, k2): 2x1x2 -> rank 1
    # Total rank = 7 + 7 + 1 + 1 = 16
    
    # Wait, let me re-examine the block structure.
    # A is 2x3: A[i, j] where i in {0,1}, j in {0,1,2}
    # B is 3x4: B[j, k] where j in {0,1,2}, k in {0,1,2,3}
    # C is 2x4: C[i, k] = sum_{j=0}^{2} A[i,j] B[j,k]
    
    # Split j into {0,1} and {2}, k into {0,1} and {2,3}.
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]
    
    # For k in {0,1}:
    #   C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]
    #   The first two terms form a 2x2x2 subproblem (i in {0,1}, j in {0,1}, k in {0,1})
    #   The third term is A[i,2]*B[2,k], a 2x1x2 subproblem (i in {0,1}, j={2}, k in {0,1})
    # For k in {2,3}:
    #   C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]
    #   Similarly, 2x2x2 + 2x1x2
    
    # So:
    # Block 1 (j in {0,1}, k in {0,1}): 2x2x2, Strassen rank 7
    # Block 2 (j in {0,1}, k in {2,3}): 2x2x2, Strassen rank 7
    # Block 3 (j={2}, k in {0,1}): 2x1x2, rank 1
    # Block 4 (j={2}, k in {2,3}): 2x1x2, rank 1
    # Total: 7 + 7 + 1 + 1 = 16
    
    n, m, p = 2, 3, 4
    total_rank = 16
    
    U = np.zeros((n*m, total_rank), dtype=float)  # 6x16
    V = np.zeros((m*p, total_rank), dtype=float)  # 12x16
    W = np.zeros((p*n, total_rank), dtype=float)  # 8x16
    
    # Block 1: j in {0,1}, k in {0,1} -> 2x2x2 Strassen
    U1, V1, W1 = _block_strassen_2x2x2()
    # U1: 4x7, indices (i,j1) where j1 in {0,1} -> map to (i, j) where j=j1
    # V1: 4x7, indices (j1,k1) where j1 in {0,1}, k1 in {0,1} -> map to (j, k) where j=j1, k=k1
    # W1: 4x7, indices (k1,i) where k1 in {0,1} -> map to (k, i) where k=k1
    
    # U mapping: row i*m+j = i*3+j, j in {0,1}
    # V mapping: row j*p+k = j*4+k, j in {0,1}, k in {0,1}
    # W mapping: row k*n+i = k*2+i, k in {0,1}
    
    for c in range(7):
        for i in range(2):
            for j in range(2):
                u_idx = i * 2 + j
                U[i * 3 + j, c] = U1[u_idx, c]
        for j in range(2):
            for k in range(2):
                v_idx = j * 2 + k
                V[j * 4 + k, c] = V1[v_idx, c]
        for k in range(2):
            for i in range(2):
                w_idx = k * 2 + i
                W[k * 2 + i, c] = W1[w_idx, c]
    
    # Block 2: j in {0,1}, k in {2,3} -> 2x2x2 Strassen
    # k2 in {0,1} maps to k = k2 + 2
    for c in range(7, 14):
        c_local = c - 7
        for i in range(2):
            for j in range(2):
                u_idx = i * 2 + j
                U[i * 3 + j, c] = U1[u_idx, c_local]
        for j in range(2):
            for k2 in range(2):
                v_idx = j * 2 + k2
                V[j * 4 + (k2 + 2), c] = V1[v_idx, c_local]
        for k2 in range(2):
            for i in range(2):
                w_idx = k2 * 2 + i
                W[(k2 + 2) * 2 + i, c] = W1[w_idx, c_local]
    
    # Block 3: j={2}, k in {0,1} -> 2x1x2 rank 1
    U3, V3, W3 = _block_rank2_2x1x2()
    # U3: 2x1, index i -> U[i*3+2, 14]
    # V3: 1x1, index j=2 -> V[2*4+k, 14] for k in {0,1}? No, V is indexed by (j,k).
    # For 2x1x2: U is 2x1 (i index), V is 1x1 (j index), W is 2x1 (k index)
    # But V should be indexed by (j,k) in the full problem. Here j=2, and k varies.
    # Actually, in the 2x1x2 subproblem, B is 1x2, so V should have entries for (j,k) where j=2, k in {0,1}.
    # Let me redo: for the 2x1x2 block with j=2, k in {0,1}:
    # T_{i,2,k} = 1 for all i,k in the subproblem.
    # So U[i, c] = 1 for i in {0,1}, V[2*4+k, c] = 1 for k in {0,1}, W[k*2+i, c] = 1 for k in {0,1}, i in {0,1}.
    
    c3 = 14
    for i in range(2):
        U[i * 3 + 2, c3] = 1
    for k in range(2):
        V[2 * 4 + k, c3] = 1
    for k in range(2):
        for i in range(2):
            W[k * 2 + i, c3] = 1
    
    # Block 4: j={2}, k in {2,3} -> 2x1x2 rank 1
    c4 = 15
    for i in range(2):
        U[i * 3 + 2, c4] = 1
    for k2 in range(2):
        V[2 * 4 + (k2 + 2), c4] = 1
    for k2 in range(2):
        for i in range(2):
            W[(k2 + 2) * 2 + i, c4] = 1
    
    # Convert to lists as required
    U_list = [[float(U[r, c]) for c in range(total_rank)] for r in range(n * m)]
    V_list = [[float(V[r, c]) for c in range(total_rank)] for r in range(m * p)]
    W_list = [[float(W[r, c]) for c in range(total_rank)] for r in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
