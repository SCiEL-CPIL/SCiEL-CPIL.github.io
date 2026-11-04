import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def _get_strassen_222():
    # Returns U, V, W for 2x2x2 multiplication
    # A is 2x2, B is 2x2, C is 2x2
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = -1
    
    return U, V, W

def _get_simple_221():
    # 2x2x1. A 2x2, B 2x1, C 2x1
    # U (4, 4), V (2, 4), W (2, 4)
    # C0 = a*e + b*g
    # C1 = c*e + d*g
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # Term 1: a*e -> C0
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # Term 2: b*g -> C0
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # Term 3: c*e -> C1
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # Term 4: d*g -> C1
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def _get_simple_212():
    # 2x1x2. A 2x1, B 1x2, C 2x2
    # U (2, 4), V (2, 4), W (4, 4)
    # C00 = a*e, C01 = a*f
    # C10 = c*e, C11 = c*f
    U = np.zeros((2, 4))
    V = np.zeros((2, 4))
    W = np.zeros((4, 4))
    
    # Term 1: a*e -> C00 (idx 0)
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # Term 2: a*f -> C01 (idx 1)
    U[0, 1] = 1
    V[1, 1] = 1
    W[1, 1] = 1
    
    # Term 3: c*e -> C10 (idx 2)
    U[1, 2] = 1
    V[0, 2] = 1
    W[2, 2] = 1
    
    # Term 4: c*f -> C11 (idx 3)
    U[1, 3] = 1
    V[1, 3] = 1
    W[3, 3] = 1
    
    return U, V, W

def _get_decomp_233():
    # 2x3x3 using 222, 221, 212, 211
    # Split 3 into 2+1
    # A: [A2, A1] (2x2, 2x1)
    # B: [[B22, B21], [B12, B11]] (2x2, 2x1, 1x2, 1x1)
    # C: [C2, C1] (2x2, 2x1)
    # C2 = A2 B22 + A1 B12
    # C1 = A2 B21 + A1 B11
    
    # Sub-blocks:
    # 1. A2 B22 (2x2x2) -> Rank 7
    # 2. A1 B12 (2x1x2) -> Rank 4
    # 3. A2 B21 (2x2x1) -> Rank 4
    # 4. A1 B11 (2x1x1) -> Rank 2
    
    U_list = []
    V_list = []
    W_list = []
    
    # 1. 2x2x2 (j=0,1; k=0,1)
    U1, V1, W1 = _get_strassen_222()
    # A indices: 0,1,2,3 (j=0,1)
    # B indices: 0,1,2,3 (j=0,1; k=0,1)
    # C indices: 0,1,2,3 (k=0,1)
    U_list.append(U1)
    V_list.append(V1)
    W_list.append(W1)
    
    # 2. 2x1x2 (j=2; k=0,1)
    # A indices: 4,5 (j=2)
    # B indices: 6,7 (j=2; k=0,1)
    # C indices: 4,5 (k=0,1)
    U2, V2, W2 = _get_simple_212()
    # Pad U to (6, 4) ? No, we will handle offsets in assembly.
    # For now, store local matrices.
    U_list.append(U2)
    V_list.append(V2)
    W_list.append(W2)
    
    # 3. 2x2x1 (j=0,1; k=2)
    # A indices: 0,1,2,3
    # B indices: 4,5 (j=0,1; k=2)
    # C indices: 6,7 (k=2)
    U3, V3, W3 = _get_simple_221()
    U_list.append(U3)
    V_list.append(V3)
    W_list.append(W3)
    
    # 4. 2x1x1 (j=2; k=2)
    # A indices: 4,5
    # B indices: 8 (j=2; k=2)
    # C indices: 8,9 (k=2)
    # 2x1x1 is two 1x1x1? No, rank 2.
    # C0 = a*e, C1 = c*e
    U4 = np.array([[1, 0], [0, 1]]) # 2x2
    V4 = np.array([[1], [1]]) # 1x2 ? No, V is (1, 2)
    # V shape (m*p, r) = (1, 2)
    V4 = np.array([[1, 1]])
    W4 = np.array([[1, 0], [0, 1]]) # 2x2 (C0, C1)
    U_list.append(U4)
    V_list.append(V4)
    W_list.append(W4)
    
    # Assembly for 2x3x3
    n, m, p = 2, 3, 3
    total_rank = sum(U.shape[1] for U in U_list)
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    col_offset = 0
    # 1. 2x2x2 (j 0-1, k 0-1)
    U_full[0:4, col_offset:col_offset+7] = U1
    V_full[0:4, col_offset:col_offset+7] = V1
    W_full[0:4, col_offset:col_offset+7] = W1
    col_offset += 7
    
    # 2. 2x1x2 (j 2, k 0-1)
    # A rows: 4,5
    # B rows: 6,7 (j=2, k=0,1 -> 2*3+0=6, 2*3+1=7)
    # C rows: 4,5 (k=0,1 -> 0*2+0=0? No. C index k*n+i. k=0 -> 0,1. k=1 -> 2,3. k=2 -> 4,5.
    # Wait, C for 2x3x3 has size 2x3=6.
    # k=0: C00, C10 -> 0, 1
    # k=1: C01, C11 -> 2, 3
    # k=2: C02, C12 -> 4, 5
    # So W1 (for k 0,1) maps to 0,1,2,3.
    # W2 (for k 0,1) maps to 0,1,2,3.
    # W3 (for k 2) maps to 4,5.
    # W4 (for k 2) maps to 4,5.
    
    U_full[4:6, col_offset:col_offset+4] = U2
    V_full[6:8, col_offset:col_offset+4] = V2
    W_full[0:4, col_offset:col_offset+4] = W2
    col_offset += 4
    
    # 3. 2x2x1 (j 0-1, k 2)
    # A rows: 0-3
    # B rows: 4,5 (j=0,1 k=2 -> 0*3+2=2? No. B index j*p+k. p=3.
    # j=0, k=2 -> 2. j=1, k=2 -> 5.
    # So B rows 2, 5?
    # V3 is 2x2x1. B is 2x1. Indices 0,1 in local.
    # Global B indices: j=0,k=2 -> 2. j=1,k=2 -> 5.
    # So V_full[2, col] and V_full[5, col].
    # My V3 from _get_simple_221 has shape (2, 4). Rows 0,1.
    # Map local 0 -> global 2, local 1 -> global 5.
    U_full[0:4, col_offset:col_offset+4] = U3
    V_full[2, col_offset:col_offset+4] = V3[0]
    V_full[5, col_offset:col_offset+4] = V3[1]
    W_full[4:6, col_offset:col_offset+4] = W3
    col_offset += 4
    
    # 4. 2x1x1 (j 2, k 2)
    # A rows: 4,5
    # B row: 8 (j=2, k=2 -> 2*3+2=8)
    # C rows: 4,5
    U_full[4:6, col_offset:col_offset+2] = U4
    V_full[8, col_offset:col_offset+2] = V4
    W_full[4:6, col_offset:col_offset+2] = W4
    col_offset += 2
    
    return U_full, V_full, W_full

def _get_decomp_223():
    # 2x2x3. Split p=3 into 2+1.
    # C = [C2, C1]
    # C2 = A B2 (2x2x2)
    # C1 = A B1 (2x2x1)
    n, m, p = 2, 2, 3
    U1, V1, W1 = _get_strassen_222()
    U2, V2, W2 = _get_simple_221()
    
    total_rank = 7 + 4
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    # 1. 2x2x2 (k 0-1)
    U_full[:, 0:7] = U1
    V_full[:, 0:7] = V1
    # W1 maps to C k=0,1. Indices 0,1,2,3.
    W_full[:, 0:7] = W1
    
    # 2. 2x2x1 (k 2)
    U_full[:, 7:11] = U2
    # V2 is 2x4. B is 2x1. Local indices 0,1.
    # Global B indices: j=0,k=2 -> 2. j=1,k=2 -> 5.
    V_full[2, 7:11] = V2[0]
    V_full[5, 7:11] = V2[1]
    # W2 maps to C k=2. Indices 4,5.
    W_full[4:6, 7:11] = W2
    
    return U_full, V_full, W_full

def _get_decomp_232():
    # 2x3x2. Split m=3 into 2+1.
    # C = A2 B2 + A1 B1
    # C2 = A2 B2 (2x2x2)
    # C1 = A1 B1 (2x1x2)
    n, m, p = 2, 3, 2
    U1, V1, W1 = _get_strassen_222()
    U2, V2, W2 = _get_simple_212()
    
    total_rank = 7 + 4
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    # 1. 2x2x2 (j 0-1)
    U_full[0:4, 0:7] = U1
    # V1 is 4x7. B is 2x2. Local 0,1,2,3.
    # Global B: j=0,k=0->0, j=0,k=1->1, j=1,k=0->2, j=1,k=1->3.
    V_full[0:4, 0:7] = V1
    W_full[:, 0:7] = W1
    
    # 2. 2x1x2 (j 2)
    U_full[4:6, 7:11] = U2
    # V2 is 2x4. B is 1x2. Local 0,1.
    # Global B: j=2,k=0->4, j=2,k=1->5.
    V_full[4, 7:11] = V2[0]
    V_full[5, 7:11] = V2[1]
    W_full[:, 7:11] = W2
    
    return U_full, V_full, W_full

def construct():
    n, m, p = DIMENSIONS
    # 2x5x5
    # Split m=5 into 3+2, p=5 into 3+2
    U1, V1, W1 = _get_decomp_233() # A3 B33
    U2, V2, W2 = _get_decomp_223() # A2 B23 (A is 2x2, B is 2x3)
    U3, V3, W3 = _get_decomp_232() # A3 B32 (A is 2x3, B is 3x2)
    U4, V4, W4 = _get_strassen_222() # A2 B22
    
    total_rank = U1.shape[1] + U2.shape[1] + U3.shape[1] + U4.shape[1]
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    # 1. A3 B33 (j 0-2, k 0-2)
    U[0:6, col:col+U1.shape[1]] = U1
    V[0:9, col:col+V1.shape[1]] = V1
    W[0:6, col:col+W1.shape[1]] = W1
    col += U1.shape[1]
    
    # 2. A2 B23 (j 3-4, k 0-2)
    # A rows: 6,7,8,9
    # B rows: j=3,4; k=0,1,2.
    # j=3: 3*5+0=15, 16, 17.
    # j=4: 4*5+0=20, 21, 22.
    # V2 is 6x11 (2x2x3). Rows 0-5.
    # Map local 0-2 to global 15-17, local 3-5 to global 20-22.
    U[6:10, col:col+U2.shape[1]] = U2
    V[15:18, col:col+V2.shape[1]] = V2[0:3]
    V[20:23, col:col+V2.shape[1]] = V2[3:6]
    W[0:6, col:col+W2.shape[1]] = W2
    col += U2.shape[1]
    
    # 3. A3 B32 (j 0-2, k 3-4)
    # A rows: 0-5
    # B rows: j=0,1,2; k=3,4.
    # j=0: 3,4. j=1: 8,9. j=2: 13,14.
    # V3 is 6x11 (2x3x2). Rows 0-5.
    # Map local 0,1 to 3,4. 2,3 to 8,9. 4,5 to 13,14.
    U[0:6, col:col+U3.shape[1]] = U3
    V[3:5, col:col+V3.shape[1]] = V3[0:2]
    V[8:10, col:col+V3.shape[1]] = V3[2:4]
    V[13:15, col:col+V3.shape[1]] = V3[4:6]
    W[6:10, col:col+W3.shape[1]] = W3
    col += U3.shape[1]
    
    # 4. A2 B22 (j 3-4, k 3-4)
    # A rows: 6-9
    # B rows: j=3,4; k=3,4.
    # j=3: 18,19. j=4: 23,24.
    # V4 is 4x7.
    U[6:10, col:col+U4.shape[1]] = U4
    V[18:20, col:col+V4.shape[1]] = V4[0:2]
    V[23:25, col:col+V4.shape[1]] = V4[2:4]
    W[6:10, col:col+W4.shape[1]] = W4
    col += U4.shape[1]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
