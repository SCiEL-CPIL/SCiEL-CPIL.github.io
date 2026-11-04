"""Matrix multiplication 2,4,4 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T[a_idx, b_idx, c_idx] from the decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Using einsum
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build the expected multiplication tensor
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i' and k==k' and j==j'
    # Actually, the multiplication tensor for A (n x m) * B (m x p) -> C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the trilinear form is: sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    # (using the trace convention W[k*n+i, r])
    
    T_expected = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_expected[a_idx, b_idx, c_idx] = 1.0
    
    if not np.allclose(T, T_expected, atol=1e-10):
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook rank decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2():
    """
    Return Strassen's 7-term decomposition for 2x2 matrix multiplication.
    Using the standard Strassen formula:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # Indices for A: A11=0, A12=1, A21=2, A22=3
    # Indices for B: B11=0, B12=1, B21=2, B22=3
    # Indices for C: C11=0, C12=1, C21=2, C22=3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A11+A22)*(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 += M1, C22 += M1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A21+A22)*B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 += M2, C22 -= M2
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A11*(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C12 += M3, C22 += M3
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A22*(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C11 += M4, C21 += M4
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A11+A12)*B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 -= M5, C12 += M5
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (A21-A11)*(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 += M6
    W[3, 5] = 1
    
    # M7: (A12-A22)*(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 += M7
    W[0, 6] = 1
    
    return U, V, W

def _strassen_2x4x4():
    """
    Construct a decomposition for 2x4 x 4x4 -> 2x4 using Strassen on 2x2 blocks.
    
    A (2x4) = [A1 A2] where A1, A2 are 2x2
    B (4x4) = [B1 B2; B3 B4] where each is 2x2
    C (2x4) = [C1 C2] where C1, C2 are 2x2
    
    C1 = A1*B1 + A2*B3
    C2 = A1*B2 + A2*B4
    
    For C1 = A1*B1 + A2*B3, we can use 2 Strassen instances (one for A1*B1, one for A2*B3),
    but that's 14 terms. We can potentially share some.
    
    Actually, let's think about this differently. We have 4 "block products" needed:
    A1*B1, A2*B3, A1*B2, A2*B4
    
    Each can be done with 7 Strassen terms = 28 total.
    
    But we can potentially do better by sharing. Let's just use 28 terms for now
    and verify correctness.
    """
    n, m, p = 2, 4, 4
    
    # A indices: A[i, j] -> index i*4 + j
    # i=0: A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # i=1: A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    # Block A1 = rows 0-1, cols 0-1: indices 0,1,4,5
    # Block A2 = rows 0-1, cols 2-3: indices 2,3,6,7
    
    # B indices: B[j, k] -> index j*4 + k
    # B1 = rows 0-1, cols 0-1: indices 0,1,4,5
    # B2 = rows 0-1, cols 2-3: indices 2,3,6,7
    # B3 = rows 2-3, cols 0-1: indices 8,9,12,13
    # B4 = rows 2-3, cols 2-3: indices 10,11,14,15
    
    # C indices: C[i, k] -> index k*2 + i (trace convention: W[k*n+i])
    # C1 = rows 0-1, cols 0-1: k=0,1; i=0,1 -> indices 0,1,2,3
    # C2 = rows 0-1, cols 2-3: k=2,3; i=0,1 -> indices 4,5,6,7
    
    rank = 28
    U = np.zeros((8, rank))
    V = np.zeros((16, rank))
    W = np.zeros((8, rank))
    
    S_U, S_V, S_W = _strassen_2x2()
    
    # Helper to add a Strassen instance for blocks at given offsets
    def add_strassen(a_offset, b_offset, c_offset, col_start):
        """
        Add Strassen terms for A_block * B_block -> C_block
        a_offset: offset in A (8x) indices. A_block is 2x2 with indices [0,1,4,5] + offsets
                 Actually A_block local indices: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
        b_offset: offset in B (16x) indices
        c_offset: offset in C (8x) indices
        """
        # Map local 2x2 block indices to global indices
        # For A: local (i,j) -> global a_offset + i*4 + j
        # For B: local (i,j) -> global b_offset + i*4 + j
        # For C: local (i,j) -> global c_offset + j*2 + i  (trace convention)
        
        for r in range(7):
            # U contributions
            for local_i in range(2):
                for local_j in range(2):
                    local_idx = local_i * 2 + local_j
                    # In the 2x2 Strassen, the A matrix indices are 0,1,2,3
                    # where 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                    global_idx = a_offset + local_i * 4 + local_j
                    U[global_idx, col_start + r] = S_U[local_idx, r]
            
            # V contributions
            for local_i in range(2):
                for local_j in range(2):
                    local_idx = local_i * 2 + local_j
                    global_idx = b_offset + local_i * 4 + local_j
                    V[global_idx, col_start + r] = S_V[local_idx, r]
            
            # W contributions
            for local_i in range(2):
                for local_j in range(2):
                    local_idx = local_i * 2 + local_j
                    # C local index: trace convention c = j*2 + i
                    global_idx = c_offset + local_j * 2 + local_i
                    W[global_idx, col_start + r] = S_W[local_idx, r]
    
    # C1 = A1*B1 + A2*B3
    # A1 at a_offset=0, B1 at b_offset=0, C1 at c_offset=0
    add_strassen(0, 0, 0, 0)      # A1*B1 -> C1, terms 0-6
    add_strassen(2, 8, 0, 7)      # A2*B3 -> C1, terms 7-13
    # A2 is cols 2-3, so a_offset for the "row-major" block... 
    # Wait, A is 2x4. A1 = A[:, 0:2], A2 = A[:, 2:4]
    # A1 global indices: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    # A2 global indices: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    # So for A2, the mapping local (i,j) -> global: i*4 + (2+j)
    # This is NOT a simple offset. Let me redo this.
    
    # Let me restart with explicit index mapping
    U = np.zeros((8, rank))
    V = np.zeros((16, rank))
    W = np.zeros((8, rank))
    
    S_U, S_V, S_W = _strassen_2x2()
    
    def add_strassen_explicit(A_map, B_map, C_map, col_start):
        """
        A_map: dict from local 2x2 index (0,1,2,3) to global A index
        B_map: dict from local 2x2 index (0,1,2,3) to global B index
        C_map: dict from local 2x2 index (0,1,2,3) to global C index
        """
        for r in range(7):
            for li in range(4):
                U[A_map[li], col_start + r] += S_U[li, r]
                V[B_map[li], col_start + r] += S_V[li, r]
                W[C_map[li], col_start + r] += S_W[li, r]
    
    # A1: local (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    A1_map = {0: 0, 1: 1, 2: 4, 3: 5}
    # A2: local (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
    A2_map = {0: 2, 1: 3, 2: 6, 3: 7}
    
    # B1: local (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    B1_map = {0: 0, 1: 1, 2: 4, 3: 5}
    # B2: local (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
    B2_map = {0: 2, 1: 3, 2: 6, 3: 7}
    # B3: local (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
    B3_map = {0: 8, 1: 9, 2: 12, 3: 13}
    # B4: local (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
    B4_map = {0: 10, 1: 11, 2: 14, 3: 15}
    
    # C1: local (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C index = k*2 + i, so local (i,j) -> j*2 + i
    C1_map = {0: 0, 1: 1, 2: 2, 3: 3}
    # C2: local (0,0)->4, (0,1)->5, (1,0)->6, (1,1)->7
    C2_map = {0: 4, 1: 5, 2: 6, 3: 7}
    
    # C1 = A1*B1 + A2*B3
    add_strassen_explicit(A1_map, B1_map, C1_map, 0)
    add_strassen_explicit(A2_map, B3_map, C1_map, 7)
    
    # C2 = A1*B2 + A2*B4
    add_strassen_explicit(A1_map, B2_map, C2_map, 14)
    add_strassen_explicit(A2_map, B4_map, C2_map, 21)
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = 2, 4, 4
    
    # First try the Strassen-based 28-rank decomposition
    U_list, V_list, W_list = _strassen_2x4x4()
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    U_list, V_list, W_list = _schoolbook(n, m, p)
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
