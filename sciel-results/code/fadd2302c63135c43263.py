"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses structured decomposition: Strassen for 2x2 blocks + schoolbook for 3rd row/col.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin before 120s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    def schoolbook_decomp():
        """Return the standard schoolbook decomposition of rank 72."""
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
    
    def strassen_2x2_decomp():
        """
        Return Strassen decomposition for 2x2 * 2x2 as (U, V, W) each of shape (4, 7).
        Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        # Strassen's 7 products:
        # M1 = (A11-A22)(B21+B22)
        # M2 = (A11+A22)B22
        # M3 = A11(B11-B22)
        # M4 = A22(B11+B12)
        # M5 = (A11+A12)B11
        # M6 = (A21-A22)(B11+B12)
        # M7 = (A11+A12)(B21-B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each term c, we need U[i*2+j,c], V[j*2+k,c], W[k*2+i,c]
        # such that sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] = 1 for all i,j,k in {0,1}
        
        # Let's define the 7 rank-1 tensors explicitly.
        # Each term contributes to specific (i,j,k) positions.
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Term 0: M1 = (A11-A22)(B21+B22)
        # A part: A11 - A22 => U[0*2+0]=1, U[1*2+1]=-1
        # B part: B21 + B22 => V[1*2+0]=1, V[1*2+1]=1
        # C part: contributes to C11 and C22
        # C11 = M1 => W[0*2+0] += 1 (i=0,k=0)
        # C22 = M1 => W[1*2+1] += 1 (i=1,k=1)
        U[0][0] = 1; U[3][0] = -1  # A[0,0]=1, A[1,1]=-1
        V[2][0] = 1; V[3][0] = 1   # B[1,0]=1, B[1,1]=1
        W[0][0] = 1; W[3][0] = 1   # C[0,0]=1, C[1,1]=1
        
        # Term 1: M2 = (A11+A22)B22
        # A part: A11 + A22 => U[0]=1, U[3]=1
        # B part: B22 => V[3]=1
        # C part: C21 = M2 => W[2*2+1] = W[5] += 1 (i=1,k=0)
        # Wait, C21 means i=1, k=0. W[k*n+i] = W[0*2+1] = W[1]
        U[0][1] = 1; U[3][1] = 1
        V[3][1] = 1
        W[1][1] = 1  # C[1,0]
        
        # Term 2: M3 = A11(B11-B22)
        # A part: A11 => U[0]=1
        # B part: B11 - B22 => V[0]=1, V[3]=-1
        # C part: C12 = M3 => W[0*2+1] = W[1] += 1 (i=0,k=1)
        # C22 = M3 => W[1*2+1] = W[3] += 1 (i=1,k=1)
        U[0][2] = 1
        V[0][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # Term 3: M4 = A22(B11+B12)
        # A part: A22 => U[3]=1
        # B part: B11 + B12 => V[0]=1, V[1]=1
        # C part: C11 = M4 => W[0] += 1
        # C21 = M4 => W[1] += 1
        U[3][3] = 1
        V[0][3] = 1; V[1][3] = 1
        W[0][3] = 1; W[1][3] = 1
        
        # Term 4: M5 = (A11+A12)B11
        # A part: A11 + A12 => U[0]=1, U[1]=1
        # B part: B11 => V[0]=1
        # C part: C11 = -M5 => W[0] += -1
        # C12 = M5 => W[1] += 1
        U[0][4] = 1; U[1][4] = 1
        V[0][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # Term 5: M6 = (A21-A22)(B11+B12)
        # A part: A21 - A22 => U[2]=1, U[3]=-1
        # B part: B11 + B12 => V[0]=1, V[1]=1
        # C part: C22 = M6 => W[3] += 1
        U[2][5] = 1; U[3][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # Term 6: M7 = (A11+A12)(B21-B22)
        # A part: A11 + A12 => U[0]=1, U[1]=1
        # B part: B21 - B22 => V[2]=1, V[3]=-1
        # C part: C11 = M7 => W[0] += 1
        U[0][6] = 1; U[1][6] = 1
        V[2][6] = 1; V[3][6] = -1
        W[0][6] = 1
        
        return U, V, W
    
    def structured_decomp():
        """
        Construct a structured decomposition for 3x3 * 3x8.
        
        Strategy:
        - Split A into A_top (2x3) and A_bottom (1x3)
        - Split B into B_left (3x2) and B_right (3x6)
        
        For the 2x2 block (A_top_left 2x2, B_left 2x2), use Strassen (rank 7)
        For other blocks, use schoolbook.
        
        Actually, let me think more carefully. The tensor is:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        We can decompose this as:
        C = C_top + C_bottom
        where C_top[i,k] = sum_{j=0,1} A[i,j]*B[j,k] for i=0,1
        and C_bottom[i,k] = sum_j A[2,j]*B[j,k] for i=2
        
        For C_top: this is a 2x3 * 3x8 multiplication.
        We can further split:
        C_top[:, 0:2] involves B[:, 0:2]
        C_top[:, 2:8] involves B[:, 2:8]
        
        This is getting complex. Let me try a different approach:
        
        Use the fact that we can write the 3x3*3x8 tensor as a sum of:
        - 2x2*2x2 blocks using Strassen
        - Cross terms using schoolbook
        
        Let me define the decomposition by specifying which (i,j,k) tuples
        are handled by which rank-1 terms.
        
        Actually, the cleanest approach: enumerate all 72 (i,j,k) tuples and
        group them into Strassen patterns where possible.
        
        For the 2x2*2x2 subproblem on indices {0,1}x{0,1}x{0,1}, we can
        use Strassen's 7 terms instead of 8 schoolbook terms.
        
        Similarly for {0,1}x{0,1}x{2,3}, {0,1}x{0,1}x{4,5}, {0,1}x{0,1}x{6,7}.
        
        Each 2x2*2x2 block saves 1 rank (8 -> 7).
        We have 4 such blocks in the j,k dimensions for i in {0,1}.
        
        For i=2, we still need schoolbook: 3*8 = 24 terms.
        
        Total: 4*7 + 24 = 28 + 24 = 52.
        
        Wait, that's not right either. Let me reconsider.
        
        The full schoolbook has 3*3*8 = 72 terms.
        
        If we use Strassen for the 2x2*2x2 sub-blocks:
        - For each fixed i in {0,1} and each pair of k values {2a, 2a+1},
          we have a 2x2*2x2 problem in (j, k) space.
        
        There are 2 choices for i, 4 pairs of k values, giving 8 such blocks.
        Each saves 1 rank, so total saving is 8.
        New rank: 72 - 8 = 64.
        
        Let me implement this.
        """
        U = [[0] * 64 for _ in range(9)]
        V = [[0] * 64 for _ in range(24)]
        W = [[0] * 64 for _ in range(24)]
        
        col = 0
        # Process i in {0, 1} with Strassen for each k-pair
        for i in [0, 1]:
            for k_pair in range(4):
                k0 = 2 * k_pair
                k1 = 2 * k_pair + 1
                # Get Strassen decomposition for this block
                U_s, V_s, W_s = strassen_2x2_decomp()
                # U_s has shape (4, 7), indexed by [j_local * 2 + k_local]
                # But we need U[i*3 + j, col], V[j*8 + k, col], W[k*3 + i, col]
                # For this block, j is in {0,1}, k is in {k0, k1}
                for c in range(7):
                    # U part: U_s[j*2 + k_local] where k_local is 0 or 1
                    # But U only depends on i and j, not k
                    # U_s[0] = U_s[0*2+0] = A[0,0] coefficient
                    # U_s[1] = U_s[0*2+1] = A[0,1] coefficient
                    # U_s[2] = U_s[1*2+0] = A[1,0] coefficient
                    # U_s[3] = U_s[1*2+1] = A[1,1] coefficient
                    for j_local in range(2):
                        for k_local in range(2):
                            u_val = U_s[j_local * 2 + k_local][c]
                            if u_val != 0:
                                U[i * 3 + j_local][col] += u_val
                    # V part: V_s[j*2 + k_local]
                    for j_local in range(2):
                        for k_local in range(2):
                            v_val = V_s[j_local * 2 + k_local][c]
                            if v_val != 0:
                                j = j_local
                                k = k0 + k_local
                                V[j * 8 + k][col] += v_val
                    # W part: W_s[k*2 + i_local]
                    for k_local in range(2):
                        for i_local in range(2):
                            w_val = W_s[k_local * 2 + i_local][c]
                            if w_val != 0:
                                k = k0 + k_local
                                i = i_local
                                W[k * 3 + i][col] += w_val
                    col += 1
        
        # Process i = 2 with schoolbook
        for j in range(3):
            for k in range(8):
                U[2 * 3 + j][col] = 1
                V[j * 8 + k][col] = 1
                W[k * 3 + 2][col] = 1
                col += 1
        
        # Should have used exactly 64 columns
        assert col == 64, f"Expected 64, got {col}"
        
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify the tensor decomposition."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vec = U_arr[i*m+j, :]
                    v_vec = V_arr[j*p+k, :]
                    w_vec = W_arr[k*n+i, :]
                    val = np.dot(np.dot(u_vec, v_vec), w_vec)
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try the structured decomposition first
    try:
        result = structured_decomp()
        
        # Verify if time permits
        if time.time() < deadline - 5:
            is_valid = verify_decomposition(result["U"], result["V"], result["W"])
            if not is_valid:
                raise ValueError("Structured decomposition failed verification")
        
        return result
        
    except Exception as e:
        # Fall back to schoolbook
        result = schoolbook_decomp()
        return result
# EVOLVE-BLOCK-END
