"""Matrix multiplication 2x4x5 rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x5 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Strategy: Split the m=4 dimension into two chunks of size 2.
    # For each chunk of A (2x2) and B (2xp), we decompose the 2x2xp tensor.
    # We further split p=5 into chunks of 2, 2, 1.
    # For each 2x2x2 block, we use Strassen's algorithm (rank 7).
    # For each 2x2x1 block, we use schoolbook (rank 4).
    # 
    # Total rank = 2 chunks * (2 * 7 + 1 * 4) = 2 * (14 + 4) = 36.
    # This is better than schoolbook rank 40.
    
    # Local 2x2x2 Strassen tensors
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # Strassen ranks 0..6
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = A10(A01+B11)  -- Wait, standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Let's define the local tensors for 2x2x2 (rank 7)
    # U is 4x7, V is 4x7, W is 4x7
    # U rows: 00, 01, 10, 11 (i*2+j)
    # V rows: 00, 01, 10, 11 (j*2+k)
    # W rows: 00, 10, 01, 11 (k*2+i)  <-- Note: W index is k*n+i. For n=2, k*2+i.
    # W row 0: k=0, i=0 (C00)
    # W row 1: k=0, i=1 (C10)
    # W row 2: k=1, i=0 (C01)
    # W row 3: k=1, i=1 (C11)
    
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((4, 7))
    W_strassen = np.zeros((4, 7))
    
    # r=0: M1
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[1, 0] = 1
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1
    
    # r=1: M2
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    W_strassen[1, 1] = 1; W_strassen[3, 1] = -1
    
    # r=2: M3
    U_strassen[0, 2] = 1
    V_strassen[2, 2] = 1; V_strassen[3, 2] = -1
    W_strassen[2, 2] = 1; W_strassen[3, 2] = 1
    
    # r=3: M4
    U_strassen[3, 3] = 1
    V_strassen[1, 3] = 1; V_strassen[0, 3] = -1
    W_strassen[0, 3] = 1; W_strassen[1, 3] = 1
    
    # r=4: M5
    U_strassen[0, 4] = 1; U_strassen[1, 4] = 1
    V_strassen[3, 4] = 1
    W_strassen[0, 4] = -1; W_strassen[2, 4] = 1
    
    # r=5: M6
    U_strassen[2, 5] = 1; U_strassen[0, 5] = -1
    V_strassen[0, 5] = 1; V_strassen[2, 5] = 1
    W_strassen[3, 5] = 1
    
    # r=6: M7
    U_strassen[1, 6] = 1; U_strassen[3, 6] = -1
    V_strassen[1, 6] = 1; V_strassen[3, 6] = 1
    W_strassen[0, 6] = 1
    
    # Verify Strassen locally
    # For all i,j,k in 0..1: sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] should be 1
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for r in range(7):
                    total += U_strassen[i*2+j, r] * V[j*2+k, r] * W_strassen[k*2+i, r]
                if abs(total - 1) > 1e-9:
                    # Fallback
                    return _schoolbook(n, m, p)
    
    # Local 2x2x1 Schoolbook tensor (rank 4)
    # U is 4x4, V is 4x4, W is 2x4 (since p=1, W rows are k*2+i = 0, 1)
    # W row 0: C0, W row 1: C1
    U_sb = np.zeros((4, 4))
    V_sb = np.zeros((4, 4))
    W_sb = np.zeros((2, 4))
    
    for j in range(2):
        for k in range(1):
            r = j + k # 0, 1
            # Wait, we need 4 terms for 2x2x1?
            # C[i] = sum_j A[i,j] B[j,0]
            # Terms: A00 B00, A01 B10, A10 B00, A11 B10
            # Rank 4 is standard.
            pass
    
    # Let's just map schoolbook 2x2x1 explicitly
    # r=0: A00 B00 -> C0
    U_sb[0, 0] = 1; V_sb[0, 0] = 1; W_sb[0, 0] = 1
    # r=1: A01 B10 -> C0
    U_sb[1, 1] = 1; V_sb[2, 1] = 1; W_sb[0, 1] = 1
    # r=2: A10 B00 -> C1
    U_sb[2, 2] = 1; V_sb[0, 2] = 1; W_sb[1, 2] = 1
    # r=3: A11 B10 -> C1
    U_sb[3, 3] = 1; V_sb[2, 3] = 1; W_sb[1, 3] = 1
    
    # Verify Schoolbook locally
    for i in range(2):
        for j in range(2):
            for k in range(1):
                total = 0
                for r in range(4):
                    total += U_sb[i*2+j, r] * V_sb[j*2+k, r] * W_sb[k*2+i, r]
                if abs(total - 1) > 1e-9:
                    return _schoolbook(n, m, p)

    # Global Assembly
    # m=4 split into m_chunks of size 2.
    # p=5 split into p_chunks of sizes [2, 2, 1].
    
    m_chunks = [2, 2]
    p_chunks = [2, 2, 1]
    
    total_rank = 0
    for mc in m_chunks:
        for pc in p_chunks:
            if pc == 2:
                total_rank += 7
            else:
                total_rank += 4
    
    # Initialize global matrices
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    offset = 0
    for m_idx, m_size in enumerate(m_chunks):
        # m_idx corresponds to rows m_idx*2 to m_idx*2+1 in A/B
        for p_idx, p_size in enumerate(p_chunks):
            # p_idx corresponds to cols p_idx*? in B/C
            # We need to map global indices to local indices
            
            if p_size == 2:
                rank_block = 7
                U_loc = U_strassen
                V_loc = V_strassen
                W_loc = W_strassen
            else:
                rank_block = 4
                U_loc = U_sb
                V_loc = V_sb
                W_loc = W_sb
            
            # Map local to global
            # Local U index: i*2 + j. Global: i*m + (m_idx*2 + j)
            # Local V index: j*2 + k. Global: (m_idx*2 + j)*p + (p_idx*p_size + k)
            # Local W index: k*2 + i. Global: (p_idx*p_size + k)*n + i
            
            for r in range(rank_block):
                for i in range(2):
                    for j in range(2):
                        g_u_row = i*m + (m_idx*m_size + j)
                        l_u_row = i*2 + j
                        U[g_u_row, offset + r] += U_loc[l_u_row, r]
                
                for j in range(2):
                    for k in range(p_size):
                        g_v_row = (m_idx*m_size + j)*p + (p_idx*p_size + k)
                        l_v_row = j*2 + k
                        V[g_v_row, offset + r] += V_loc[l_v_row, r]
                
                for i in range(2):
                    for k in range(p_size):
                        g_w_row = (p_idx*p_size + k)*n + i
                        l_w_row = k*2 + i
                        W[g_w_row, offset + r] += W_loc[l_w_row, r]
            
            offset += rank_block
    
    # Final Verification
    # Check a few random points or all points (40 points is fast)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(total_rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1) > 1e-9:
                    return _schoolbook(n, m, p)
    
    # Convert to lists of lists for the required format
    # The task says "integer/half-integer entries". Our entries are integers.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition of rank n*m*p."""
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
# EVOLVE-BLOCK-END
