import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _time_remaining():
    return _DEADLINE - time.time()

def _build_rank19_block():
    """
    Build rank-19 decomposition for 2x2 * 2x7 using:
    - Rank-11 Laderman/Winograd for columns 0-3 (4 columns)
    - Rank-8 optimal for columns 4-6 (3 columns)
    Total: 19
    """
    n, m, p = 2, 2, 7
    rank = 19
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # --- Block 1: Columns 0-3, rank 11 ---
    # Winograd FWT for 2x2 * 2x4
    # Using the standard Winograd transform for m=2, p=4
    # U factors (4 entries each for A = [[a00,a01],[a10,a11]])
    # V factors (8 entries each for B = [[b00,b01,b02,b03],[b10,b11,b12,b13]])
    # W factors (8 entries each for C = [[c00,c01,c02,c03],[c10,c11,c12,c13]])
    
    # Winograd FWT for 2x2*2x4: rank 11
    # This uses the structure:
    # U columns: (a00+a11), (a10+a11), a00, a11, (a00+a01), (a10-a00), (a01-a11), 
    #            plus 4 more for the 4-column case
    # Actually let me use a known rank-11 decomposition for 2x2*2x4
    
    # Known rank-11 for 2x2*2x4 (from Winograd):
    # Let's construct it explicitly.
    
    # For 2x2 * 2x4, we can use:
    # 3 Strassen blocks would give 7+7=14, but Winograd gives 11.
    
    # Winograd FWT approach for 2x2 * 2x4:
    # Transform A (2x2) with matrix M_A (11x4)
    # Transform B (2x4) with matrix M_B (11x8) 
    # Transform C (2x4) with matrix M_C (11x8)
    
    # Let me use a direct construction:
    # Columns 0,1: Strassen (rank 7)
    # Columns 2,3: Strassen (rank 7)
    # But that's rank 14, not 11.
    
    # Actually, for 2x2*2x4, the known minimum rank is 11.
    # Let me construct it using the Winograd transform.
    
    # Winograd FWT for 2x2*2x4:
    # The transform matrices are:
    
    # For A (2x2 = 4 elements):
    # U_A[i, j] where i is the transform index (0..10), j is the A element index (0..3)
    # A = [a00, a01, a10, a11]
    
    # For B (2x4 = 8 elements):
    # V_B[i, j] where i is the transform index (0..10), j is the B element index (0..7)
    # B = [b00, b01, b02, b03, b10, b11, b12, b13]
    
    # For C (2x4 = 8 elements):
    # W_C[i, j] where i is the transform index (0..10), j is the C element index (0..7)
    # C = [c00, c01, c02, c03, c10, c11, c12, c13]
    
    # The Winograd FWT for 2x2*2x4 uses the following structure:
    # We split B into two 2x2 blocks: B_left = B[:,0:2], B_right = B[:,2:4]
    # Use Strassen on each, but share some terms.
    
    # Actually, let me just use the known rank-11 construction:
    # It's based on the fact that 2x2*2x4 can be done with 11 multiplications.
    
    # Construction:
    # M1 = (a00+a11) * (b00+b11)
    # M2 = (a10+a11) * b00
    # M3 = a00 * (b01-b11)
    # M4 = a11 * (b10-b00)
    # M5 = (a00+a01) * b11
    # M6 = (a10-a00) * (b00+b01)
    # M7 = (a01-a11) * (b10+b11)
    # These 7 handle columns 0,1.
    
    # For columns 2,3, we need 4 more (not 7) due to sharing:
    # M8 = a00 * b02
    # M9 = a01 * b12
    # M10 = a10 * b03
    # M11 = a11 * b13
    # Wait, that's schoolbook for 2 columns = rank 4, total = 11.
    
    # But we need to verify this gives correct C for all 4 columns.
    
    # Let me be more careful. For 2x2 * 2x4:
    # C[0,0] = a00*b00 + a01*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[0,2] = a00*b02 + a01*b12
    # C[0,3] = a00*b03 + a01*b13
    # C[1,0] = a10*b00 + a11*b10
    # C[1,1] = a10*b01 + a11*b11
    # C[1,2] = a10*b02 + a11*b12
    # C[1,3] = a10*b03 + a11*b13
    
    # Using Strassen for cols 0,1 (rank 7):
    # M1 = (a00+a11)*(b00+b11) -> contributes to C[0,0], C[0,1], C[1,0], C[1,1]
    # M2 = (a10+a11)*b00 -> contributes to C[1,0], C[1,1]
    # M3 = a00*(b01-b11) -> contributes to C[0,1]
    # M4 = a11*(b10-b00) -> contributes to C[1,0]
    # M5 = (a00+a01)*b11 -> contributes to C[0,1], C[1,1]
    # M6 = (a10-a00)*(b00+b01) -> contributes to C[1,1]
    # M7 = (a01-a11)*(b10+b11) -> contributes to C[0,0]
    
    # For cols 2,3, use schoolbook (rank 4):
    # M8 = a00*b02 -> C[0,2]
    # M9 = a01*b12 -> C[0,2]
    # M10 = a10*b02 -> C[1,2]
    # M11 = a11*b12 -> C[1,2]
    # But that only handles col 2. For col 3 we need 4 more = rank 15 total.
    
    # Hmm, that doesn't work. Let me reconsider.
    
    # Actually, the rank of 2x2*2x4 is known to be 11.
    # Let me use a different approach: use Winograd FWT properly.
    
    # Winograd FWT for 2x2*2x4:
    # The key insight is that we can use a 4-point transform on the B dimension.
    
    # Let me just use the block approach: Strassen(2x2) + Strassen(2x2) = rank 14
    # and accept rank 14 for the 4-column block, giving total rank 14+8=22.
    # Or Strassen(2x2) + schoolbook(2x2) = 7+4 = 11 for 4 columns!
    
    # Yes! Strassen for cols 0,1 (rank 7) + schoolbook for cols 2,3 (rank 4) = rank 11.
    
    # Block 1: Columns 0-3
    # Strassen for columns 0,1:
    k_start = 0
    col_start = 0
    
    # M1 = (A00 + A11) * (B0,0 + B1,1)
    U[0, col_start+0] += 1
    U[3, col_start+0] += 1
    V[7*0 + 0, col_start+0] += 1
    V[7*1 + 1, col_start+0] += 1
    W[2*0 + 0, col_start+0] += 1
    W[2*1 + 1, col_start+0] += 1

    # M2 = (A10 + A11) * B0,0
    U[2, col_start+1] += 1
    U[3, col_start+1] += 1
    V[7*0 + 0, col_start+1] += 1
    W[2*0 + 1, col_start+1] += 1
    W[2*1 + 1, col_start+1] -= 1

    # M3 = A00 * (B0,1 - B1,1)
    U[0, col_start+2] += 1
    V[7*0 + 1, col_start+2] += 1
    V[7*1 + 1, col_start+2] -= 1
    W[2*1 + 0, col_start+2] += 1
    W[2*1 + 1, col_start+2] += 1

    # M4 = A11 * (B1,0 - B0,0)
    U[3, col_start+3] += 1
    V[7*1 + 0, col_start+3] += 1
    V[7*0 + 0, col_start+3] -= 1
    W[2*0 + 0, col_start+3] += 1
    W[2*0 + 1, col_start+3] += 1

    # M5 = (A00 + A01) * B1,1
    U[0, col_start+4] += 1
    U[1, col_start+4] += 1
    V[7*1 + 1, col_start+4] += 1
    W[2*0 + 0, col_start+4] -= 1
    W[2*1 + 0, col_start+4] += 1

    # M6 = (A10 - A00) * (B0,0 + B0,1)
    U[2, col_start+5] += 1
    U[0, col_start+5] -= 1
    V[7*0 + 0, col_start+5] += 1
    V[7*0 + 1, col_start+5] += 1
    W[2*1 + 1, col_start+5] += 1

    # M7 = (A01 - A11) * (B1,0 + B1,1)
    U[1, col_start+6] += 1
    U[3, col_start+6] -= 1
    V[7*1 + 0, col_start+6] += 1
    V[7*1 + 1, col_start+6] += 1
    W[2*0 + 0, col_start+6] += 1
    
    # Schoolbook for columns 2,3 (rank 4):
    col_start2 = 7
    # A00 * B0,2 -> C0,2
    U[0, col_start2+0] += 1
    V[7*0 + 2, col_start2+0] += 1
    W[2*2 + 0, col_start2+0] += 1
    
    # A01 * B1,2 -> C0,2
    U[1, col_start2+1] += 1
    V[7*1 + 2, col_start2+1] += 1
    W[2*2 + 0, col_start2+1] += 1
    
    # A10 * B0,2 -> C1,2
    U[2, col_start2+2] += 1
    V[7*0 + 2, col_start2+2] += 1
    W[2*2 + 1, col_start2+2] += 1
    
    # A11 * B1,2 -> C1,2
    U[3, col_start2+3] += 1
    V[7*1 + 2, col_start2+3] += 1
    W[2*2 + 1, col_start2+3] += 1
    
    # Wait, I only did column 2. I need column 3 too.
    # That would be rank 15 for the 4-column block.
    
    # Let me reconsider. For 2x2*2x4, the optimal rank is 11.
    # The construction I described (Strassen + schoolbook for 2 cols) gives 7+4=11
    # but only handles 2 columns with schoolbook, not 4.
    
    # I think I'm confusing myself. Let me restart with a cleaner approach.
    
    # For 2x2 * 2x7, let's use:
    # - Strassen for cols 0,1: rank 7
    # - Strassen for cols 2,3: rank 7  
    # - Strassen for cols 4,5: rank 7
    # - Schoolbook for col 6: rank 4
    # Total: 25 (this is what we already have)
    
    # Alternative:
    # - Strassen for cols 0,1: rank 7
    # - Schoolbook for cols 2,3,4,5,6: rank 4*5 = 20
    # Total: 27 (worse)
    
    # The question is whether we can do better than 25.
    # Known result: rank(2x2*2x7) <= 25, and it's conjectured to be 25.
    
    # Let me just return the rank-25 solution and focus on verifying it's correct.
    
    return None  # Placeholder, will implement properly

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1
    - Use Strassen's rank-7 algorithm for columns 2,3
    - Use Strassen's rank-7 algorithm for columns 4,5
    - Use schoolbook (rank 4) for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    if _time_remaining() < 5.0:
        # Safety: return a trivially valid schoolbook rank-28 decomposition
        return _schoolbook_decomposition()
    
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
        
    # Add Strassen for columns 0,1
    add_strassen(0, 0)
    
    # Add Strassen for columns 2,3
    add_strassen(7, 2)
    
    # Add Strassen for columns 4,5
    add_strassen(14, 4)
    
    # Add schoolbook for column 6 (4 terms)
    col_start = 21
    # A00 * B0,6 -> C0,6
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    
    # A01 * B1,6 -> C0,6
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    
    # A10 * B0,6 -> C1,6
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    
    # A11 * B1,6 -> C1,6
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, num_trials=50):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = 2, 2, 7
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    np.random.seed(42)
    for _ in range(num_trials):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            SA = np.dot(U[:, r], A.flatten())
            SB = np.dot(V[:, r], B.flatten())
            prod = SA * SB
            for k in range(p):
                for i in range(n):
                    C_computed[i, k] += W[k*n + i, r] * prod
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _try_rank24():
    """
    Attempt to find a rank-24 decomposition by modifying the rank-25 solution.
    Strategy: try to merge one schoolbook term with a Strassen term via linear combination.
    """
    if _time_remaining() < 10.0:
        return None
    
    n, m, p = 2, 2, 7
    # Start from rank-25 and try to eliminate one term
    # The schoolbook terms for column 6 are:
    #   A00*B06, A01*B16, A10*B06, A11*B16
    # Try to express one of these as a combination of existing terms.
    
    # This is a research problem; for now, return None if we can't find it quickly.
    # The known result is that rank(2x2*2x7) = 25, so this likely won't succeed.
    return None

if __name__ == "__main__":
    import random
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Random matrices
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 7)
    C_expected = A @ B
    
    C_W = np.zeros(14)
    for r in range(U.shape[1]):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        for idx in range(14):
            C_W[idx] += W[idx, r] * SA * SB
            
    C_mat = np.zeros((2,7))
    for k in range(7):
        for i in range(2):
            C_mat[i, k] = C_W[2*k+i]
            
    if np.allclose(C_mat, C_expected):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
        print(C_mat)
        print(C_expected)
