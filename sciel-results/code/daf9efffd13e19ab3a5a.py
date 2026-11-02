"""
Tensor decomposition for 2x4 by 4x4 matrix multiplication.
Uses a structured rank-28 decomposition via Strassen on two 2x4x2 sub-problems.
"""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x4 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Strategy:
    - Split B into two 4x2 blocks: B_left (cols 0,1) and B_right (cols 2,3).
    - Split C into two 2x2 blocks: C_left (cols 0,1) and C_right (cols 2,3).
    - C_left = A @ B_left and C_right = A @ B_right are independent 2x4x2 multiplications.
    - Each 2x4x2 multiplication can be done with rank 14 using a Strassen-like scheme.
    - Total rank = 14 + 14 = 28.
    
    For a 2x4x2 multiplication:
    A is 2x4, B is 4x2, C is 2x2.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can view this as:
    C = [A_left @ B_left_half, A_right @ B_right_half]
    where A_left = A[:,0:2], A_right = A[:,2:4]
    B_left_half = B[0:2,:], B_right_half = B[2:4,:]
    
    Actually, let's use a more direct Strassen-like approach for 2x4x2.
    
    For 2x2 matrix multiplication, Strassen uses 7 multipliers.
    For 2x4x2, we can split A into two 2x2 blocks and B into two 2x2 blocks (vertically).
    
    A = [A1 | A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x2
    
    C = A1*B1 + A2*B2
    
    Each 2x2*2x2 can be done with 7 multipliers, so naive would be 14.
    But we can share some multipliers if we structure it right.
    
    Actually, A1*B1 + A2*B2 is just the sum of two 2x2 products.
    If we compute each with Strassen (7 each), that's 14 total.
    
    Let's implement this.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # We'll build rank-28 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Helper: add a Strassen 2x2x2 decomposition
    # A is 2x2, B is 2x2, C is 2x2
    # U indices for A: u_base + [0,1,2,3] (row-major: [A00, A01, A10, A11])
    # V indices for B: v_base + [0,1,2,3] (row-major: [B00, B01, B10, B11])
    # W indices for C: w_base + [0,1,2,3] (row-major: [C00, C01, C10, C11])
    # But W uses trace convention: W[k*n+i] = W[k*2+i]
    # For 2x2 C: C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    # So w_base + [0,1,2,3] corresponds to C[0,0], C[0,1], C[1,0], C[1,1]
    
    def add_strassen_2x2(u_base, v_base, w_base, col_offset):
        """
        Add Strassen multipliers for a 2x2*2x2 multiplication.
        A indices: u_base + [0,1,2,3] = [A00, A01, A10, A11]
        B indices: v_base + [0,1,2,3] = [B00, B01, B10, B11]
        C indices: w_base + [0,1,2,3] = [C00, C01, C10, C11]
        
        Strassen multipliers:
        M1 = (A00+A11)*(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B01-B11)
        M4 = A11*(B10-B00)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B00+B01)
        M7 = (A01-A11)*(B10+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        # M1 = (A00+A11)*(B00+B11)
        U[u_base + 0][col_offset + 0] += 1  # A00
        U[u_base + 3][col_offset + 0] += 1  # A11
        V[v_base + 0][col_offset + 0] += 1  # B00
        V[v_base + 3][col_offset + 0] += 1  # B11
        W[w_base + 0][col_offset + 0] += 1  # C00
        W[w_base + 3][col_offset + 0] += 1  # C11
        
        # M2 = (A10+A11)*B00
        U[u_base + 2][col_offset + 1] += 1  # A10
        U[u_base + 3][col_offset + 1] += 1  # A11
        V[v_base + 0][col_offset + 1] += 1  # B00
        W[w_base + 2][col_offset + 1] += 1  # C10
        W[w_base + 3][col_offset + 1] -= 1  # C11
        
        # M3 = A00*(B01-B11)
        U[u_base + 0][col_offset + 2] += 1  # A00
        V[v_base + 1][col_offset + 2] += 1  # B01
        V[v_base + 3][col_offset + 2] -= 1  # B11
        W[w_base + 1][col_offset + 2] += 1  # C01
        W[w_base + 3][col_offset + 2] += 1  # C11
        
        # M4 = A11*(B10-B00)
        U[u_base + 3][col_offset + 3] += 1  # A11
        V[v_base + 2][col_offset + 3] += 1  # B10
        V[v_base + 0][col_offset + 3] -= 1  # B00
        W[w_base + 0][col_offset + 3] += 1  # C00
        W[w_base + 2][col_offset + 3] += 1  # C10
        
        # M5 = (A00+A01)*B11
        U[u_base + 0][col_offset + 4] += 1  # A00
        U[u_base + 1][col_offset + 4] += 1  # A01
        V[v_base + 3][col_offset + 4] += 1  # B11
        W[w_base + 0][col_offset + 4] -= 1  # C00
        W[w_base + 1][col_offset + 4] += 1  # C01
        
        # M6 = (A10-A00)*(B00+B01)
        U[u_base + 2][col_offset + 5] += 1  # A10
        U[u_base + 0][col_offset + 5] -= 1  # A00
        V[v_base + 0][col_offset + 5] += 1  # B00
        V[v_base + 1][col_offset + 5] += 1  # B01
        W[w_base + 3][col_offset + 5] += 1  # C11
        
        # M7 = (A01-A11)*(B10+B11)
        U[u_base + 1][col_offset + 6] += 1  # A01
        U[u_base + 3][col_offset + 6] -= 1  # A11
        V[v_base + 2][col_offset + 6] += 1  # B10
        V[v_base + 3][col_offset + 6] += 1  # B11
        W[w_base + 0][col_offset + 6] += 1  # C00
        W[w_base + 3][col_offset + 6] += 1  # C11
    
    # Left half: C_left = A @ B[:,0:2]
    # A is 2x4, B_left is 4x2
    # A = [A1 | A2] where A1 = A[:,0:2], A2 = A[:,2:4]
    # B_left = [B1; B2] where B1 = B[0:2,:], B2 = B[2:4,:]
    # C_left = A1*B1 + A2*B2
    
    # A1 indices in U: [0,1,2,3] = [A[0,0], A[0,1], A[1,0], A[1,1]]
    # A2 indices in U: [4,5,6,7] = [A[0,2], A[0,3], A[1,2], A[1,3]]
    # B1 indices in V: [0,1,2,3] = [B[0,0], B[0,1], B[1,0], B[1,1]]
    # B2 indices in V: [4,5,6,7] = [B[2,0], B[2,1], B[3,0], B[3,1]]
    # C_left indices in W: [0,1,2,3] = [C[0,0], C[0,1], C[1,0], C[1,1]]
    
    # A1*B1: u_base=0, v_base=0, w_base=0
    add_strassen_2x2(0, 0, 0, 0)
    
    # A2*B2: u_base=4, v_base=4, w_base=0
    add_strassen_2x2(4, 4, 0, 7)
    
    # Right half: C_right = A @ B[:,2:4]
    # B_right = [B3; B4] where B3 = B[0:2,2:4], B4 = B[2:4,2:4]
    # B3 indices in V: [8,9,10,11] = [B[0,2], B[0,3], B[1,2], B[1,3]]
    # B4 indices in V: [12,13,14,15] = [B[2,2], B[2,3], B[3,2], B[3,3]]
    # C_right indices in W: [4,5,6,7] = [C[0,2], C[0,3], C[1,2], C[1,3]]
    
    # A1*B3: u_base=0, v_base=8, w_base=4
    add_strassen_2x2(0, 8, 4, 14)
    
    # A2*B4: u_base=4, v_base=12, w_base=4
    add_strassen_2x2(4, 12, 4, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
