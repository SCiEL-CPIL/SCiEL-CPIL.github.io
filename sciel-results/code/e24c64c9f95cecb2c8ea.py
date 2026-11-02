"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_strassen_2x2x2():
    """Returns U, V, W for 2x2x2 matrix multiplication with rank 7."""
    # U: 4x7, V: 4x7, W: 4x7
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a0+a3)(b0+b3)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00 gets +1, C11 gets +1? No, W maps to C.
    # W[k*n+i, r] is coefficient for C[i, k]? 
    # Task says W[k*n+i, r]. So row index is k*n + i.
    # C00 is i=0, k=0 -> index 0.
    # C11 is i=1, k=1 -> index 3.
    W[0, 0] = 1
    W[3, 0] = 1
    
    # m2 = (a2+a3)b0
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10 (i=1, k=0) -> index 2
    
    # m3 = a0(b1-b3)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C01 (i=0, k=1) -> index 1
    
    # m4 = a3(b2-b0)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C00 +1, C10 +1
    
    # m5 = (a0+a1)b3
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # C00 -1, C01 +1
    
    # m6 = (a2-a0)(b0+b1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11 +1
    
    # m7 = (a1+a3)(b2-b0)
    U[1, 6] = 1; U[3, 6] = 1
    V[2, 6] = 1; V[0, 6] = -1
    W[0, 6] = 1; W[3, 6] = -1 # C00 +1, C11 -1
    
    return U, V, W

def _get_schoolbook(n, m, p):
    """Returns U, V, W for n x m x p matrix multiplication with rank n*m*p."""
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
    return U, V, W

def _combine_decompositions(n, m, p, decomp_list):
    """
    Combines a list of decompositions.
    decomp_list is a list of tuples (U, V, W, offset_U, offset_V, offset_W, scale_U, scale_V, scale_W).
    This is a simplified version. We will just concatenate and map indices.
    """
    # This function is too complex to implement generically in this short time.
    # I will manually construct the 112 rank solution.
    pass

def construct():
    deadline = time.time() + 120
    
    n, m, p = 4, 5, 6
    
    # We will build the 112 rank solution.
    # Split m=5 into 2+3.
    # A (4x5) = [A1 (4x2), A2 (4x3)]
    # B (5x6) = [B11 (2x3), B12 (2x3); B21 (3x3), B22 (3x3)]
    # C (4x6) = [C1 (4x3), C2 (4x3)]
    
    # Subproblems:
    # M1 = A1 B11 (4x2x3)
    # M2 = A2 B21 (4x3x3)
    # M3 = A1 B12 (4x2x3)
    # M4 = A2 B22 (4x3x3)
    
    # We need decompositions for 4x2x3 and 4x3x3.
    
    # 4x2x3:
    # Split 4x2 into two 2x2.
    # A1 (4x2) = [A1_top (2x2); A1_bot (2x2)]
    # B11 (2x3) = [B11_1 (2x2), B11_2 (2x1)]
    # C1 (4x3) = [C1_top (2x3); C1_bot (2x3)]
    # C1_top = A1_top B11_1 + A1_top B11_2? No.
    # C1_top (2x3) = A1_top (2x2) * B11 (2x3).
    # This is 2x2x3.
    # We found R(2,2,3) <= 11.
    # So R(4,2,3) <= 2 * 11 = 22.
    
    # 4x3x3:
    # Split 4x3 into two 2x3.
    # A2 (4x3) = [A2_top (2x3); A2_bot (2x3)]
    # B21 (3x3) = [B21_top (2x3); B21_bot (1x3)]? No, B is 3x3.
    # B21 (3x3) split into B21_11 (2x2), B21_12 (2x1), B21_21 (1x2), B21_22 (1x1).
    # C2_top (2x3) = A2_top (2x3) * B21 (3x3).
    # This is 2x3x3.
    # We found R(2,3,3) <= 17.
    # So R(4,3,3) <= 2 * 17 = 34.
    
    # Total rank = 22 + 34 + 22 + 34 = 112.
    
    # I will implement the 112 rank solution by constructing the U, V, W matrices.
    
    # Helper to get 2x2x3 decomposition (rank 11)
    def _get_2x2x3():
        # A (2x2), B (2x3), C (2x3)
        # B = [B1 (2x2), B2 (2x1)]
        # C = [C1 (2x2), C2 (2x1)]
        # C1 = A B1 (2x2x2, rank 7)
        # C2 = A B2 (2x2x1, rank 4)
        
        U_222, V_222, W_222 = _get_strassen_2x2x2()
        
        # 2x2x1 schoolbook
        U_221 = np.zeros((4, 4))
        V_221 = np.zeros((2, 4))
        W_221 = np.zeros((2, 4))
        # A: 0,1,2,3. B: 0,1. C: 0,1.
        # C0 = a0 b0 + a1 b1
        # C1 = a2 b0 + a3 b1
        U_221[0, 0] = 1; U_221[1, 1] = 1
        V_221[0, 0] = 1; V_221[1, 1] = 1
        W_221[0, 0] = 1; W_221[0, 1] = 1
        
        U_221[2, 2] = 1; U_221[3, 3] = 1
        V_221[0, 2] = 1; V_221[1, 3] = 1
        W_221[1, 2] = 1; W_221[1, 3] = 1
        
        # Combine
        U = np.hstack([U_222, U_221])
        V = np.vstack([V_222, V_221]) # V_222 is 4x7, V_221 is 2x4.
        # Wait, V indices for B1 (2x2) are 0,1,2,3.
        # V indices for B2 (2x1) are 4,5?
        # In 2x2x3, B has 6 elements.
        # B1 is B[:, 0:2], indices 0,1,2,3.
        # B2 is B[:, 2:3], indices 4,5.
        # So V_221 should map to indices 4,5.
        # Let's adjust V_221.
        V_221_adj = np.zeros((6, 4))
        V_221_adj[4, 0] = 1; V_221_adj[5, 1] = 1
        V_221_adj[4, 2] = 1; V_221_adj[5, 3] = 1
        
        V = np.hstack([V_222, V_221_adj])
        
        # W indices for C1 (2x2) are 0,1,2,3.
        # W indices for C2 (2x1) are 4,5?
        # C is 2x3. C1 is C[:, 0:2], indices 0,1,2,3.
        # C2 is C[:, 2:3], indices 4,5.
        # W_221 maps to C indices 0,1.
        # We need to map them to 4,5.
        W_221_adj = np.zeros((6, 4))
        W_221_adj[4, 0] = 1; W_221_adj[4, 1] = 1
        W_221_adj[5, 2] = 1; W_221_adj[5, 3] = 1
        
        W = np.hstack([W_222, W_221_adj])
        
        return U, V, W

    # Helper to get 2x3x3 decomposition (rank 17)
    def _get_2x3x3():
        # A (2x3), B (3x3), C (2x3)
        # A = [A1 (2x2), A2 (2x1)]
        # B = [B11 (2x2), B12 (2x1); B21 (1x2), B22 (1x1)]
        # C = [C1 (2x2), C2 (2x1)]
        # C1 = A1 B11 + A2 B21
        # C2 = A1 B12 + A2 B22
        
        # A1 B11: 2x2x2, rank 7
        U_1, V_1, W_1 = _get_strassen_2x2x2()
        
        # A2 B21: 2x1x2, rank 4
        # A2 (2x1), B21 (1x2), C_part (2x2)
        # C_part = A2 B21
        # A2: 0,1. B21: 0,1. C: 0,1,2,3.
        U_2 = np.zeros((4, 4))
        V_2 = np.zeros((2, 4))
        W_2 = np.zeros((4, 4))
        # A2 indices in A (2x3) are 2,3.
        # B21 indices in B (3x3) are 4,5.
        # C indices are 0,1,2,3.
        
        # m1 = a0 b0 -> C00
        # m2 = a0 b1 -> C01
        # m3 = a1 b0 -> C10
        # m4 = a1 b1 -> C11
        
        U_2[2, 0] = 1; U_2[2, 1] = 1
        U_2[3, 2] = 1; U_2[3, 3] = 1
        
        V_2[4, 0] = 1; V_2[5, 1] = 1
        V_2[4, 2] = 1; V_2[5, 3] = 1
        
        W_2[0, 0] = 1
        W_2[1, 1] = 1
        W_2[2, 2] = 1
        W_2[3, 3] = 1
        
        # A1 B12: 2x2x1, rank 4
        # A1 (2x2), B12 (2x1), C_part (2x1)
        # A1 indices: 0,1,2,3.
        # B12 indices: 2,3.
        # C indices: 4,5.
        
        U_3 = np.zeros((4, 4))
        V_3 = np.zeros((2, 4))
        W_3 = np.zeros((2, 4))
        
        # m1 = a0 b0 -> C0
        # m2 = a1 b0 -> C1
        # m3 = a2 b1 -> C0
        # m4 = a3 b1 -> C1
        
        U_3[0, 0] = 1; U_3[2, 0] = 1
        U_3[1, 1] = 1; U_3[3, 1] = 1
        
        V_3[2, 0] = 1; V_3[2, 1] = 1
        V_3[3, 2] = 1; V_3[3, 3] = 1
        
        W_3[4, 0] = 1; W_3[4, 2] = 1
        W_3[5, 1] = 1; W_3[5, 3] = 1
        
        # A2 B22: 2x1x1, rank 2
        # A2 (2x1), B22 (1x1), C_part (2x1)
        # A2 indices: 2,3.
        # B22 index: 8.
        # C indices: 4,5.
        
        U_4 = np.zeros((4, 2))
        V_4 = np.zeros((1, 2))
        W_4 = np.zeros((2, 2))
        
        U_4[2, 0] = 1
        U_4[3, 1] = 1
        
        V_4[8, 0] = 1
        V_4[8, 1] = 1
        
        W_4[4, 0] = 1
        W_4[5, 1] = 1
        
        # Combine
        # U: 6 rows (2x3)
        U = np.zeros((6, 17))
        V = np.zeros((9, 17))
        W = np.zeros((6, 17))
        
        # Col 0-6: A1 B11
        U[0:4, 0:7] = U_1
        V[0:4, 0:7] = V_1
        W[0:4, 0:7] = W_1
        
        # Col 7-10: A2 B21
        U[0:4, 7:11] = U_2
        V[0:4, 7:11] = V_2
        W[0:4, 7:11] = W_2
        
        # Col 11-14: A1 B12
        U[0:4, 11:15] = U_3
        V[0:4, 11:15] = V_3
        W[0:4, 11:15] = W_3
        
        # Col 15-16: A2 B22
        U[0:4, 15:17] = U_4
        V[0:4, 15:17] = V_4
        W[0:4, 15:17] = W_4
        
        return U, V, W

    # Now combine 4x2x3 and 4x3x3.
    U_423, V_423, W_423 = _get_4x2x3()
    U_433, V_433, W_433 = _get_4x3x3()
    
    # ...

# EVOLVE-BLOCK-END
