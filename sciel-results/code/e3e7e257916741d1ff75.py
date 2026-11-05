import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses Strassen's algorithm for 2x2x2 blocks to reduce the rank from the schoolbook 60
    to 56 by splitting the 3x4x5 multiplication into a 2x4x5 part (rank 36) and a 1x4x5 part (rank 20).
    The decomposition is given by U, V, W such that T = sum_r U[:, r] outer V[:, r] outer W[:, r].
    Indices:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank) with W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def strassen_2x2x2():
        """Strassen decomposition for 2x2x2 matrix multiplication.
        Returns U(4x7), V(4x7), W(4x7) where:
        U indices: 0:A00, 1:A01, 2:A10, 3:A11
        V indices: 0:B00, 1:B01, 2:B10, 3:B11
        W indices: 0:C00, 1:C01, 2:C10, 3:C11
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def schoolbook_2x2x1():
        """Schoolbook decomposition for 2x2x1. Rank 4.
        A(2x2) * B(2x1) = C(2x1)
        U indices: 0:A00, 1:A01, 2:A10, 3:A11
        V indices: 0:B0, 1:B1
        W indices: 0:C0, 1:C1
        """
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    def schoolbook_1x2x2():
        """Schoolbook decomposition for 1x2x2. Rank 4.
        A(1x2) * B(2x2) = C(1x2)
        U indices: 0:A0, 1:A1
        V indices: 0:B00, 1:B01, 2:B10, 3:B11
        W indices: 0:C0, 1:C1
        """
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W
    
    def schoolbook_1x2x1():
        """Schoolbook decomposition for 1x2x1. Rank 2.
        A(1x2) * B(2x1) = C(1x1)
        U indices: 0:A0, 1:A1
        V indices: 0:B0, 1:B1
        W indices: 0:C0
        """
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    # Assemble 3x4x5 multiplication using block decomposition.
    # U indices: i*m + j for A[i,j], i in 0..2, j in 0..3
    # V indices: i*p + j for B[i,j], i in 0..3, j in 0..4
    # W indices: k*n + i for C[i,k], i in 0..2, k in 0..4
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            u = np.zeros(n * m)
            for i, idx in enumerate(u_idx):
                u[idx] = U_blk[i, r]
            U_parts.append(u)
            
            v = np.zeros(m * p)
            for i, idx in enumerate(v_idx):
                v[idx] = V_blk[i, r]
            V_parts.append(v)
            
            w = np.zeros(p * n)
            for i, idx in enumerate(w_idx):
                w[idx] = W_blk[i, r]
            W_parts.append(w)
    
    # W index helper: C[i,k] -> k*n + i
    def w_idx_c(i, k):
        return k * n + i
    
    # Part 1: Rows 0,1 of A (2x4) times B (4x5)
    # Split columns of A into [0,1] and [2,3], rows of B into [0,1] and [2,3]
    # 4 blocks of 2x2x5, each split into 2x2x2 + 2x2x2 + 2x2x1
    
    # A rows 0,1 cols 0,1: U indices [0,1,4,5]
    # A rows 0,1 cols 2,3: U indices [2,3,6,7]
    # B rows 0,1 cols 0,1: V indices [0,1,5,6]
    # B rows 0,1 cols 2,3: V indices [2,3,7,8]
    # B rows 0,1 col 4: V indices [4,9]
    # B rows 2,3 cols 0,1: V indices [10,11,15,16]
    # B rows 2,3 cols 2,3: V indices [12,13,17,18]
    # B rows 2,3 col 4: V indices [14,19]
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] (2x2x2)
    add_block([0,1,4,5], [0,1,5,6], [w_idx_c(0,0),w_idx_c(0,1),w_idx_c(1,0),w_idx_c(1,1)],
              *strassen_2x2x2())
    
    # Block 2: A[0:2,0:2] * B[0:2,2:4] (2x2x2)
    add_block([0,1,4,5], [2,3,7,8], [w_idx_c(0,2),w_idx_c(0,3),w_idx_c(1,2),w_idx_c(1,3)],
              *strassen_2x2x2())
    
    # Block 3: A[0:2,0:2] * B[0:2,4] (2x2x1)
    add_block([0,1,4,5], [4,9], [w_idx_c(0,4),w_idx_c(1,4)],
              *schoolbook_2x2x1())
    
    # Block 4: A[0:2,2:4] * B[2:4,0:2] (2x2x2)
    add_block([2,3,6,7], [10,11,15,16], [w_idx_c(0,0),w_idx_c(0,1),w_idx_c(1,0),w_idx_c(1,1)],
              *strassen_2x2x2())
    
    # Block 5: A[0:2,2:4] * B[2:4,2:4] (2x2x2)
    add_block([2,3,6,7], [12,13,17,18], [w_idx_c(0,2),w_idx_c(0,3),w_idx_c(1,2),w_idx_c(1,3)],
              *strassen_2x2x2())
    
    # Block 6: A[0:2,2:4] * B[2:4,4] (2x2x1)
    add_block([2,3,6,7], [14,19], [w_idx_c(0,4),w_idx_c(1,4)],
              *schoolbook_2x2x1())
    
    # Part 2: Row 2 of A (1x4) times B (4x5)
    # A row 2 cols 0,1: U indices [8,9]
    # A row 2 cols 2,3: U indices [10,11]
    
    # Block 7: A[2,0:2] * B[0:2,0:2] (1x2x2)
    add_block([8,9], [0,1,5,6], [w_idx_c(2,0),w_idx_c(2,1)],
              *schoolbook_1x2x2())
    
    # Block 8: A[2,0:2] * B[0:2,2:4] (1x2x2)
    add_block([8,9], [2,3,7,8], [w_idx_c(2,2),w_idx_c(2,3)],
              *schoolbook_1x2x2())
    
    # Block 9: A[2,0:2] * B[0:2,4] (1x2x1)
    add_block([8,9], [4,9], [w_idx_c(2,4)],
              *schoolbook_1x2x1())
    
    # Block 10: A[2,2:4] * B[2:4,0:2] (1x2x2)
    add_block([10,11], [10,11,15,16], [w_idx_c(2,0),w_idx_c(2,1)],
              *schoolbook_1x2x2())
    
    # Block 11: A[2,2:4] * B[2:4,2:4] (1x2x2)
    add_block([10,11], [12,13,17,18], [w_idx_c(2,2),w_idx_c(2,3)],
              *schoolbook_1x2x2())
    
    # Block 12: A[2,2:4] * B[2:4,4] (1x2x1)
    add_block([10,11], [14,19], [w_idx_c(2,4)],
              *schoolbook_1x2x1())
    
    # Total rank: 7+7+4+7+7+4+4+4+2+4+4+2 = 56
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
