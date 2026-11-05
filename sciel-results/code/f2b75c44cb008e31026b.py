import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses Strassen for 2x2x2 blocks and schoolbook for the rest.
    Rank 56.
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Tensor convention: T[k*n+i, j, r] where:
    # - i in 0..n-1 (rows of A)
    # - k in 0..p-1 (cols of B / rows of C)
    # - j in 0..m-1 (cols of A / rows of B)
    # The output W is indexed as W[k*n+i, r]
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_decomposition(u_indices, v_indices, w_indices, U_blk, V_blk, W_blk):
        """Add a block decomposition to the parts."""
        rank = U_blk.shape[1]
        for r in range(rank):
            u_vec = np.zeros(n * m)  # 12 entries
            v_vec = np.zeros(m * p)  # 20 entries
            w_vec = np.zeros(p * n)  # 15 entries
            
            for i, idx in enumerate(u_indices):
                u_vec[idx] = U_blk[i, r]
            for i, idx in enumerate(v_indices):
                v_vec[idx] = V_blk[i, r]
            for i, idx in enumerate(w_indices):
                w_vec[idx] = W_blk[i, r]
                
            U_parts.append(u_vec)
            V_parts.append(v_vec)
            W_parts.append(w_vec)

    def strassen_2x2x2():
        """
        Strassen decomposition for 2x2 matrix multiplication.
        A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        U is 4x7 (4 entries of A), V is 4x7 (4 entries of B), W is 4x7 (4 entries of C)
        Indexing: A[0]=a=A[0,0], A[1]=b=A[0,1], A[2]=c=A[1,0], A[3]=d=A[1,1]
                  B[0]=e=B[0,0], B[1]=f=B[0,1], B[2]=g=B[1,0], B[3]=h=B[1,1]
                  C[0]=C[0,0]=ae+bg, C[1]=C[0,1]=af+bh, C[2]=C[1,0]=ce+dg, C[3]=C[1,1]=cf+dh
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 3] = 1
        
        # M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 2] = 1; W[3, 3] = -1
        
        # M3 = a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 1] = 1; W[3, 3] = 1
        
        # M4 = d(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 0] = 1; W[2, 2] = 1
        
        # M5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 0] = -1; W[1, 1] = 1
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 2] = 1
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 0] = 1
        
        return U, V, W

    def schoolbook_2x2x1():
        """
        Schoolbook 2x2 times 2x1 multiplication.
        A = [[a,b],[c,d]], B = [[e],[f]], C = [[ae+bf],[ce+df]]
        U is 4x4 (4 entries of A), V is 2x4 (2 entries of B), W is 2x4 (2 entries of C)
        """
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1  # ae
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1  # bf
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1  # ce
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1  # df
        
        return U, V, W

    def schoolbook_1x2x2():
        """
        Schoolbook 1x2 times 2x2 multiplication.
        A = [a, b], B = [[e, f], [g, h]], C = [ae+bg, af+bh]
        U is 2x4 (2 entries of A), V is 4x4 (4 entries of B), W is 2x4 (2 entries of C)
        """
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1  # ae
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1  # bg
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1  # af
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1  # bh
        
        return U, V, W

    def schoolbook_1x2x1():
        """
        Schoolbook 1x2 times 2x1 multiplication.
        A = [a, b], B = [[e],[f]], C = [ae+bf]
        U is 2x2, V is 2x2, W is 1x2
        """
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1  # ae
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1  # bf
        
        return U, V, W

    # Matrix multiplication: C = A * B where A is 3x4, B is 4x5, C is 3x5
    # In tensor form: C[k*n+i, j] but actually C[i, k] for standard matrix mult
    # The task says: W[k*n+i, r] where i is row of A (0..2), k is col of B (0..4)
    # So W index = k*3 + i, i.e., for C[i,k] the W index is k*3+i
    
    # A has 3 rows (i=0,1,2) and 4 cols (j=0,1,2,3)
    # B has 4 rows (j=0,1,2,3) and 5 cols (k=0,1,2,3,4)
    # C has 3 rows (i=0,1,2) and 5 cols (k=0,1,2,3,4)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # W index for C[i,k] is k*3 + i
    
    # U index for A[i,j] is i*4 + j (row-major, 3*4=12 entries)
    # V index for B[j,k] is j*5 + k (row-major, 4*5=20 entries)
    # W index for C[i,k] is k*3 + i (as specified, 5*3=15 entries)
    
    # Decomposition:
    # Row 0 of A (i=0): A[0,0], A[0,1], A[0,2], A[0,3] -> U indices 0,1,2,3
    # Row 1 of A (i=1): A[1,0], A[1,1], A[1,2], A[1,3] -> U indices 4,5,6,7
    # Row 2 of A (i=2): A[2,0], A[2,1], A[2,2], A[2,3] -> U indices 8,9,10,11
    
    # Row 0 of B (j=0): B[0,0], B[0,1], B[0,2], B[0,3], B[0,4] -> V indices 0,1,2,3,4
    # Row 1 of B (j=1): B[1,0], B[1,1], B[1,2], B[1,3], B[1,4] -> V indices 5,6,7,8,9
    # Row 2 of B (j=2): B[2,0], B[2,1], B[2,2], B[2,3], B[2,4] -> V indices 10,11,12,13,14
    # Row 3 of B (j=3): B[3,0], B[3,1], B[3,2], B[3,3], B[3,4] -> V indices 15,16,17,18,19
    
    # C[0,0] -> W[0*3+0] = W[0]
    # C[0,1] -> W[1*3+0] = W[3]
    # C[0,2] -> W[2*3+0] = W[6]
    # C[0,3] -> W[3*3+0] = W[9]
    # C[0,4] -> W[4*3+0] = W[12]
    # C[1,0] -> W[0*3+1] = W[1]
    # C[1,1] -> W[1*3+1] = W[4]
    # C[1,2] -> W[2*3+1] = W[7]
    # C[1,3] -> W[3*3+1] = W[10]
    # C[1,4] -> W[4*3+1] = W[13]
    # C[2,0] -> W[0*3+2] = W[2]
    # C[2,1] -> W[1*3+2] = W[5]
    # C[2,2] -> W[2*3+2] = W[8]
    # C[2,3] -> W[3*3+2] = W[11]
    # C[2,4] -> W[4*3+2] = W[14]
    
    # Strategy:
    # For rows 0 and 1 of A, group columns (0,1) and (2,3):
    #   A[0:2, 0:2] * B[0:2, :] gives partial C[0:2, :]
    #   A[0:2, 2:4] * B[2:4, :] gives partial C[0:2, :]
    # For row 2 of A:
    #   A[2, 0:2] * B[0:2, :] gives partial C[2, :]
    #   A[2, 2:4] * B[2:4, :] gives partial C[2, :]
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] (2x2x2)
    # A indices: [0,1,4,5] (A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5)
    # B indices: [0,1,5,6] (B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6)
    # C indices: [0,3,1,4] (C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4)
    U_b, V_b, W_b = strassen_2x2x2()
    add_decomposition([0, 1, 4, 5], [0, 1, 5, 6], [0, 3, 1, 4], U_b, V_b, W_b)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] (2x2x2)
    # A indices: [0,1,4,5]
    # B indices: [2,3,7,8] (B[0,2]=2, B[0,3]=3, B[1,2]=7, B[1,3]=8)
    # C indices: [6,9,7,10] (C[0,2]=6, C[0,3]=9, C[1,2]=7, C[1,3]=10)
    U_b, V_b, W_b = strassen_2x2x2()
    add_decomposition([0, 1, 4, 5], [2, 3, 7, 8], [6, 9, 7, 10], U_b, V_b, W_b)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] (2x2x1)
    # A indices: [0,1,4,5]
    # B indices: [4,9] (B[0,4]=4, B[1,4]=9)
    # C indices: [12,13] (C[0,4]=12, C[1,4]=13)
    U_b, V_b, W_b = schoolbook_2x2x1()
    add_decomposition([0, 1, 4, 5], [4, 9], [12, 13], U_b, V_b, W_b)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2] (2x2x2)
    # A indices: [2,3,6,7] (A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7)
    # B indices: [10,11,15,16] (B[2,0]=10, B[2,1]=11, B[3,0]=15, B[3,1]=16)
    # C indices: [0,3,1,4]
    U_b, V_b, W_b = strassen_2x2x2()
    add_decomposition([2, 3, 6, 7], [10, 11, 15, 16], [0, 3, 1, 4], U_b, V_b, W_b)
    
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4] (2x2x2)
    # A indices: [2,3,6,7]
    # B indices: [12,13,17,18] (B[2,2]=12, B[2,3]=13, B[3,2]=17, B[3,3]=18)
    # C indices: [6,9,7,10]
    U_b, V_b, W_b = strassen_2x2x2()
    add_decomposition([2, 3, 6, 7], [12, 13, 17, 18], [6, 9, 7, 10], U_b, V_b, W_b)
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] (2x2x1)
    # A indices: [2,3,6,7]
    # B indices: [14,19] (B[2,4]=14, B[3,4]=19)
    # C indices: [12,13]
    U_b, V_b, W_b = schoolbook_2x2x1()
    add_decomposition([2, 3, 6, 7], [14, 19], [12, 13], U_b, V_b, W_b)
    
    # Block 7: A[2, 0:2] * B[0:2, 0:2] (1x2x2)
    # A indices: [8,9] (A[2,0]=8, A[2,1]=9)
    # B indices: [0,1,5,6]
    # C indices: [2,5] (C[2,0]=2, C[2,1]=5)
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_decomposition([8, 9], [0, 1, 5, 6], [2, 5], U_b, V_b, W_b)
    
    # Block 8: A[2, 0:2] * B[0:2, 2:4] (1x2x2)
    # A indices: [8,9]
    # B indices: [2,3,7,8]
    # C indices: [8,11] (C[2,2]=8, C[2,3]=11)
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_decomposition([8, 9], [2, 3, 7, 8], [8, 11], U_b, V_b, W_b)
    
    # Block 9: A[2, 0:2] * B[0:2, 4:5] (1x2x1)
    # A indices: [8,9]
    # B indices: [4,9]
    # C indices: [14] (C[2,4]=14)
    U_b, V_b, W_b = schoolbook_1x2x1()
    add_decomposition([8, 9], [4, 9], [14], U_b, V_b, W_b)
    
    # Block 10: A[2, 2:4] * B[2:4, 0:2] (1x2x2)
    # A indices: [10,11] (A[2,2]=10, A[2,3]=11)
    # B indices: [10,11,15,16]
    # C indices: [2,5]
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_decomposition([10, 11], [10, 11, 15, 16], [2, 5], U_b, V_b, W_b)
    
    # Block 11: A[2, 2:4] * B[2:4, 2:4] (1x2x2)
    # A indices: [10,11]
    # B indices: [12,13,17,18]
    # C indices: [8,11]
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_decomposition([10, 11], [12, 13, 17, 18], [8, 11], U_b, V_b, W_b)
    
    # Block 12: A[2, 2:4] * B[2:4, 4:5] (1x2x1)
    # A indices: [10,11]
    # B indices: [14,19]
    # C indices: [14]
    U_b, V_b, W_b = schoolbook_1x2x1()
    add_decomposition([10, 11], [14, 19], [14], U_b, V_b, W_b)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
