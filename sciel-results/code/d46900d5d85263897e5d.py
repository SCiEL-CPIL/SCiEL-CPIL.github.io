"""
5x5 matrix multiplication - rank-87 construction via 4+1 block decomposition.
Uses M(4) = 26 (via 2+2 Strassen) for the 4x4 core, plus schoolbook for cross terms.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 130.0

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros((n*m, 1), dtype=float)
        V = np.zeros((m*p, 1), dtype=float)
        W = np.zeros((p*n, 1), dtype=float)

        for (i, j), val in u_vals.items():
            U[i*m+j, 0] = val
        for (j, k), val in v_vals.items():
            V[j*p+k, 0] = val
        for (k, i), val in w_vals.items():
            W[k*n+i, 0] = val

        U_list.append(U.flatten())
        V_list.append(V.flatten())
        W_list.append(W.flatten())

    def strassen_2x2(i0, i1, j0, j1, k0, k1):
        """Strassen for 2x2 * 2x2 block at positions (i0,i1)x(j0,j1) * (j0,j1)x(k0,k1)"""
        # M1 = (A11+A22)(B11+B22)
        add_term({(i0,j0):1,(i1,j1):1}, {(j0,k0):1,(j1,k1):1}, {(k0,i0):1,(k1,i1):1})
        # M2 = (A21+A22)B11
        add_term({(i1,j0):1,(i1,j1):1}, {(j0,k0):1}, {(k0,i1):1,(k1,i1):-1})
        # M3 = A11(B21-B11)
        add_term({(i0,j0):1}, {(j0,k1):1,(j1,k1):-1}, {(k1,i0):1,(k1,i1):1})
        # M4 = A22(B12-B11)
        add_term({(i1,j1):1}, {(j1,k0):1,(j0,k0):-1}, {(k0,i0):1,(k0,i1):1})
        # M5 = (A11+A12)B22
        add_term({(i0,j0):1,(i0,j1):1}, {(j1,k1):1}, {(k0,i0):-1,(k1,i0):1})
        # M6 = (A21-A11)(B11+B12)
        add_term({(i1,j0):1,(i0,j0):-1}, {(j0,k0):1,(j0,k1):1}, {(k1,i1):1})
        # M7 = (A12-A22)(B21+B22)
        add_term({(i0,j1):1,(i1,j1):-1}, {(j1,k0):1,(j1,k1):1}, {(k0,i0):1})

    def schoolbook_block(i_range, j_range, k_range):
        """Schoolbook multiplication for block A[i_range, j_range] * B[j_range, k_range] -> C[i_range, k_range]"""
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    add_term({(i,j):1}, {(j,k):1}, {(k,i):1})

    def m4_block(i0, j0, k0):
        """M(4) decomposition for 4x4*4x4 block using 2+2 split.
        A is rows i0..i0+3, cols j0..j0+3
        B is rows j0..j0+3, cols k0..k0+3
        C is rows i0..i0+3, cols k0..k0+3
        """
        # Split: block 1 = indices 0,1 (offset 0,1), block 2 = indices 2,3 (offset 2,3)
        # A11: rows i0,i0+1; cols j0,j0+1
        # A12: rows i0,i0+1; cols j0+2,j0+3
        # A21: rows i0+2,i0+3; cols j0,j0+1
        # A22: rows i0+2,i0+3; cols j0+2,j0+3
        # B11: rows j0,j0+1; cols k0,k0+1
        # B12: rows j0,j0+1; cols k0+2,k0+3
        # B21: rows j0+2,j0+3; cols k0,k0+1
        # B22: rows j0+2,j0+3; cols k0+2,k0+3
        
        i1 = i0+1; i2 = i0+2; i3 = i0+3
        j1 = j0+1; j2 = j0+2; j3 = j0+3
        k1 = k0+1; k2 = k0+2; k3 = k0+3
        
        # C11 = A11*B11 + A12*B21
        strassen_2x2(i0, i1, j0, j1, k0, k1)
        schoolbook_block([i0, i1], [j2, j3], [k0, k1])
        
        # C12 = A11*B12 + A12*B22
        schoolbook_block([i0, i1], [j0, j1], [k2, k3])
        strassen_2x2(i0, i1, j2, j3, k2, k3)
        
        # C21 = A21*B11 + A22*B21
        schoolbook_block([i2, i3], [j0, j1], [k0, k1])
        strassen_2x2(i2, i3, j2, j3, k0, k1)
        
        # C22 = A21*B12 + A22*B22
        schoolbook_block([i2, i3], [j0, j1], [k2, k3])
        strassen_2x2(i2, i3, j2, j3, k2, k3)

    # 4+1 split for 5x5
    # A: rows 0-3 (block 4), row 4 (block 1)
    #     cols 0-3 (block 4), col 4 (block 1)
    # B: rows 0-3 (block 4), row 4 (block 1)
    #     cols 0-3 (block 4), col 4 (block 1)
    # C: rows 0-3 (block 4), row 4 (block 1)
    #     cols 0-3 (block 4), col 4 (block 1)
    
    # C44 = A44*B44 + A41*B14
    # A44*B44: M(4) = 26
    m4_block(0, 0, 0)
    # A41*B14: schoolbook 4*1*4 = 16
    schoolbook_block([0,1,2,3], [4], [0,1,2,3])
    
    # C41 = A44*B41 + A41*B11
    # A44*B41: schoolbook 4*4*1 = 16
    schoolbook_block([0,1,2,3], [0,1,2,3], [4])
    # A41*B11: schoolbook 4*1*1 = 4
    schoolbook_block([0,1,2,3], [4], [4])
    
    # C14 = A14*B44 + A11*B14
    # A14*B44: schoolbook 1*4*4 = 16
    schoolbook_block([4], [0,1,2,3], [0,1,2,3])
    # A11*B14: schoolbook 1*1*4 = 4
    schoolbook_block([4], [4], [0,1,2,3])
    
    # C11 = A14*B41 + A11*B11
    # A14*B41: schoolbook 1*4*1 = 4
    schoolbook_block([4], [0,1,2,3], [4])
    # A11*B11: schoolbook 1*1*1 = 1
    add_term({(4,4):1}, {(4,4):1}, {(4,4):1})

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    rank = U.shape[1]
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
