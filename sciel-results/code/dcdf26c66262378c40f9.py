import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c_offset = 0
    
    # Helper to add Strassen 2x2x2 decomposition
    # j_base: starting j index
    # k_base: starting k index
    def add_strassen_2x2x2(j_base, k_base):
        nonlocal c_offset
        # Local indices i in {0,1}, j' in {0,1}, k' in {0,1}
        # Global j = j_base + j'
        # Global k = k_base + k'
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C11
        U[0*m + (j_base+0)][c_offset] = 1.0
        U[1*m + (j_base+1)][c_offset] = 1.0
        V[(j_base+0)*p + (k_base+0)][c_offset] = 1.0
        V[(j_base+1)*p + (k_base+1)][c_offset] = 1.0
        W[(k_base+0)*n + 0][c_offset] = 1.0
        W[(k_base+1)*n + 1][c_offset] = 1.0
        c_offset += 1
        
        # M2 = (A10+A11)B00
        U[1*m + (j_base+0)][c_offset] = 1.0
        U[1*m + (j_base+1)][c_offset] = 1.0
        V[(j_base+0)*p + (k_base+0)][c_offset] = 1.0
        W[(k_base+0)*n + 1][c_offset] = 1.0
        c_offset += 1
        
        # M3 = A00(B01-B11)
        U[0*m + (j_base+0)][c_offset] = 1.0
        V[(j_base+0)*p + (k_base+1)][c_offset] = 1.0
        V[(j_base+1)*p + (k_base+1)][c_offset] = -1.0
        W[(k_base+1)*n + 0][c_offset] = 1.0
        c_offset += 1
        
        # M4 = A11(B10-B00)
        U[1*m + (j_base+1)][c_offset] = 1.0
        V[(j_base+1)*p + (k_base+0)][c_offset] = 1.0
        V[(j_base+0)*p + (k_base+0)][c_offset] = -1.0
        W[(k_base+0)*n + 1][c_offset] = 1.0
        c_offset += 1
        
        # M5 = (A00+A01)B11
        U[0*m + (j_base+0)][c_offset] = 1.0
        U[0*m + (j_base+1)][c_offset] = 1.0
        V[(j_base+1)*p + (k_base+1)][c_offset] = 1.0
        W[(k_base+1)*n + 0][c_offset] = 1.0
        c_offset += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1*m + (j_base+0)][c_offset] = 1.0
        U[0*m + (j_base+0)][c_offset] = -1.0
        V[(j_base+0)*p + (k_base+0)][c_offset] = 1.0
        V[(j_base+0)*p + (k_base+1)][c_offset] = 1.0
        W[(k_base+1)*n + 1][c_offset] = 1.0
        c_offset += 1
        
        # M7 = (A11-A01)(B01+B11)
        U[1*m + (j_base+1)][c_offset] = 1.0
        U[0*m + (j_base+1)][c_offset] = -1.0
        V[(j_base+0)*p + (k_base+1)][c_offset] = 1.0
        V[(j_base+1)*p + (k_base+1)][c_offset] = 1.0
        W[(k_base+1)*n + 1][c_offset] = 1.0
        c_offset += 1

    # Helper to add schoolbook 2x1x6 decomposition
    # j_base: starting j index (only 1 j)
    def add_schoolbook_2x1x6(j_base):
        nonlocal c_offset
        # i in {0,1}, j = j_base, k in {0..5}
        for i in range(2):
            for k in range(p):
                U[i*m + j_base][c_offset] = 1.0
                V[j_base*p + k][c_offset] = 1.0
                W[k*n + i][c_offset] = 1.0
                c_offset += 1

    # Block 1: j in {0,1}
    # Split k into {0,1}, {2,3}, {4,5}
    add_strassen_2x2x2(0, 0)
    add_strassen_2x2x2(0, 2)
    add_strassen_2x2x2(0, 4)
    
    # Block 2: j in {2,3}
    # Split k into {0,1}, {2,3}, {4,5}
    add_strassen_2x2x2(2, 0)
    add_strassen_2x2x2(2, 2)
    add_strassen_2x2x2(2, 4)
    
    # Block 3: j in {4}
    add_schoolbook_2x1x6(4)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(res["U"].shape, res["V"].shape, res["W"].shape)
