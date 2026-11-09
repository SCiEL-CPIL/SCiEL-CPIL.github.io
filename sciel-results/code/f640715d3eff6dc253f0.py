import numpy as np
from fractions import Fraction
import time

def construct():
    n, m, p = 2, 4, 5
    deadline = time.time() + 110

    def strassen_2x2x2_decomp():
        """Return U, V, W for 2x2x2 with rank 7 (Strassen)."""
        n2, m2, p2 = 2, 2, 2
        rank = 7
        U = np.zeros((n2*m2, rank), dtype=object)
        V = np.zeros((m2*p2, rank), dtype=object)
        W = np.zeros((p2*n2, rank), dtype=object)
        
        # U indices: [A00, A01, A10, A11] -> [0,1,2,3]
        # V indices: [B00, B01, B10, B11] -> [0,1,2,3]
        # W indices: [C00, C01, C10, C11] -> [0,1,2,3] where Cki -> index k*2+i
        
        # P1 = (A00+A11)*(B00+B11)
        U[0, 0] = Fraction(1); U[3, 0] = Fraction(1)
        V[0, 0] = Fraction(1); V[3, 0] = Fraction(1)
        W[0, 0] = Fraction(1); W[3, 0] = Fraction(1)
        
        # P2 = (A10+A11)*B00
        U[2, 1] = Fraction(1); U[3, 1] = Fraction(1)
        V[0, 1] = Fraction(1)
        W[2, 1] = Fraction(1)  # C10
        
        # P3 = A00*(B01-B11)
        U[0, 2] = Fraction(1)
        V[1, 2] = Fraction(1); V[3, 2] = Fraction(-1)
        W[1, 2] = Fraction(1)  # C01
        
        # P4 = A11*(B10-B00)
        U[3, 3] = Fraction(1)
        V[2, 3] = Fraction(1); V[0, 3] = Fraction(-1)
        W[0, 3] = Fraction(1); W[2, 3] = Fraction(1)  # C00, C10
        
        # P5 = (A00+A01)*B11
        U[0, 4] = Fraction(1); U[1, 4] = Fraction(1)
        V[3, 4] = Fraction(1)
        W[0, 4] = Fraction(-1); W[1, 4] = Fraction(1)  # C00 with -1, C01 with +1
        
        # P6 = (A10-A00)*(B00+B01)
        U[2, 5] = Fraction(1); U[0, 5] = Fraction(-1)
        V[0, 5] = Fraction(1); V[1, 5] = Fraction(1)
        W[3, 5] = Fraction(1)  # C11
        
        # P7 = (A01-A11)*(B10+B11)
        U[1, 6] = Fraction(1); U[3, 6] = Fraction(-1)
        V[2, 6] = Fraction(1); V[3, 6] = Fraction(1)
        W[0, 6] = Fraction(1)  # C00
        
        return U, V, W
    
    def trivial_2x2x1_decomp():
        """Return U, V, W for 2x2x1 with rank 4 (trivial)."""
        n2, m2, p2 = 2, 2, 1
        rank = 4
        U = np.zeros((n2*m2, rank), dtype=object)
        V = np.zeros((m2*p2, rank), dtype=object)
        W = np.zeros((p2*n2, rank), dtype=object)
        
        for i in range(2):
            for j in range(2):
                idx = i * 2 + j  # 0,1,2,3
                U[idx, idx] = Fraction(1)
                V[j * 1 + 0, idx] = Fraction(1)
                W[0 * 2 + i, idx] = Fraction(1)
        
        return U, V, W
    
    def decomp_2x2x5():
        """Return U, V, W for 2x2x5 with rank 18."""
        n2, m2, p2 = 2, 2, 5
        rank = 18
        
        U = np.zeros((n2*m2, rank), dtype=object)
        V = np.zeros((m2*p2, rank), dtype=object)
        W = np.zeros((p2*n2, rank), dtype=object)
        
        # Block 1: columns 0,1 (indices 0,1 in k)
        U1, V1, W1 = strassen_2x2x2_decomp()
        for r in range(7):
            for ij in range(4):
                if U1[ij, r] != 0:
                    U[ij, r] += U1[ij, r]
            for jk in range(4):
                j = jk // 2
                k = jk % 2
                if V1[jk, r] != 0:
                    V[j*5+k, r] += V1[jk, r]
            for ki in range(4):
                k = ki // 2
                i = ki % 2
                if W1[ki, r] != 0:
                    W[k*2+i, r] += W1[ki, r]
        
        # Block 2: columns 2,3 (indices 2,3 in k)
        U2, V2, W2 = strassen_2x2x2_decomp()
        for r in range(7):
            col_offset = 7
            r2 = col_offset + r
            for ij in range(4):
                if U2[ij, r] != 0:
                    U[ij, r2] += U2[ij, r]
            for jk in range(4):
                j = jk // 2
                k = jk % 2
                if V2[jk, r] != 0:
                    V[j*5+(k+2), r2] += V2[jk, r]
            for ki in range(4):
                k = ki // 2
                i = ki % 2
                if W2[ki, r] != 0:
                    W[(k+2)*2+i, r2] += W2[ki, r]
        
        # Block 3: column 4
        U3, V3, W3 = trivial_2x2x1_decomp()
        for r in range(4):
            col_offset = 14
            r3 = col_offset + r
            for ij in range(4):
                if U3[ij, r] != 0:
                    U[ij, r3] += U3[ij, r]
            for j in range(2):
                if V3[j, r] != 0:
                    V[j*5+4, r3] += V3[j, r]
            for i in range(2):
                if W3[i, r] != 0:
                    W[4*2+i, r3] += W3[i, r]
        
        return U, V, W
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Verify that the decomposition exactly computes the matrix multiplication tensor"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if total != 1:
                        return False
        return True
    
    # Build 2x4x5 from two 2x2x5 blocks
    # A is 2x4: A[:, 0:2] and A[:, 2:4]
    # B is 4x5: B[0:2, :] and B[2:4, :]
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    
    U1, V1, W1 = decomp_2x2x5()
    U2, V2, W2 = decomp_2x2x5()
    
    rank = 36
    U_out = np.zeros((2*4, rank), dtype=object)
    V_out = np.zeros((4*5, rank), dtype=object)
    W_out = np.zeros((5*2, rank), dtype=object)
    
    # Block 1: j in {0,1} (first 2 columns of A, first 2 rows of B)
    for r in range(18):
        for ij in range(4):  # i in {0,1}, j_local in {0,1}
            i = ij // 2
            j_local = ij % 2
            j = j_local  # j_global = 0,1
            if U1[ij, r] != 0:
                U_out[i*4+j, r] += U1[ij, r]
        for jk in range(10):  # j_local in {0,1}, k in {0,1,2,3,4}
            j_local = jk // 5
            k = jk % 5
            j = j_local  # j_global = 0,1
            if V1[jk, r] != 0:
                V_out[j*5+k, r] += V1[jk, r]
        for ki in range(10):  # k in {0,1,2,3,4}, i in {0,1}
            k = ki // 2
            i = ki % 2
            if W1[ki, r] != 0:
                W_out[k*2+i, r] += W1[ki, r]
    
    # Block 2: j in {2,3} (last 2 columns of A, last 2 rows of B)
    for r in range(18):
        r2 = 18 + r
        for ij in range(4):
            i = ij // 2
            j_local = ij % 2
            j = j_local + 2  # j_global = 2,3
            if U2[ij, r] != 0:
                U_out[i*4+j, r2] += U2[ij, r]
        for jk in range(10):
            j_local = jk // 5
            k = jk % 5
            j = j_local + 2  # j_global = 2,3
            if V2[jk, r] != 0:
                V_out[j*5+k, r2] += V2[jk, r]
        for ki in range(10):
            k = ki // 2
            i = ki % 2
            if W2[ki, r] != 0:
                W_out[k*2+i, r2] += W2[ki, r]
    
    # Verify
    if not verify_decomposition(U_out, V_out, W_out, 36, n, m, p):
        # Fallback to trivial rank-40
        U_out = np.zeros((8, 40), dtype=object)
        V_out = np.zeros((20, 40), dtype=object)
        W_out = np.zeros((10, 40), dtype=object)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U_out[i * m + j, idx] = Fraction(1)
                    V_out[j * p + k, idx] = Fraction(1)
                    W_out[k * n + i, idx] = Fraction(1)
    
    def convert_to_list(arr):
        res = []
        for row in arr:
            res_row = []
            for val in row:
                if isinstance(val, Fraction):
                    if val.denominator == 1:
                        res_row.append(int(val))
                    elif val.denominator == 2:
                        res_row.append(val.numerator / 2.0)
                    else:
                        res_row.append(float(val))
                else:
                    res_row.append(val)
            res.append(res_row)
        return res
    
    return {"U": convert_to_list(U_out), "V": convert_to_list(V_out), "W": convert_to_list(W_out)}
