"""Rewritten program for matrix multiplication 3,5,7 with Strassen optimization."""
import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 105
    # We will use Strassen for one 2x2x2 block to reduce rank to 104.
    
    # Initialize with zeros
    # We don't know the final rank yet, so we'll build lists of columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[i * m + j] = 1
        v[j * p + k] = 1
        w[k * n + i] = 1
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    # Helper to add Strassen terms for a 2x2x2 block
    # Block defined by rows i1..i1+1, inner cols j1..j1+1, outer cols k1..k1+1
    def add_strassen(i1, j1, k1):
        # Strassen for C = A*B
        # A is 2x2, B is 2x2
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # We need to map these to our global indices.
        # A indices: (i1, j1), (i1, j1+1), (i1+1, j1), (i1+1, j1+1)
        # B indices: (j1, k1), (j1, k1+1), (j1+1, k1), (j1+1, k1+1)
        # C indices: (i1, k1), (i1, k1+1), (i1+1, k1), (i1+1, k1+1)
        
        # Term 1: (A11 + A22)(B11 + B22)
        u1 = [0] * (n * m)
        v1 = [0] * (m * p)
        w1 = [0] * (p * n)
        u1[i1 * m + j1] = 1
        u1[(i1+1) * m + (j1+1)] = 1
        v1[j1 * p + k1] = 1
        v1[(j1+1) * p + (k1+1)] = 1
        # C11 = M1 + M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to add 7 columns.
        # Col 1: M1
        U_cols.append(u1)
        V_cols.append(v1)
        w1[k1 * n + i1] = 1
        w1[(k1+1) * n + i1] = 1 # M5 part? No, M1 contributes to C11 and C22.
        # Wait, Strassen outputs are linear combinations.
        # We define 7 intermediate products P1..P7.
        # Then C is sum of P's with +/- 1 coefficients.
        # So W matrix will have entries -1, 0, 1.
        
        # Let's define the 7 products explicitly.
        # P1 = (A11 + A22)(B11 + B22)
        # P2 = (A21 + A22)B11
        # P3 = A11(B21 - B22)
        # P4 = A22(B12 - B11)
        # P5 = (A11 + A12)B22
        # P6 = (A21 - A11)(B11 + B12)
        # P7 = (A12 - A22)(B21 + B22)
        
        # We already added P1's U and V. Now add P1's W.
        # P1 contributes to C11 (+1), C22 (+1)
        w1[k1 * n + i1] = 1
        w1[(k1+1) * n + (i1+1)] = 1
        W_cols.append(w1)
        
        # P2 = (A21 + A22)B11
        u2 = [0] * (n * m)
        v2 = [0] * (m * p)
        w2 = [0] * (p * n)
        u2[(i1+1) * m + j1] = 1
        u2[(i1+1) * m + (j1+1)] = 1
        v2[j1 * p + k1] = 1
        # P2 contributes to C21 (+1), C22 (-1)
        w2[k1 * n + (i1+1)] = 1
        w2[(k1+1) * n + (i1+1)] = -1
        U_cols.append(u2)
        V_cols.append(v2)
        W_cols.append(w2)
        
        # P3 = A11(B21 - B22)
        u3 = [0] * (n * m)
        v3 = [0] * (m * p)
        w3 = [0] * (p * n)
        u3[i1 * m + j1] = 1
        v3[(j1+1) * p + k1] = 1
        v3[(j1+1) * p + (k1+1)] = -1
        # P3 contributes to C12 (+1), C22 (+1)
        w3[(k1+1) * n + i1] = 1
        w3[(k1+1) * n + (i1+1)] = 1
        U_cols.append(u3)
        V_cols.append(v3)
        W_cols.append(w3)
        
        # P4 = A22(B12 - B11)
        u4 = [0] * (n * m)
        v4 = [0] * (m * p)
        w4 = [0] * (p * n)
        u4[(i1+1) * m + (j1+1)] = 1
        v4[j1 * p + (k1+1)] = 1
        v4[j1 * p + k1] = -1
        # P4 contributes to C11 (+1), C21 (+1)
        w4[k1 * n + i1] = 1
        w4[k1 * n + (i1+1)] = 1
        U_cols.append(u4)
        V_cols.append(v4)
        W_cols.append(w4)
        
        # P5 = (A11 + A12)B22
        u5 = [0] * (n * m)
        v5 = [0] * (m * p)
        w5 = [0] * (p * n)
        u5[i1 * m + j1] = 1
        u5[i1 * m + (j1+1)] = 1
        v5[(j1+1) * p + (k1+1)] = 1
        # P5 contributes to C11 (+1), C12 (+1)
        w5[k1 * n + i1] = 1
        w5[(k1+1) * n + i1] = 1
        U_cols.append(u5)
        V_cols.append(v5)
        W_cols.append(w5)
        
        # P6 = (A21 - A11)(B11 + B12)
        u6 = [0] * (n * m)
        v6 = [0] * (m * p)
        w6 = [0] * (p * n)
        u6[(i1+1) * m + j1] = 1
        u6[i1 * m + j1] = -1
        v6[j1 * p + k1] = 1
        v6[j1 * p + (k1+1)] = 1
        # P6 contributes to C22 (+1)
        w6[(k1+1) * n + (i1+1)] = 1
        U_cols.append(u6)
        V_cols.append(v6)
        W_cols.append(w6)
        
        # P7 = (A12 - A22)(B21 + B22)
        u7 = [0] * (n * m)
        v7 = [0] * (m * p)
        w7 = [0] * (p * n)
        u7[i1 * m + (j1+1)] = 1
        u7[(i1+1) * m + (j1+1)] = -1
        v7[(j1+1) * p + k1] = 1
        v7[(j1+1) * p + (k1+1)] = 1
        # P7 contributes to C11 (-1)
        w7[k1 * n + i1] = -1
        U_cols.append(u7)
        V_cols.append(v7)
        W_cols.append(w7)

    # Use Strassen for block (0,0,0) i.e., rows 0-1, inner 0-1, outer 0-1
    add_strassen(0, 0, 0)
    
    # Use Schoolbook for all other terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the terms covered by the Strassen block
                if i < 2 and j < 2 and k < 2:
                    continue
                add_schoolbook(i, j, k)
                
    rank = len(U_cols)
    U = [col for col in U_cols]
    V = [col for col in V_cols]
    W = [col for col in W_cols]
    
    return {"U": U, "V": V, "W": W}
