import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add Strassen block
    # a_rows: list of 2 global row indices for A (i)
    # a_cols: list of 2 global col indices for A (j)
    # b_rows: list of 2 global row indices for B (j)
    # b_cols: list of 2 global col indices for B (k)
    # c_rows: list of 2 global row indices for C (k) -- wait, C is p x n? No, C is n x p?
    # Interface: W shape (p*n, rank). Index k*n + i.
    # So W[k, i] corresponds to C[i, k]?
    # Standard: C[i, k] = sum_j A[i, j] B[j, k].
    # W index is k*n + i. So W is flattened C^T? Or C with k major?
    # If W is (p*n), index k*n+i.
    # The term is W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c].
    # This sums to C[i, k]?
    # Let's check the seed code logic again.
    # Seed: W[k*n+i, col] = 1.
    # And target check: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    # If this equals delta_jk, it's not C[i,k].
    # But for schoolbook, we established it works for C[i,k] if we interpret the sum correctly.
    # Actually, the sum S[i,j,k] = 1 for all i,j,k in schoolbook.
    # The bilinear form is sum_c W[k,i,c] u_c(A) v_c(B).
    # If we want C[i,k], we need the coefficient of C[i,k] in the sum.
    # The term W[k,i,c] multiplies the product.
    # So W[k,i,c] IS the weight for C[i,k] from term c.
    # So W is indexed by (k, i).
    
    def add_strassen(a_i, a_j, b_j, b_k, c_k, c_i):
        nonlocal col
        # Strassen 7 products
        # P1 = A11 (B12 - B22)
        # P2 = (A11 + A12) B22
        # P3 = (A21 + A22) B11
        # P4 = A22 (B21 - B11)
        # P5 = (A11 + A22) (B11 + B22)
        # P6 = (A12 - A22) (B21 + B22)
        # P7 = (A11 - A21) (B11 + B12)
        
        # C11 = P5 + P4 - P2 + P6
        # C12 = P1 + P5
        # C21 = P3 + P7
        # C22 = P5 + P1 - P3 + P4
        
        # Indices:
        # A11 -> A[a_i[0], a_j[0]]
        # A12 -> A[a_i[0], a_j[1]]
        # A21 -> A[a_i[1], a_j[0]]
        # A22 -> A[a_i[1], a_j[1]]
        
        # B11 -> B[b_j[0], b_k[0]]
        # B12 -> B[b_j[0], b_k[1]]
        # B21 -> B[b_j[1], b_k[0]]
        # B22 -> B[b_j[1], b_k[1]]
        
        # C11 -> C[c_i[0], c_k[0]] -> W[c_k[0]*n + c_i[0]]
        # C12 -> C[c_i[0], c_k[1]] -> W[c_k[1]*n + c_i[0]]
        # C21 -> C[c_i[1], c_k[0]] -> W[c_k[0]*n + c_i[1]]
        # C22 -> C[c_i[1], c_k[1]] -> W[c_k[1]*n + c_i[1]]
        
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # P1
            ({(a_i[0], a_j[0]): 1}, {(b_j[0], b_k[1]): 1, (b_j[1], b_k[1]): -1}, {(c_k[0], c_i[0]): 1, (c_k[1], c_i[0]): 1, (c_k[0], c_i[1]): 0, (c_k[1], c_i[1]): 0}), # C12 part? No, P1 is in C12 and C22.
            # Wait, let's just map the 7 P's to the 4 C's.
        ]
        
        # It's easier to just define the 7 columns directly.
        
        # P1
        U[a_i[0]*m + a_j[0], col] = 1
        V[b_j[0]*p + b_k[1], col] = 1
        V[b_j[1]*p + b_k[1], col] = -1
        W[c_k[1]*n + c_i[0], col] = 1 # C12
        W[c_k[1]*n + c_i[1], col] = 1 # C22
        col += 1
        
        # P2
        U[a_i[0]*m + a_j[0], col] = 1
        U[a_i[0]*m + a_j[1], col] = 1
        V[b_j[1]*p + b_k[1], col] = 1
        W[c_k[0]*n + c_i[0], col] = -1 # C11
        col += 1
        
        # P3
        U[a_i[1]*m + a_j[0], col] = 1
        U[a_i[1]*m + a_j[1], col] = 1
        V[b_j[0]*p + b_k[0], col] = 1
        W[c_k[0]*n + c_i[1], col] = 1 # C21
        col += 1
        
        # P4
        U[a_i[1]*m + a_j[1], col] = 1
        V[b_j[1]*p + b_k[0], col] = 1
        V[b_j[0]*p + b_k[0], col] = -1
        W[c_k[0]*n + c_i[0], col] = 1 # C11
        W[c_k[1]*n + c_i[1], col] = 1 # C22
        col += 1
        
        # P5
        U[a_i[0]*m + a_j[0], col] = 1
        U[a_i[1]*m + a_j[1], col] = 1
        V[b_j[0]*p + b_k[0], col] = 1
        V[b_j[1]*p + b_k[1], col] = 1
        W[c_k[0]*n + c_i[0], col] = 1 # C11
        W[c_k[1]*n + c_i[0], col] = 1 # C12
        W[c_k[1]*n + c_i[1], col] = 1 # C22
        col += 1
        
        # P6
        U[a_i[0]*m + a_j[1], col] = 1
        U[a_i[1]*m + a_j[1], col] = -1
        V[b_j[1]*p + b_k[0], col] = 1
        V[b_j[1]*p + b_k[1], col] = 1
        W[c_k[0]*n + c_i[0], col] = 1 # C11
        col += 1
        
        # P7
        U[a_i[0]*m + a_j[0], col] = 1
        U[a_i[1]*m + a_j[0], col] = -1
        V[b_j[0]*p + b_k[0], col] = 1
        V[b_j[0]*p + b_k[1], col] = 1
        W[c_k[0]*n + c_i[1], col] = 1 # C21
        col += 1

    # Helper to add Schoolbook 2x2x1 block
    def add_schoolbook_2x2x1(a_i, a_j, b_j, b_k, c_k, c_i):
        nonlocal col
        # C[i, k] = A[i, j1] B[j1, k] + A[i, j2] B[j2, k]
        # Here b_k is a single index.
        # c_k is a single index.
        # a_i has 2 indices, a_j has 2 indices, b_j has 2 indices.
        
        # Term 1: A[i, j1] B[j1, k] -> C[i, k]
        for idx_i in range(2):
            U[a_i[idx_i]*m + a_j[0], col] = 1
            V[b_j[0]*p + b_k, col] = 1
            W[c_k*n + c_i[idx_i], col] = 1
            col += 1
            
        # Term 2: A[i, j2] B[j2, k] -> C[i, k]
        for idx_i in range(2):
            U[a_i[idx_i]*m + a_j[1], col] = 1
            V[b_j[1]*p + b_k, col] = 1
            W[c_k*n + c_i[idx_i], col] = 1
            col += 1

    # Block 1: A1 (i 0,1; j 0,1) * B1,1 (j 0,1; k 0,1) -> C (i 0,1; k 0,1)
    add_strassen([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # Block 2: A1 (i 0,1; j 0,1) * B1,2 (j 0,1; k 2,3) -> C (i 0,1; k 2,3)
    add_strassen([0,1], [0,1], [0,1], [2,3], [2,3], [0,1])
    
    # Block 3: A1 (i 0,1; j 0,1) * B1,3 (j 0,1; k 4) -> C (i 0,1; k 4)
    add_schoolbook_2x2x1([0,1], [0,1], [0,1], 4, 4, [0,1])
    
    # Block 4: A2 (i 0,1; j 2,3) * B2,1 (j 2,3; k 0,1) -> C (i 0,1; k 0,1)
    add_strassen([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    
    # Block 5: A2 (i 0,1; j 2,3) * B2,2 (j 2,3; k 2,3) -> C (i 0,1; k 2,3)
    add_strassen([0,1], [2,3], [2,3], [2,3], [2,3], [0,1])
    
    # Block 6: A2 (i 0,1; j 2,3) * B2,3 (j 2,3; k 4) -> C (i 0,1; k 4)
    add_schoolbook_2x2x1([0,1], [2,3], [2,3], 4, 4, [0,1])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

