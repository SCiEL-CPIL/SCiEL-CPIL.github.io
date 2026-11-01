import numpy as np
import time

def construct():
    # Dimensions
    n, m, p = 2, 2, 6
    rank = 21
    
    # Initialize factor matrices with zeros
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen 7-term decomposition coefficients for 2x2 matrix mult
    # Each term is a tuple (u_vec, v_vec, w_vec)
    # u_vec, v_vec, w_vec are length-4 vectors corresponding to indices in [0, 1, 2, 3]
    # Indexing convention for 2x2 matrices:
    # A = [[a00, a01], [a10, a11]] -> flat [a00, a01, a10, a11]
    # B = [[b00, b01], [b10, b11]] -> flat [b00, b01, b10, b11]
    # C = [[c00, c01], [c10, c11]] -> flat [c00, c01, c10, c11]
    
    # The 7 products P1..P7:
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b10-b11)
    # P4 = a11(b01-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b10)
    # P7 = (a01-a11)(b11+b01)
    
    # Extracting coefficients for A, B, C
    # A coeffs:
    # P1: 1, 0, 0, 1
    # P2: 0, 0, 1, 1
    # P3: 1, 0, 0, 0
    # P4: 0, 0, 0, 1
    # P5: 1, 1, 0, 0
    # P6: -1, 0, 1, 0
    # P7: 0, 1, 0, -1
    
    # B coeffs:
    # P1: 1, 0, 0, 1
    # P2: 1, 0, 0, 0
    # P3: 0, 1, 0, -1
    # P4: -1, 0, 1, 0
    # P5: 0, 0, 0, 1
    # P6: 1, 0, 1, 0
    # P7: 0, 1, 1, 1  <- Wait, let's re-verify P7
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B21)
    # M7 = (A12-A22)(B12+B22)
    
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Let's map indices:
    # A11 -> A[0], A12 -> A[1], A21 -> A[2], A22 -> A[3]
    # B11 -> B[0], B12 -> B[1], B21 -> B[2], B22 -> B[3]
    # C11 -> C[0], C12 -> C[1], C21 -> C[2], C22 -> C[3]
    
    # M1: A[0]+A[3], B[0]+B[3]. C[0]+=1, C[3]+=1
    # M2: A[2]+A[3], B[0]. C[2]+=1, C[3]-=1
    # M3: A[0], B[2]-B[3]. C[1]+=1, C[3]+=1
    # M4: A[3], B[1]-B[0]. C[0]+=1, C[2]+=1
    # M5: A[0]+A[1], B[3]. C[0]-=1, C[1]+=1
    # M6: A[2]-A[0], B[0]+B[2]. C[3]+=1
    # M7: A[1]-A[3], B[1]+B[3]. C[0]+=1
    
    strassen_tensors = [
        # (U_coeffs, V_coeffs, W_coeffs)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 1, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    # Fill U, V, W for the 3 blocks
    # Block b (0, 1, 2) corresponds to B columns 2b, 2b+1 and C rows 2b, 2b+1 (wait, C is 2x6, so rows 0,1, cols 2b, 2b+1)
    # A is shared.
    
    for b in range(3):
        # U is the same for all blocks, but placed in different columns of the full U matrix?
        # No, the tensor decomposition is C = sum_k U[:,k] V[:,k] W[:,k]^T (conceptually)
        # The rank is the number of terms.
        # For block b, we use 7 terms.
        # The U vector for term s in block b is the same as Strassen U_s, but it contributes to the product.
        # In the full problem, A is a single 2x2 matrix.
        # So for each block b, the U factor is identical (the A coefficients).
        # But in the factorization C = U V W^T, the columns of U, V, W must align.
        # Term k (0 to 20):
        # If k is in block b (k = 7b + s), then:
        # U[:, k] = Strassen_U_s
        # V[:, k] = Strassen_V_s placed in the rows corresponding to block b of B.
        # W[:, k] = Strassen_W_s placed in the rows corresponding to block b of C.
        
        for s in range(7):
            k = 7 * b + s
            u_coeff, v_coeff, w_coeff = strassen_tensors[s]
            
            # Fill U
            for i in range(4):
                if u_coeff[i] != 0:
                    U[i, k] = u_coeff[i]
            
            # Fill V
            # V is 12x21. Rows are j*6 + l. j in 0,1. l in 0..5.
            # Block b uses l = 2b, 2b+1.
            # Local indices 0,1 -> l=2b, 2b+1 for j=0?
            # Let's check the mapping in the previous code:
            # v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            # This maps local 0 -> 2b (j=0, l=2b)
            # local 1 -> 2b+1 (j=0, l=2b+1)
            # local 2 -> 6+2b (j=1, l=2b)
            # local 3 -> 6+2b+1 (j=1, l=2b+1)
            # This matches the Strassen B vector [B11, B12, B21, B22] mapping to [b_{0,2b}, b_{0,2b+1}, b_{1,2b}, b_{1,2b+1}].
            
            v_indices = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for i in range(4):
                if v_coeff[i] != 0:
                    V[v_indices[i], k] = v_coeff[i]
            
            # Fill W
            # W is 12x21. Rows are l*n + i. l in 0..5, i in 0,1.
            # Block b uses l = 2b, 2b+1.
            # Previous mapping:
            # w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
            # local 0 -> 4b (l=2b, i=0) -> 2*(2b)+0
            # local 1 -> 4b+1 (l=2b, i=1) -> 2*(2b)+1
            # local 2 -> 4b+2 (l=2b+1, i=0) -> 2*(2b+1)+0
            # local 3 -> 4b+3 (l=2b+1, i=1) -> 2*(2b+1)+1
            # This matches Strassen C vector [C11, C12, C21, C22] mapping to [c_{2b,0}, c_{2b,1}, c_{2b+1,0}, c_{2b+1,1}].
            
            w_indices = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for i in range(4):
                if w_coeff[i] != 0:
                    W[w_indices[i], k] = w_coeff[i]
    
    return {"U": U, "V": V, "W": W}
