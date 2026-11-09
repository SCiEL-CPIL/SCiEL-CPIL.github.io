import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Laderman decomposition for 3x3x3
    # Products P_k = X_k(Y_k) where X_k is linear in A, Y_k is linear in B
    # We represent X_k and Y_k as vectors of coefficients for A and B entries
    # A entries: A[0,0], A[0,1], A[0,2], A[1,0], ... A[2,2] (9 entries)
    # B entries: B[0,0], B[0,1], B[0,2], B[1,0], ... B[2,2] (9 entries)
    
    # Index mapping for A: i*m + j
    # Index mapping for B: j*m + k (for 3x3)
    
    A_idx = lambda i, j: i * 3 + j
    B_idx = lambda j, k: j * 3 + k
    
    # Define the 23 products
    # Each product is a pair (X_coeffs, Y_coeffs)
    # X_coeffs is a vector of length 9
    # Y_coeffs is a vector of length 9
    
    products = []
    
    # Helper to create a vector with 1 at index
    def vec(idx):
        v = [0.0] * 9
        v[idx] = 1.0
        return v
    
    # 1-9: Diagonal products
    for i in range(3):
        for j in range(3):
            X = vec(A_idx(i, j))
            Y = vec(B_idx(j, i)) # Note: B index is (row, col) = (j, k) in AB? 
            # Wait, C[i,k] = sum_j A[i,j] B[j,k].
            # So B entry is B[j,k].
            # In the product A[i,j] * B[j,k], the B index is (j,k).
            # My B_idx(j,k) maps to j*3+k.
            # So for A[i,j] * B[j,k], X is vec(A_idx(i,j)), Y is vec(B_idx(j,k)).
            # The products above were A[i,j] * B[j,i]? No, they were specific pairs.
            # Let's re-read the Laderman terms.
            # P1 = A11 B11. i=0,j=0. B index (0,0).
            # P2 = A12 B21. i=0,j=1. B index (1,0).
            # P3 = A13 B31. i=0,j=2. B index (2,0).
            # P4 = A21 B12. i=1,j=0. B index (0,1).
            # P5 = A22 B22. i=1,j=1. B index (1,1).
            # P6 = A23 B32. i=1,j=2. B index (2,1).
            # P7 = A31 B13. i=2,j=0. B index (0,2).
            # P8 = A32 B23. i=2,j=1. B index (1,2).
            # P9 = A33 B33. i=2,j=2. B index (2,2).
            
            # My loop:
            # i=0, j=0 -> A[0,0] * B[0,0]. Correct.
            # i=0, j=1 -> A[0,1] * B[1,0]. Correct.
            # i=0, j=2 -> A[0,2] * B[2,0]. Correct.
            # i=1, j=0 -> A[1,0] * B[0,1]. Correct.
            # i=1, j=1 -> A[1,1] * B[1,1]. Correct.
            # i=1, j=2 -> A[1,2] * B[2,1]. Correct.
            # i=2, j=0 -> A[2,0] * B[0,2]. Correct.
            # i=2, j=1 -> A[2,1] * B[1,2]. Correct.
            # i=2, j=2 -> A[2,2] * B[2,2]. Correct.
            
            products.append((X, Y))
            
    # 10-15: 2x2 blocks
    # P10 = (A11+A22)(B11+B22)
    X = [0.0]*9; X[A_idx(0,0)]=1; X[A_idx(1,1)]=1
    Y = [0.0]*9; Y[B_idx(0,0)]=1; Y[B_idx(1,1)]=1
    products.append((X, Y))
    
    # P11 = (A11+A33)(B11+B33)
    X = [0.0]*9; X[A_idx(0,0)]=1; X[A_idx(2,2)]=1
    Y = [0.0]*9; Y[B_idx(0,0)]=1; Y[B_idx(2,2)]=1
    products.append((X, Y))
    
    # P12 = (A22+A33)(B22+B33)
    X = [0.0]*9; X[A_idx(1,1)]=1; X[A_idx(2,2)]=1
    Y = [0.0]*9; Y[B_idx(1,1)]=1; Y[B_idx(2,2)]=1
    products.append((X, Y))
    
    # P13 = (A12+A21)(B12+B21)
    X = [0.0]*9; X[A_idx(0,1)]=1; X[A_idx(1,0)]=1
    Y = [0.0]*9; Y[B_idx(1,0)]=1; Y[B_idx(0,1)]=1
    products.append((X, Y))
    
    # P14 = (A13+A31)(B13+B31)
    X = [0.0]*9; X[A_idx(0,2)]=1; X[A_idx(2,0)]=1
    Y = [0.0]*9; Y[B_idx(0,2)]=1; Y[B_idx(2,0)]=1
    products.append((X, Y))
    
    # P15 = (A23+A32)(B23+B32)
    X = [0.0]*9; X[A_idx(1,2)]=1; X[A_idx(2,1)]=1
    Y = [0.0]*9; Y[B_idx(1,2)]=1; Y[B_idx(2,1)]=1
    products.append((X, Y))
    
    # 16-23: 3x3 blocks
    # P16 = (A11+A12+A21)(B11+B12+B21)
    X = [0.0]*9; X[A_idx(0,0)]=1; X[A_idx(0,1)]=1; X[A_idx(1,0)]=1
    Y = [0.0]*9; Y[B_idx(0,0)]=1; Y[B_idx(1,0)]=1; Y[B_idx(0,1)]=1
    products.append((X, Y))
    
    # P17 = (A11+A13+A31)(B11+B13+B31)
    X = [0.0]*9; X[A_idx(0,0)]=1; X[A_idx(0,2)]=1; X[A_idx(2,0)]=1
    Y = [0.0]*9; Y[B_idx(0,0)]=1; Y[B_idx(0,2)]=1; Y[B_idx(2,0)]=1
    products.append((X, Y))
    
    # P18 = (A22+A23+A32)(B22+B23+B32)
    X = [0.0]*9; X[A_idx(1,1)]=1; X[A_idx(1,2)]=1; X[A_idx(2,1)]=1
    Y = [0.0]*9; Y[B_idx(1,1)]=1; Y[B_idx(1,2)]=1; Y[B_idx(2,1)]=1
    products.append((X, Y))
    
    # P19 = (A33+A31+A13)(B33+B31+B13)
    X = [0.0]*9; X[A_idx(2,2)]=1; X[A_idx(2,0)]=1; X[A_idx(0,2)]=1
    Y = [0.0]*9; Y[B_idx(2,2)]=1; Y[B_idx(2,0)]=1; Y[B_idx(0,2)]=1
    products.append((X, Y))
    
    # P20 = (A12+A23+A31)(B12+B23+B31)
    X = [0.0]*9; X[A_idx(0,1)]=1; X[A_idx(1,2)]=1; X[A_idx(2,0)]=1
    Y = [0.0]*9; Y[B_idx(1,0)]=1; Y[B_idx(1,2)]=1; Y[B_idx(2,0)]=1
    products.append((X, Y))
    
    # P21 = (A21+A32+A13)(B21+B32+B13)
    X = [0.0]*9; X[A_idx(1,0)]=1; X[A_idx(2,1)]=1; X[A_idx(0,2)]=1
    Y = [0.0]*9; Y[B_idx(0,1)]=1; Y[B_idx(2,1)]=1; Y[B_idx(0,2)]=1
    products.append((X, Y))
    
    # P22 = (A11+A23+A32)(B11+B23+B32)
    X = [0.0]*9; X[A_idx(0,0)]=1; X[A_idx(1,2)]=1; X[A_idx(2,1)]=1
    Y = [0.0]*9; Y[B_idx(0,0)]=1; Y[B_idx(1,2)]=1; Y[B_idx(2,1)]=1
    products.append((X, Y))
    
    # P23 = (A22+A31+A13)(B22+B31+B13)
    X = [0.0]*9; X[A_idx(1,1)]=1; X[A_idx(2,0)]=1; X[A_idx(0,2)]=1
    Y = [0.0]*9; Y[B_idx(1,1)]=1; Y[B_idx(2,0)]=1; Y[B_idx(0,2)]=1
    products.append((X, Y))
    
    # Now we have 23 products.
    # We need to check if they can form the 9 outputs of 3x3x3 multiplication.
    # Output C[i,k] = sum_j A[i,j] B[j,k].
    # We want C[i,k] = sum_{l=0}^{22} R[i,k,l] P_l.
    
    # This is a linear system. We can solve it by least squares or exact check.
    # Since the coefficients are integers, we can use Fraction or just float with high precision.
    # Let's use numpy to check if the outputs are in the span.
    
    # Construct the tensor for each product P_l
    # P_l is a 3x3x3 tensor T_l[i,j,k] = X_l[i,j] * Y_l[j,k]
    # Wait, X_l is a linear form in A. Y_l is a linear form in B.
    # The product is X_l(A) * Y_l(B).
    # This is a rank-1 tensor in the space of (A,B) pairs?
    # No, it's a bilinear form.
    # We can represent each P_l as a 3x3x3 matrix M_l where M_l[i,k] is the coefficient of A[i,j]B[j,k]?
    # No, P_l is a scalar.
    # The output C is a 3x3 matrix.
    # We want C = sum R_l P_l.
    # This means for each (i,k), C[i,k] = sum_l R_l[i,k] P_l.
    
    # Let's evaluate this at random A, B to check consistency?
    # Or better, check the tensor identity.
    # T_C[i,k] = sum_j A[i,j] B[j,k].
    # T_P_l = X_l(A) Y_l(B).
    # We need T_C[i,k] = sum_l R_l[i,k] T_P_l.
    
    # This is equivalent to saying that the tensor T_C is in the span of T_P_l.
    # We can check this by flattening the tensors and checking rank?
    # Or simply, for each (i,k), the bilinear form C[i,k] must be a linear combination of the bilinear forms P_l.
    
    # Let's try to solve for R.
    # We have 9 outputs. Each output is a bilinear form.
    # Each P_l is a bilinear form.
    # We can represent a bilinear form as a 9x9 matrix (vector of 81 coeffs).
    # Let's do that.
    
    def bilinear_to_vec(X, Y):
        # X is 9x1, Y is 9x1
        # The bilinear form is X^T A Y^T B? No.
        # X is coeffs for A. Y is coeffs for B.
        # The form is (sum_i X_i A_i) * (sum_j Y_j B_j).
        # The coefficient of A_i B_j is X_i Y_j.
        vec = [0.0] * 81
        for i in range(9):
            for j in range(9):
                vec[i * 9 + j] = X[i] * Y[j]
        return vec
    
    P_vecs = []
    for X, Y in products:
        P_vecs.append(bilinear_to_vec(X, Y))
    P_matrix = np.array(P_vecs).T # 81 x 23
    
    # Now for each output C[i,k], construct its bilinear form vector.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Coeff of A[a] B[b] is 1 if a=i*3+j and b=j*3+k, else 0.
    # So we need to find j such that a=i*3+j and b=j*3+k.
    # This implies j = a-i*3 and j = b/3? No.
    # a = i*3 + j. b = j*3 + k.
    # So for a given (i,k), the non-zero coeffs are at (a,b) where a=i*3+j, b=j*3+k for j=0,1,2.
    
    C_vecs = []
    for i in range(3):
        for k in range(3):
            vec = [0.0] * 81
            for j in range(3):
                a = i * 3 + j
                b = j * 3 + k
                vec[a * 9 + b] = 1.0
            C_vecs.append(vec)
            
    C_matrix = np.array(C_vecs).T # 81 x 9
    
    # We want C_matrix = P_matrix @ R_matrix
    # R_matrix is 23 x 9.
    # We can solve for R using least squares.
    R_solution, residuals, rank, s = np.linalg.lstsq(P_matrix, C_matrix, rcond=None)
    
    # Check if the solution is exact
    if np.allclose(P_matrix @ R_solution, C_matrix):
        # The Laderman decomposition is valid!
        # Construct U, V, W for 3x3x6
        # Rank = 46
        rank = 46
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # First 23 columns for B1 (k=0,1,2)
        for l in range(23):
            col = l
            X, Y = products[l]
            # Fill U
            for idx in range(9):
                if X[idx] != 0:
                    U[idx][col] = X[idx]
            # Fill V (for B1)
            for idx in range(9):
                if Y[idx] != 0:
                    # B1 index j,k maps to V index j*p + k = j*6 + k
                    j = idx // 3
                    k = idx % 3
                    V[j * 6 + k][col] = Y[idx]
            # Fill W
            # W[k*n+i, col] = R_solution[l, i*3+k]
            for i in range(3):
                for k in range(3):
                    W[k * 3 + i][col] = R_solution[l, i * 3 + k]
                    
        # Next 23 columns for B2 (k=3,4,5)
        for l in range(23):
            col = 23 + l
            X, Y = products[l]
            # Fill U (same X)
            for idx in range(9):
                if X[idx] != 0:
                    U[idx][col] = X[idx]
            # Fill V (for B2)
            for idx in range(9):
                if Y[idx] != 0:
                    j = idx // 3
                    k = idx % 3
                    # B2 index j,k maps to V index j*p + (k+3) = j*6 + k + 3
                    V[j * 6 + k + 3][col] = Y[idx]
            # Fill W
            # W[k*n+i, col] = R_solution[l, i*3+k]
            # But output k is 3,4,5.
            for i in range(3):
                for k in range(3):
                    W[(k + 3) * 3 + i][col] = R_solution[l, i * 3 + k]
                    
        return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
