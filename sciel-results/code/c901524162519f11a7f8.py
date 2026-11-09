import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0
    
    # Part 1: 3x3x3 using Laderman (Rank 19)
    # Part 2: 3x3x4 using Naive (Rank 36)
    rank1 = 19
    rank2 = n * m * (p - 3)
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Laderman U and V definitions
    # A indices: 0..8 (row major)
    # B indices: 0..8 (row major)
    
    laderman_U = [
        [1,0,0, 0,1,0, 0,0,1],
        [1,0,0, 0,1,0, 0,0,-1],
        [1,0,0, 0,-1,0, 0,0,1],
        [1,0,0, 0,-1,0, 0,0,-1],
        [0,1,0, 1,0,0, 0,0,1],
        [0,1,0, 1,0,0, 0,0,-1],
        [0,1,0, -1,0,0, 0,0,1],
        [0,1,0, -1,0,0, 0,0,-1],
        [0,0,1, 0,0,1, 0,0,1],
        [0,0,1, 0,0,1, 0,0,-1],
        [0,0,1, 0,0,-1, 0,0,1],
        [0,0,1, 0,0,-1, 0,0,-1],
        [1,1,0, 0,0,0, 0,0,0],
        [0,0,1, 0,1,0, 0,0,0],
        [0,0,0, 1,0,0, 1,0,0],
        [0,0,0, 0,0,1, 0,1,0],
        [1,0,1, 0,0,0, 0,0,0],
        [0,1,0, 0,0,0, 1,0,0],
        [0,0,0, 0,1,0, 0,1,0]
    ]
    
    laderman_V = [
        [1,0,0, 0,1,0, 0,0,1],
        [1,0,0, 0,1,0, 0,0,-1],
        [1,0,0, 0,-1,0, 0,0,1],
        [1,0,0, 0,-1,0, 0,0,-1],
        [0,1,0, 1,0,0, 0,0,1],
        [0,1,0, 1,0,0, 0,0,-1],
        [0,1,0, -1,0,0, 0,0,1],
        [0,1,0, -1,0,0, 0,0,-1],
        [0,0,1, 0,0,1, 0,0,1],
        [0,0,1, 0,0,1, 0,0,-1],
        [0,0,1, 0,0,-1, 0,0,1],
        [0,0,1, 0,0,-1, 0,0,-1],
        [1,1,0, 0,0,0, 0,0,0],
        [0,0,1, 0,1,0, 0,0,0],
        [0,0,0, 1,0,0, 1,0,0],
        [0,0,0, 0,0,1, 0,1,0],
        [1,0,1, 0,0,0, 0,0,0],
        [0,1,0, 0,0,0, 1,0,0],
        [0,0,0, 0,1,0, 0,1,0]
    ]
    
    # Fill U and V for Part 1
    for k in range(19):
        for idx in range(9):
            U[idx][k] = laderman_U[k][idx]
            V[idx][k] = laderman_V[k][idx]
            
    # Solve for W for Part 1
    # For each output (i, j) in 3x3:
    for i in range(n):
        for j in range(m):
            # Target tensor for C[i,j]
            # C[i,j] = sum_b A[i,b] B[b,j]
            # Tensor indices: (a, b, c) where a is A index, b is B index?
            # U is indexed by A elements. V by B elements.
            # Product term is U[k, a] * V[k, b] * W[k, c]
            # We want sum_k W[k, c] * U[k, a] * V[k, b] = delta(a, i) * delta(c, j) * delta(b, b)?
            # No, the summation is over the "inner" dimension.
            # A is n x m. B is m x p.
            # A[i, b] * B[b, j].
            # So for fixed i, j, the non-zero terms in the tensor are at (a=i, b=b, c=j) for all b in 0..m-1.
            # Wait, the tensor T is 3-way.
            # T[a, b, c] = 1 if a=i, c=j, and b is the summation index?
            # No, the matrix multiplication tensor is defined such that
            # C[i, j] = sum_k T[i, j, k, a, b] ...
            # Let's use the matrix form.
            # We want sum_k W[k, i*n+j] * (sum_a U[k, a] A[a]) * (sum_b V[k, b] B[b]) = C[i,j]
            # This must hold for all A, B.
            # So the tensor M_k = U[k] \otimes V[k] must satisfy
            # sum_k W[k, i*n+j] M_k[a, b] = delta(a, i) * delta(b, j) ?
            # No. C[i,j] depends on A[i, *] and B[*, j].
            # So the target tensor for output (i,j) has 1s at (a=i, b=k, c=j)?
            # Let's check indices.
            # U index corresponds to A element. V index corresponds to B element.
            # A is 3x3. Elements indexed 0..8.
            # B is 3x3. Elements indexed 0..8.
            # C is 3x3. Elements indexed 0..8.
            # C[i, j] = sum_b A[i, b] B[b, j].
            # So for output C[i, j], the contributing terms are A[i, b] and B[b, j] for b=0,1,2.
            # So the target tensor vector (flattened U \otimes V space) has 1s at:
            # (U_index = i*3 + b, V_index = b*3 + j) for b=0,1,2.
            # All other entries 0.
            
            target = np.zeros(27)
            for b in range(m):
                u_idx = i * m + b
                v_idx = b * p + j
                # In the 3x3 part, m=3, p=3.
                # u_idx is index in U (0..8)
                # v_idx is index in V (0..8)
                # The tensor product U \otimes V has size 9x9=81?
                # No, U is (9, 19). V is (9, 19).
                # The "basis" tensors are vectors of length 9*9=81?
                # No, the linear combination is sum_k W[k] (U[k] \cdot A) (V[k] \cdot B).
                # This is a bilinear form.
                # We can represent the "coefficient matrix" for A and B.
                # For each k, we have a matrix M_k (9x9) such that term_k = A^T M_k B?
                # No, term_k = (u_k^T A_vec) (v_k^T B_vec).
                # This is a rank-1 tensor in A_vec \otimes B_vec space.
                # The space is 9x9 = 81 dimensional.
                # We have 19 such vectors.
                # We want to express the target tensor T_ij (81-dim vector) as linear combo of these 19.
                
                # Flatten U[k] \otimes V[k]
                # U[k] is length 9. V[k] is length 9.
                # Tensor product is length 81.
                # Index in 81-vector: u_idx * 9 + v_idx.
                
                target[u_idx * 9 + v_idx] = 1
            
            # Build matrix A_mat (81 x 19)
            A_mat = np.zeros((81, 19))
            for k in range(19):
                u_vec = np.array(laderman_U[k])
                v_vec = np.array(laderman_V[k])
                tensor_vec = np.kron(u_vec, v_vec)
                A_mat[:, k] = tensor_vec
            
            # Solve A_mat w = target
            # Use least squares
            w_vec, residuals, rank, sv = np.linalg.lstsq(A_mat, target, rcond=None)
            
            # Assign to W
            # W index for C[i,j] is j * n + i ?
            # Task: W[k*n+i, r].
            # C is p x n.
            # C[row, col].
            # Row is from B (0..p-1). Col is from A (0..n-1).
            # In our 3x3 part, p=3, n=3.
            # C[i, j] in math notation usually means Row i, Col j.
            # But matrix mult C = A B. A is n x m. B is m x p.
            # C is n x p.
            # C[i, j] = sum A[i, k] B[k, j].
            # Here i is row of C (0..n-1), j is col of C (0..p-1).
            # Task W index: W[k*n+i, r].
            # This implies W is indexed by (col, row)?
            # "W[k*n+i, r]".
            # If k is col index (0..p-1) and i is row index (0..n-1).
            # Then W index = k * n + i.
            # In our loop, i is row of A (0..n-1), j is col of B (0..p-1).
            # So C[i, j] corresponds to Row i, Col j.
            # So W index should be j * n + i.
            
            w_idx = j * n + i
            for k in range(19):
                W[w_idx][k] = w_vec[k]
                
    # Part 2: Naive 3x3x4
    col = 19
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Round W to half-integers
    for i in range(len(W)):
        for j in range(len(W[i])):
            W[i][j] = round(W[i][j] * 2) / 2
            
    return {
        "U": U,
        "V": V,
        "W": W
    }
