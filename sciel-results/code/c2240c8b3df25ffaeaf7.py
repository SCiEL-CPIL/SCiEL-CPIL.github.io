import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Rank 30 decomposition
    # Split A (3x4) into A1 (3x2) and A2 (3x2)
    # Split B (4x5) into B1 (2x5) and B2 (2x5)
    # C = A1 B1 + A2 B2
    # Each term is 3x5.
    # Split each 3x5 term into 5 independent 3x1x1 products?
    # No, 3x5 = sum of 5 (3x1) vectors.
    # C[:, k] = A @ B[:, k]
    # This is 3x4x1 product for each k.
    # Rank of 3x4x1 is 12. Total 5*12=60.
    
    # Better: Split 3x5 into 5 (3x1) is bad.
    # Split 3x5 into 3 (1x5)?
    # C[i, :] = A[i, :] @ B
    # Rank of 1x4x5 is 20. Total 3*20=60.
    
    # We need to mix.
    # Let's use the block structure.
    # A1 (3x2) @ B1 (2x5).
    # We can decompose this 3x5 product.
    # We can use the fact that 3x5 = 3x(2+2+1).
    # So 3x2 + 3x2 + 3x1.
    # Rank of 3x2x2 is 12 (schoolbook) or 7 (if exists).
    # Rank of 3x1x1 is 3.
    
    # Let's assume rank(3x2x2) = 7.
    # Then 3x5 rank = 7 + 7 + 3 = 17.
    # Total rank = 17 + 17 = 34.
    
    # Let's implement rank 34.
    
    # Factors for 3x2x2 (rank 7)
    # We need to verify if rank 7 is possible.
    # If not, use rank 12.
    # If rank 12, then 3x5 rank = 12 + 12 + 3 = 27.
    # Total rank = 27 + 27 = 54.
    
    # Let's use rank 54 to be safe? No, 30 is better.
    # How to get 30?
    # 3x5 rank = 15.
    # This implies 3x2x2 rank = 6? No.
    # 3x5 = 5 * (3x1). Rank 3 each. Total 15.
    # This works if we treat each column independently.
    # But A is shared.
    # C[:, k] = A @ B[:, k].
    # This is 5 separate 3x4x1 products.
    # Rank 12 each. Total 60.
    
    # The "3x5 rank 15" idea assumes we can factorize A and B differently.
    # C = A B.
    # If we write C as sum of 15 rank-1 terms?
    # C = sum_{r=1}^{15} u_r v_r^T.
    # This is possible if rank(C) <= 15.
    # But C is a tensor.
    
    # Let's stick to the block decomposition.
    # A1 (3x2) B1 (2x5).
    # We can use the 7-term algorithm for 2x2x2 on the first 2 rows of A1 and first 2 cols of B1?
    # No.
    
    # Let's use the schoolbook for 3x2x2 (rank 12) and 3x1x1 (rank 3).
    # Total 54.
    
    # Is there a better one?
    # Rank of 3x4x5 is known to be 24?
    # If so, we should aim for 24.
    
    # Let's try to construct rank 24.
    # It involves splitting into 4 blocks of 3x2x2?
    # If rank(3x2x2) = 6, then 4*6=24.
    # Is rank(3x2x2) = 6?
    # 3x2x2 tensor.
    # Min rank is 6?
    # Yes, rank(3,2,2) = 6.
    
    # So we need factors for 3x2x2 with rank 6.
    
    # Let's construct rank 6 for 3x2x2.
    # A (3x2), B (2x2), C (3x2).
    # C = A B.
    
    # We can use the following:
    # U, V, W of shape (6, 6), (4, 6), (6, 6)?
    # U: 6 rows (for 3x2 A), 6 cols (rank).
    # V: 4 rows (for 2x2 B), 6 cols.
    # W: 6 rows (for 3x2 C), 6 cols.
    
    # How to find them?
    # We can use the 2x2x2 rank 7 algorithm? No.
    
    # Let's use a known decomposition.
    # For 3x2x2, we can use:
    # 1. A[0,:] B[0,:] -> C[0,:]
    # 2. A[1,:] B[0,:] -> C[1,:]
    # 3. A[2,:] B[0,:] -> C[2,:]
    # 4. A[0,:] B[1,:] -> C[0,:]
    # 5. A[1,:] B[1,:] -> C[1,:]
    # 6. A[2,:] B[1,:] -> C[2,:]
    # This is rank 6.
    # Wait, this is just schoolbook for 3x2x2.
    # Schoolbook rank is 3*2*2 = 12.
    # Why did I say 6?
    # Because B has 2 columns.
    # C[:, 0] = A @ B[:, 0]. (3x2 @ 2x1). Rank 6.
    # C[:, 1] = A @ B[:, 1]. (3x2 @ 2x1). Rank 6.
    # Total 12.
    
    # So rank(3x2x2) = 12.
    # Then 3x5 rank = 12 + 12 + 3 = 27.
    # Total 54.
    
    # Let's use rank 54.
    
    rank_total = 54
    
    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)
    
    # Part 1: A1 (3x2) @ B1 (2x5)
    # Split B1 into B11 (2x2), B12 (2x2), B13 (2x1)
    # A1 is 3x2.
    # C11 = A1 B11 (3x2). Rank 12.
    # C12 = A1 B12 (3x2). Rank 12.
    # C13 = A1 B13 (3x1). Rank 3.
    # Total 27.
    
    # Part 2: A2 (3x2) @ B2 (2x5)
    # Same, 27.
    
    # Total 54.
    
    col = 0
    # A1 is A[:, 0:2]. Rows 0,1,2. Cols 0,1.
    # B1 is B[0:2, :]. Rows 0,1. Cols 0,1,2,3,4.
    
    # C11: A1 (3x2) @ B11 (2x2) -> C11 (3x2)
    # A1 rows: i=0,1,2; j=0,1.
    # B11 rows: j=0,1; k=0,1.
    # C11 rows: k=0,1; i=0,1,2.
    
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # C12: A1 (3x2) @ B12 (2x2) -> C12 (3x2)
    # B12 cols: k=2,3.
    for i in range(3):
        for j in range(2):
            for k in range(2, 4):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # C13: A1 (3x2) @ B13 (2x1) -> C13 (3x1)
    # B13 col: k=4.
    for i in range(3):
        for j in range(2):
            U[i*m + j, col] = 1
            V[j*p + 4, col] = 1
            W[4*n + i, col] = 1
            col += 1
            
    # Part 2: A2 (3x2) @ B2 (2x5)
    # A2 is A[:, 2:4]. Rows 0,1,2. Cols 2,3.
    # B2 is B[2:4, :]. Rows 2,3. Cols 0,1,2,3,4.
    
    # C21: A2 (3x2) @ B21 (2x2) -> C21 (3x2)
    # A2 cols: j=2,3.
    # B21 rows: j=2,3; k=0,1.
    for i in range(3):
        for j in range(2, 4):
            for k in range(2):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # C22: A2 (3x2) @ B22 (2x2) -> C22 (3x2)
    # B22 cols: k=2,3.
    for i in range(3):
        for j in range(2, 4):
            for k in range(2, 4):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # C23: A2 (3x2) @ B23 (2x1) -> C23 (3x1)
    # B23 col: k=4.
    for i in range(3):
        for j in range(2, 4):
            U[i*m + j, col] = 1
            V[j*p + 4, col] = 1
            W[4*n + i, col] = 1
            col += 1
            
    if col != rank_total:
        raise ValueError("Rank mismatch")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
