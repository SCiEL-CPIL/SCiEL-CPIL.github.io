"""Program for rank-minimal exact tensor decomposition of 2x2 * 2x8 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (16, rank)
    
    # We treat the multiplication as 8 independent 2x2 matrix-vector products.
    # For each column k of B (0..7), we compute C[:,k] = A @ B[:,k].
    # We use a rank-7 decomposition for each 2x2x2 tensor.
    # The total rank will be 8 * 7 = 56.
    
    # Known rank-7 decomposition for 2x2x2 matrix multiplication.
    # This is a standard result (Strassen-like or exact rank-7).
    # We need U0, V0, W0 of shape (4,7), (2,7), (2,7) such that
    # for each k, the k-th column of the result is computed by:
    # C0_k = sum_r W0[r] * (sum_{ij} U0[i,j,r] A[i,j]) * (sum_j V0[j,r] B[j,k])
    # Wait, the indices for V are (m*p, rank). For a single column k, V is (m, 1) effectively.
    # Let's construct the full matrices.
    
    # We will define the rank-7 decomposition coefficients for a single 2x2x2 block.
    # Let the decomposition be:
    # C_{i,k} = sum_{r=1}^7 w_{i,k,r} * (sum_{j} a_{i,j,r} * A_{i,j}) * (sum_{l} b_{j,l,r} * B_{j,l})
    # Actually, the standard form is:
    # C_{i,k} = sum_r W_{k*n+i, r} * M_r
    # M_r = (sum_{i,j} U_{i*m+j, r} A_{i,j}) * (sum_{j,k} V_{j*p+k, r} B_{j,k})
    
    # For the 2x2x2 case (n=2, m=2, p=2), we need U0 (4x7), V0 (4x7), W0 (4x7).
    # Let's use the known rank-7 decomposition.
    # One such decomposition uses the following pattern (from literature on tensor rank of 2x2x2):
    
    # We can use the decomposition from:
    # "The rank of the 2x2x2 matrix multiplication tensor is 7"
    # A common construction uses the following 7 terms:
    
    U0 = np.array([
        [1, 0, 0, 0, 0, 0, 0],  # A[0,0]
        [0, 1, 0, 0, 0, 0, 0],  # A[0,1]
        [0, 0, 1, 0, 0, 0, 0],  # A[1,0]
        [0, 0, 0, 1, 0, 0, 0],  # A[1,1]
    ])
    
    # This is just the identity, which is rank 4. We need rank 7.
    # Let's use a known rank-7 decomposition.
    # Reference: 
    # The tensor T_{i,j,k} = 1 for 2x2x2 has rank 7.
    # One decomposition is:
    # M1 = A00 * B00 -> C00
    # M2 = A11 * B11 -> C11
    # M3 = A00 * B11 -> C10
    # M4 = A11 * B00 -> C01
    # M5 = A01 * B00 -> C01
    # M6 = A10 * B11 -> C10
    # M7 = (A00+A11) * (B00+B11) - M1 - M2 -> C00+C11+...
    
    # Let's be precise. We want:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Let's use the standard rank-7 decomposition.
    # It is known that the rank is 7. We can find the coefficients by solving a system.
    # However, for the purpose of this code, we can use a known set of coefficients.
    
    # Here is a known rank-7 decomposition for 2x2x2:
    # (U, V, W) for each term r.
    # Term 1: U=[1,0,0,0], V=[1,0,0,0], W=[1,0,0,0] -> A00*B00 -> C00
    # Term 2: U=[0,0,1,0], V=[0,0,0,0], W=[0,0,1,0] -> A10*B01? No.
    
    # Let's use a simpler approach: use the rank-4 schoolbook for each 2x2 block, but that's rank 4*8=32.
    # The task asks for rank-minimal. The known rank for 2x2x2 is 7.
    # So total rank 56.
    
    # Let's define the 7 vectors for U, V, W for a single 2x2x2 block.
    # U has 4 components (indices 0..3 for A[0,0], A[0,1], A[1,0], A[1,1])
    # V has 4 components (indices 0..3 for B[0,0], B[0,1], B[1,0], B[1,1])
    # W has 4 components (indices 0..3 for C[0,0], C[0,1], C[1,0], C[1,1])
    
    # Known rank-7 decomposition (one of many):
    # We can use the following (from "Rank of matrix multiplication tensors"):
    #
    # r=0: U=[1,0,0,0], V=[1,0,0,0], W=[1,0,0,0]  # A00*B00 -> C00
    # r=1: U=[0,0,1,0], V=[0,1,0,0], W=[0,0,1,0]  # A10*B01 -> C10? No, C10 is A10*B00+A11*B10.
    
    # Let's use a different known decomposition.
    # The rank-7 decomposition can be constructed using the following 7 terms:
    # 1. A00 * B00 -> C00
    # 2. A00 * B10 -> C01? No, C01 = A00*B01 + A01*B11.
    
    # Let's look up the exact coefficients for rank-7 2x2x2.
    # One standard decomposition is:
    # 
    # Term 0: U=[1,0,0,0], V=[1,0,0,0], W=[1,0,0,0]
    # Term 1: U=[0,1,0,0], V=[0,0,0,1], W=[0,1,0,0]
    # Term 2: U=[0,0,1,0], V=[0,0,0,0], W=[0,0,1,0] -- This is getting complicated to guess.
    
    # Instead, let's use the fact that we can use the rank-4 schoolbook decomposition for each of the 8 blocks.
    # This gives rank 32.
    # Can we do better?
    # The tensor for 2x2x8 is the direct sum of 8 tensors 2x2x2? No, it's not a direct sum in the simple sense because the B matrix is shared.
    # Actually, the multiplication C = A * B where A is 2x2 and B is 2x8.
    # C[:,k] = A * B[:,k].
    # The tensor is the direct sum of 8 copies of the 2x2x2 tensor?
    # No, the A part is shared.
    # The tensor is: T = sum_{k=0}^7 T_k, where T_k is the tensor for C[:,k] = A * B[:,k].
    # But A is the same for all k.
    # So the tensor is not a direct sum.
    
    # However, we can still use a decomposition that reuses the A coefficients.
    # For each k, we need to compute C[:,k] = A * B[:,k].
    # Let's use a rank-r decomposition for 2x2x2.
    # C[:,k] = sum_{r=0}^{r-1} W[:,r] * (sum_{ij} U[i,j,r] A[i,j]) * (sum_j V[j,r] B[j,k])
    # If we use the same U for all k, and different V for each k, and different W for each k.
    # Then the total rank is 8 * r.
    # If r=7, total rank is 56.
    
    # Let's find a rank-7 decomposition for 2x2x2.
    # We can use the following known rank-7 decomposition (from Strassen's work or later improvements):
    # 
    # The 7 terms are:
    # 1. A00 * B00 -> C00
    # 2. A00 * B01 -> C01
    # 3. A01 * B10 -> C01
    # 4. A01 * B11 -> C01? No.
    
    # Let's use a systematic approach. We'll use the rank-4 schoolbook for now, which is rank 32.
    # The task says "Find a rank-minimal exact tensor decomposition".
    # The known minimum rank for 2x2x2 is 7.
    # So we should aim for 56.
    
    # Let's use the following rank-7 decomposition for 2x2x2:
    # (This is one of the standard ones)
    U0 = np.array([
        [1, 1, 0, 0, 0, 0, 0],  # A00
        [0, 0, 1, 1, 0, 0, 0],  # A01
        [0, 0, 0, 0, 1, 1, 0],  # A10
        [0, 0, 0, 0, 0, 0, 1],  # A11
    ])
    V0 = np.array([
        [1, 0, 1, 0, 1, 0, 0],  # B00
        [0, 1, 0, 1, 0, 1, 0],  # B01
        [0, 0, 1, 0, 1, 0, 1],  # B10
        [0, 0, 0, 1, 0, 1, 1],  # B11
    ])
    W0 = np.array([
        [1, 0, 0, 0, 0, 0, 0],  # C00
        [0, 1, 0, 0, 0, 0, 0],  # C01
        [0, 0, 1, 0, 0, 0, 0],  # C10
        [0, 0, 0, 1, 0, 0, 0],  # C11
    ])
    
    # Let's verify this decomposition.
    # C00 = 1*(A00*B00) = A00*B00. But C00 = A00*B00 + A01*B10. So this is wrong.
    
    # Let's use a verified rank-7 decomposition.
    # From the literature, one rank-7 decomposition is:
    # 
    # M1 = A00 * B00
    # M2 = A00 * B10
    # M3 = A01 * B00
    # M4 = A01 * B10
    # M5 = A10 * B01
    # M6 = A10 * B11
    # M7 = A11 * B01
    # 
    # And the combinations are:
    # C00 = M1 + M3
    # C01 = M2 + M4
    # C10 = M5 + M7
    # C11 = M6
    # 
    # This is not a rank-7 decomposition because the M_i are not linearly independent in the right way.
    
    # Let's use the known result that the rank is 7.
    # We can use the following decomposition (from a paper on tensor rank):
    # 
    # We will use the schoolbook decomposition of rank 4 for each block, giving total rank 32.
    # This is safe and correct.
    # 
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
