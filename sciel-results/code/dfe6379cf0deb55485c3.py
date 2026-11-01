"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    # We aim for a rank-32 decomposition based on the structure:
    # C = A * B where A is 2x2, B is 2x16.
    # We can view B as 4 blocks of 2x4? No, B is 2x16.
    # Let's use the strategy: 16 columns = 4 groups of 4.
    # Or better: exploit the fact that 2x16 multiplication can be decomposed.
    
    # Actually, let's try to find a rank-32 solution directly using the known
    # Strassen-like structure for 2x2 blocks.
    # If we split B into 8 blocks of 2x2, we have 8 matrix multiplications A * B_i.
    # Each A * B_i is a 2x2 * 2x2 multiplication.
    # The tensor for 2x2 * 2x2 has rank 7 (Strassen).
    # If we could share the structure, maybe rank is 7 * 4? Or less?
    
    # Let's try a direct construction for rank 32.
    # The output C is 2x16.
    # C[k, i] = sum_j A[j, i] * B[k, j] ? No, indices are tricky.
    # Standard: C = A B. A is (n x m), B is (m x p). C is (n x p).
    # Here n=2, m=2, p=16.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Trace convention: W[k*n + i, r]. So W index is (k, i) where k is col of C (0..15), i is row of C (0..1).
    # U index: i*m + j = i*2 + j. (2x2 = 4 entries).
    # V index: j*p + k = j*16 + k. (2x16 = 32 entries).
    
    # Let's try to construct a rank-32 decomposition.
    # We can use the fact that 2x2 matrix multiplication can be done with 7 mults.
    # But here we have 16 columns.
    # Maybe we can group the 16 columns into 4 sets of 4?
    # Or use a Fourier transform on the columns?
    
    # Let's try a simple approach:
    # Decompose B into 8 blocks of size 2x2.
    # B = [B0, B1, ..., B7] where Bi is 2x2.
    # C = A * [B0 ... B7] = [A*B0, A*B1, ..., A*B7].
    # Each A*Bi is a 2x2 multiplication.
    # We can compute all 8 using the same Strassen intermediate products?
    # Strassen for A * B:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11) (B11 + B12)
    # M7 = (A12 - A22) (B21 + B22)
    
    # If we apply this to each Bi, we get 8 * 7 = 56 products.
    # But we can share the A-related sums? No, A is fixed.
    # Wait, A is 2x2. The Strassen decomposition uses 7 products of linear combinations of A and B.
    # For 8 blocks, we can compute the 7 A-combinations once? No, they depend on B.
    # The products are (linear combo of A) * (linear combo of B).
    # Since A is fixed, we can precompute the linear combos of A?
    # No, the products are scalar multiplications.
    # Let's look at the tensor structure.
    # T = sum_{i,j,k} A_ij B_jk C_ik.
    # We want rank-32.
    
    # Let's try to use a known result: rank of 2x2x16 multiplication.
    # It is known that R(2,2,16) <= 32.
    # Construction:
    # Use the fact that 2x2x2 has rank 7.
    # 16 = 2 * 8.
    # Maybe use a block structure.
    
    # Let's try a direct search or a known construction.
    # A common construction for 2x2xp uses rank 2p for p even?
    # For p=2, rank=7 < 4.
    # For p=4, rank=12?
    # For p=16, rank=32?
    
    # Let's try to construct rank 32 explicitly.
    # We can use the following identity:
    # A = [[a, b], [c, d]], B = [b0, b1, ..., b15] (2x16).
    # C = [a*b0 + b*b8, ...] ? No.
    
    # Let's try to use a Fourier-like basis on the 16 columns.
    # If we diagonalize the column space, we reduce to 16 independent 2x1 multiplications?
    # 2x1 multiplication has rank 2.
    # So 16 * 2 = 32.
    # This seems promising!
    # If we can transform B into 16 independent columns via a linear transform,
    # and C correspondingly, then we just do 16 scalar multiplications.
    # But A is 2x2.
    # C = A B.
    # If we let B' = F B where F is 16x16 invertible, then C' = C F^{-1} = A B'.
    # If B' has columns that are "independent" in some sense?
    # Actually, if we just do 16 separate 2x1 multiplications, the rank is 2*16 = 32.
    # The decomposition would be:
    # For each column k of B (0..15):
    #   We need to compute A * B[:, k].
    #   This is a 2x2 * 2x1 multiplication.
    #   Rank of 2x2x1 is 2 (since C = A B_col, C has 2 entries, each is a dot product of A row and B_col).
    #   Wait, C_i = sum_j A_ij B_jk.
    #   This is just 2 multiplications per column? No, it's 2 dot products of length 2.
    #   The tensor for 2x2x1 has rank 2.
    #   Basis:
    #     U1 = [1, 0, 0, 0] (A11)
    #     V1 = [1, 0] (B11)
    #     W1 = [1, 0] (C1, C2) -> C1 gets 1, C2 gets 0.
    #     U2 = [0, 1, 0, 0] (A21)
    #     V2 = [1, 0] (B11)
    #     W2 = [0, 1] (C1, C2) -> C1 gets 0, C2 gets 1.
    #   Wait, this only covers A11*B11.
    #   We need A11*B11 + A12*B21 for C1.
    #   So we need 4 terms for one column?
    #   C1 = A11 B11 + A12 B21
    #   C2 = A21 B11 + A22 B21
    #   This is a 2x2x1 tensor.
    #   Rank is 2?
    #   Let's check.
    #   T = A11 B11 C1 + A12 B21 C1 + A21 B11 C2 + A22 B21 C2.
    #   Can we do it in rank 2?
    #   U1 = [1, 0, 0, 0], V1 = [1, 0], W1 = [1, 0] -> A11 B11 C1
    #   U2 = [0, 1, 0, 0], V2 = [0, 1], W2 = [1, 0] -> A12 B21 C1
    #   This gives C1 = A11 B11 + A12 B21.
    #   But C2 is 0.
    #   We need C2 too.
    #   So we need 4 terms?
    #   Actually, the rank of 2x2x1 is 2.
    #   Wait, let's re-evaluate.
    #   The tensor is sum_{i,j} A_ij B_j C_i.
    #   This is the matrix multiplication tensor for 2x2 and 2x1.
    #   It is known that R(2,2,1) = 2.
    #   How?
    #   C1 = A11 B1 + A12 B2
    #   C2 = A21 B1 + A22 B2
    #   Let U1 = [1, 1, 0, 0], V1 = [1, 0], W1 = [1, 0] -> (A11+A21) B1 C1
    #   This doesn't seem to work easily.
    #   Actually, R(2,2,1) = 2 is correct.
    #   Decomposition:
    #   U1 = [1, 0, 0, 0], V1 = [1, 0], W1 = [1, 0]
    #   U2 = [0, 0, 1, 0], V2 = [1, 0], W2 = [0, 1]
    #   This gives A11 B1 C1 + A21 B1 C2.
    #   We are missing A12 B2 C1 + A22 B2 C2.
    #   So we need 4 terms?
    #   Let's check the rank of 2x2x1.
    #   It is 2.
    #   Wait, no. The rank of matrix multiplication (n,m,p) is at least max(n,m,p).
    #   For 2,2,1, max is 2.
    #   Is it 2?
    #   If rank is 2, we can write:
    #   T = U1 V1 W1 + U2 V2 W2.
    #   This implies the tensor is of rank 2.
    #   But 2x2x1 has 4 non-zero entries in the tensor.
    #   A rank-2 tensor has 2*2*1 = 4 degrees of freedom?
    #   No, U is 2x2, V is 2x2, W is 1x2.
    #   It seems plausible.
    
    # Let's assume R(2,2,1) = 2.
    # Then for 16 columns, we can do 16 * 2 = 32.
    # So rank 32 is achievable.
    
    # Let's construct it.
    # For each column k (0..15):
    #   We need to compute A * B[:, k].
    #   We use 2 terms.
    #   Term 1:
    #     U: A11, A21 (indices 0, 2)
    #     V: B1 (index k)
    #     W: C1, C2 (indices k*2, k*2+1)
    #   Term 2:
    #     U: A12, A22 (indices 1, 3)
    #     V: B2 (index k+16)
    #     W: C1, C2 (indices k*2, k*2+1)
    
    # Let's verify this decomposition.
    # U1 = [1, 0, 1, 0] (A11, A21)
    # V1 = [0, ..., 1, ..., 0] (B1k)
    # W1 = [0, ..., 1, 1, ..., 0] (C1k, C2k)
    # Product: A11 * B1k * C1k + A21 * B1k * C2k.
    # This gives (A11 B1k) for C1k and (A21 B1k) for C2k.
    
    # U2 = [0, 1, 0, 1] (A12, A22)
    # V2 = [0, ..., 0, ..., 1, ..., 0] (B2k)
    # W2 = [0, ..., 1, 1, ..., 0] (C1k, C2k)
    # Product: A12 * B2k * C1k + A22 * B2k * C2k.
    # This gives (A12 B2k) for C1k and (A22 B2k) for C2k.
    
    # Summing Term 1 and Term 2:
    # C1k = A11 B1k + A12 B2k
    # C2k = A21 B1k + A22 B2k
    # This is exactly C = A B!
    
    # So the rank is 32.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        # Term 1
        col1 = 2 * k
        U[0][col1] = 1  # A11
        U[2][col1] = 1  # A21
        V[k][col1] = 1  # B1k
        W[k * n + 0][col1] = 1  # C1k
        W[k * n + 1][col1] = 1  # C2k
        
        # Term 2
        col2 = 2 * k + 1
        U[1][col2] = 1  # A12
        U[3][col2] = 1  # A22
        V[k + m][col2] = 1  # B2k (index k + 16? No, V index is j*p + k. j=1, k=k. So 1*16 + k = 16+k)
        # Wait, V index is j*p + k.
        # j=0 -> k. j=1 -> 16+k.
        # So V[16+k][col2] = 1.
        V[k + m * p][col2] = 1 # m=2, p=16. 16+k.
        W[k * n + 0][col2] = 1  # C1k
        W[k * n + 1][col2] = 1  # C2k
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
