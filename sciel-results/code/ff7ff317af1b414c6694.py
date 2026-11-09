"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=3
    # We use a Strassen-like block decomposition.
    # Split A (2x3) into [A1 (2x1) | A2 (2x2)]
    # Split B (3x3) into [B1 (1x3); B2 (2x3)]
    # C = A1*B1 + A2*B2
    # A1*B1 is 2x1 * 1x3 = 2x3. Rank of 2x1x3 tensor is 2.
    # A2*B2 is 2x2 * 2x3 = 2x2x3 tensor.
    # For 2x2x3, we can use Strassen-like or known ranks.
    # R(2,2,3) is known to be 8 (via 2x2x2 rank 7 + 2x2x1 rank 1? No, 2x2x1 is rank 3? No, 2x2x1 is rank 3? 
    # Actually R(2,2,1) = 3. So R(2,2,3) <= 7 + 3 = 10? 
    # But there are better decompositions.
    # Let's just use the schoolbook for the 2x2x3 part to be safe and simple, then optimize if needed.
    # Schoolbook for 2x2x3: 2*2*3 = 12.
    # Total rank: 2 + 12 = 14.
    # Can we do better?
    # R(2,2,2) = 7.
    # R(2,2,1) = 3.
    # 2x2x3 can be seen as two 2x2x1 blocks? No.
    # Let's try to construct a rank 14 decomposition.
    
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (9, rank)
    # W shape: (p*n, rank) = (6, rank)
    
    # Indices for A (2x3): i in 0..1, j in 0..2. Index in U: i*m + j.
    # Indices for B (3x3): j in 0..2, k in 0..2. Index in V: j*p + k.
    # Indices for C (2x3): k in 0..2, i in 0..1. Index in W: k*n + i.
    
    # Part 1: A1 * B1 (2x1 * 1x3)
    # A1 is column j=0 of A. B1 is row j=0 of B.
    # C contribution: C[i,k] += A[i,0] * B[0,k]
    # Rank 2:
    # Term 1: A[0,0] * B[0,k] -> contributes to C[0,k]
    # Term 2: A[1,0] * B[0,k] -> contributes to C[1,k]
    # Wait, the tensor decomposition is sum_r U_i,r V_j,r W_k,r = T_ij,k
    # Here T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # For A1*B1: j=0. T_{i,0,k} = delta_{i,k}? No.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # Actually, the tensor T is defined such that C_{i,k} = sum_{j,r} A_{i,j} B_{j,k} U_{i,j,r} V_{j,k,r} W_{k,i,r}?
    # No, standard definition: T_{i,j,k} is the entry of the tensor.
    # The decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # Where U_r is a vector of size n*m, V_r of size m*p, W_r of size p*n.
    # And T_{i,j,k} = sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i].
    # Wait, the indices in the prompt are:
    # U: (n*m, rank) -> U[row, col] where row = i*m+j
    # V: (m*p, rank) -> V[row, col] where row = j*p+k
    # W: (p*n, rank) -> W[row, col] where row = k*n+i
    # And the condition is: for all i,j,k, T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # And T_{i,j,k} = 1 if the product A_{i,j} B_{j,k} appears in the sum for C_{i,k}.
    # Since C_{i,k} = sum_j A_{i,j} B_{j,k}, T_{i,j,k} = 1 for all valid i,j,k.
    # Wait, is that right?
    # Yes, for matrix multiplication, the tensor T has 1s at all (i,j,k) corresponding to the terms in the sum.
    # So T_{i,j,k} = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # This is the trivial tensor.
    # The goal is to find a low-rank decomposition of this tensor.
    
    # Let's re-read the prompt carefully.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 3 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "exactly decompose the matrix multiplication tensor"
    
    # So yes, T_{i,j,k} = 1 for all i,j,k.
    # We need to decompose the all-ones tensor of shape 2x3x3.
    
    # Let's try the Strassen-like approach again.
    # Split into blocks.
    # A = [A1 A2], B = [B1; B2]
    # C = A1 B1 + A2 B2
    # A1 is 2x1, B1 is 1x3. Tensor T1 is 2x1x3.
    # A2 is 2x2, B2 is 2x3. Tensor T2 is 2x2x3.
    
    # Decompose T1 (2x1x3):
    # T1_{i,0,k} = 1 for i=0,1; k=0,1,2.
    # Rank 2:
    # r=0: U1[0,0]=1, V1[0,0]=1, W1[0,0]=1, W1[1,0]=1, W1[2,0]=1?
    # No, W is indexed by k*n+i.
    # For i=0: W[k*2+0] = W[0], W[2], W[4]
    # For i=1: W[k*2+1] = W[1], W[3], W[5]
    # We need T1_{0,0,k} = 1 and T1_{1,0,k} = 1.
    # Let r=0: U[0,0]=1 (i=0,j=0), V[0,0]=1 (j=0,k=0), W[0,0]=1, W[2,0]=1, W[4,0]=1.
    # Then term 0 contributes 1 to T_{0,0,0}, T_{0,0,1}, T_{0,0,2}.
    # Let r=1: U[1,1]=1 (i=1,j=0), V[0,1]=1, W[1,1]=1, W[3,1]=1, W[5,1]=1.
    # Then term 1 contributes 1 to T_{1,0,0}, T_{1,0,1}, T_{1,0,2}.
    # This works for T1. Rank 2.
    
    # Decompose T2 (2x2x3):
    # i=0,1; j=1,2 (local indices for A2, global j=1,2); k=0,1,2.
    # Global indices: i=0,1; j=1,2; k=0,1,2.
    # U indices: i*m+j = i*3+j.
    # i=0,j=1 -> 1
    # i=0,j=2 -> 2
    # i=1,j=1 -> 4
    # i=1,j=2 -> 5
    # V indices: j*p+k = j*3+k.
    # j=1,k=0 -> 3
    # j=1,k=1 -> 4
    # j=1,k=2 -> 5
    # j=2,k=0 -> 6
    # j=2,k=1 -> 7
    # j=2,k=2 -> 8
    # W indices: k*n+i = k*2+i.
    # k=0,i=0 -> 0
    # k=0,i=1 -> 1
    # k=1,i=0 -> 2
    # k=1,i=1 -> 3
    # k=2,i=0 -> 4
    # k=2,i=1 -> 5
    
    # We need to decompose the 2x2x3 all-ones tensor.
    # Let's use rank 8 for this part.
    # Or maybe we can do better.
    # Let's just use schoolbook for T2: 2*2*3 = 12 terms.
    # Total rank 2 + 12 = 14.
    
    # Can we do better for T2?
    # R(2,2,3) is 8?
    # Let's try to find a rank 8 decomposition for 2x2x3.
    # Or just use the known Strassen for 2x2x2 (rank 7) and add 1 for the extra column?
    # 2x2x3 = 2x2x2 + 2x2x1.
    # R(2,2,2) = 7.
    # R(2,2,1) = 3.
    # Total 10.
    # So 2 + 10 = 12.
    
    # Let's try to implement rank 12.
    # T2 part:
    # Split B2 (2x3) into B2a (2x2) and B2b (2x1).
    # C2 = A2 B2a + A2 B2b.
    # A2 B2a is 2x2x2. Rank 7.
    # A2 B2b is 2x2x1. Rank 3.
    # Total 10.
    
    # Strassen for 2x2x2:
    # A2 = [[a, b], [c, d]]
    # B2a = [[e, f], [g, h]]
    # C2a = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # M1 = (a+d)(e+h)
    # M2 = (a+d)(g+f)
    # M3 = (c+d)g
    # M4 = (a-b)e
    # M5 = (b+d)h
    # M6 = (c-a)(f+e)
    # M7 = (d-b)(h+g)
    # C2a[0,0] = M1 + M4 - M5 - M7
    # C2a[0,1] = M5 + M7
    # C2a[1,0] = M3 + M6
    # C2a[1,1] = M1 - M3 + M4 + M6
    
    # Let's map this to the tensor decomposition.
    # We need to express the all-ones tensor for 2x2x2 as a sum of 7 rank-1 tensors.
    # This is a known result. I will hardcode the coefficients.
    
    # For 2x2x1 (A2 B2b):
    # A2 = [[a, b], [c, d]]
    # B2b = [i, j]^T
    # C2b = [ai+bj, ci+dj]
    # Rank 3:
    # M8 = (a+b)(i+j)
    # M9 = a i
    # M10 = b j
    # C2b[0] = M8 - M9 - M10 + a j + b i? No.
    # Actually, R(2,2,1) = 3.
    # M8 = a i
    # M9 = b j
    # M10 = (c+d)(i+j) - c i - d j?
    # Let's use the standard rank 3 decomposition for 2x2x1.
    # C0 = a i + b j
    # C1 = c i + d j
    # M1 = a i
    # M2 = b j
    # M3 = (a+b+c+d)(i+j) - M1 - M2 - c i - d j?
    # This is getting complicated.
    
    # Let's stick to rank 14 (2 + 12) for safety.
    # Or even rank 13 if I can find a simple improvement.
    # Actually, let's just output rank 14.
    
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A1 B1 (rank 2)
    # r=0: i=0, j=0, k=0..2
    U[0*3+0][0] = 1
    V[0*3+0][0] = 1
    for k in range(3):
        W[k*2+0][0] = 1
    # r=1: i=1, j=0, k=0..2
    U[1*3+0][1] = 1
    V[0*3+0][1] = 1
    for k in range(3):
        W[k*2+1][1] = 1
        
    # Part 2: A2 B2 (rank 12, schoolbook)
    # i=0,1; j=1,2; k=0,1,2
    r = 2
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*3 + (j+1)][r] = 1
                V[(j+1)*3 + k][r] = 1
                W[k*2 + i][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
