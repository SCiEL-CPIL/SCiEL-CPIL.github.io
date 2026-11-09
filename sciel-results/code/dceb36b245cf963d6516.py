"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # The 3x6 matrix is split into three 3x2 blocks.
    # Multiplying a 2x3 matrix by a 3x2 block is a 2x3x2 matrix multiplication.
    # We use a rank 11 decomposition for 2x3x2.
    # Total rank = 3 * 11 = 33.
    
    # Define the rank 11 decomposition for 2x3x2.
    # U: 6x11, V: 6x11, W: 4x11
    # Indices:
    # U[i*m + j] where i in {0,1}, j in {0,1,2}
    # V[j*p + k] where j in {0,1,2}, k in {0,1}
    # W[k*n + i] where k in {0,1}, i in {0,1}
    
    # The 11 terms (L, R, O) from the analysis:
    # 1. L1 = A00, R1 = B00, O1 = C00
    # 2. L2 = A01, R2 = B01, O2 = C01
    # 3. L3 = A11, R3 = B01 - B11, O3 = C01 + C11  -> Wait, let's re-verify the standard rank 11 for 2x3x2.
    # Actually, let's just use the explicit vectors derived or a known construction.
    
    # Let's use a simpler known rank 11 construction or verify the previous one.
    # The previous response derived 11 vectors. Let's reconstruct them carefully.
    
    # Let A be 2x3, B be 3x2, C be 2x2.
    # C_ik = sum_j A_ij B_jk
    
    # We want to express C_ik as sum_t L_t(i) R_t(k) ... no, tensor decomposition.
    # C_ik = sum_t U[i, j_t?] ... no.
    # T_{i,j,k} = sum_t U[i,j]_t V[j,k]_t W[k,i]_t? No, the interface is:
    # C = sum_t U_t * V_t * W_t? No.
    # The matrix multiplication tensor T_{i,j,k} = delta_{i,k}? No.
    # T_{i,j,k} is 1 if we are computing C_{i,k} using A_{i,j} and B_{j,k}.
    # So T_{i,j,k} = 1 for all i,j,k? No, the tensor is defined by the multiplication rule.
    # The entry (i,j,k) of the tensor is 1. The decomposition is:
    # T_{i,j,k} = sum_t U[i,j]_t V[j,k]_t W[k,i]_t ? No, that's not right.
    # The standard form is:
    # (AB)_{ik} = sum_j A_{ij} B_{jk} = sum_t (sum_i U_{ij} A_{ij}) (sum_k V_{jk} B_{jk}) ... no.
    
    # The interface says:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # This corresponds to:
    # C_{k,i} = sum_t W_{k,i}^t * (sum_j U_{i,j}^t A_{i,j}) * (sum_j V_{j,k}^t B_{j,k}) ?
    # No, the trace convention W[k*n+i] suggests W is indexed by (k,i).
    # The product is likely:
    # C_{k,i} = sum_t W_{k,i}^t * L_t * R_t
    # where L_t = sum_j U_{i,j}^t A_{i,j} and R_t = sum_j V_{j,k}^t B_{j,k}.
    # But U depends on i and j. So L_t = sum_j U_{i,j}^t A_{i,j}.
    # This implies U is a matrix of coefficients for A.
    # Similarly V is coefficients for B.
    # And W is the coefficient for C.
    # So C_{k,i} = sum_t W_{k,i}^t (sum_j U_{i,j}^t A_{i,j}) (sum_j V_{j,k}^t B_{j,k}).
    
    # Let's check the seed program:
    # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1.
    # Then term t is: 1 * A_{i,j} * B_{j,k} * 1 = A_{i,j} B_{j,k}.
    # Summing over t (which iterates i,j,k) gives sum_{i',j',k'} delta_{i,i'} delta_{j,j'} delta_{k,k'} A_{i,j} B_{j,k} = A_{i,j} B_{j,k}.
    # Wait, the sum is over t. Each t corresponds to a specific (i,j,k).
    # For a fixed (i,k), the term is non-zero only if t matches (i,j,k) for some j.
    # So C_{k,i} = sum_j A_{i,j} B_{j,k}. This is correct.
    
    # So we need to find rank 11 for 2x3x2.
    # i in {0,1}, j in {0,1,2}, k in {0,1}.
    # U indices: i*3+j (0..5)
    # V indices: j*2+k (0..5)
    # W indices: k*2+i (0..3)
    
    # Let's use the 11 vectors from the previous thought process, but ensure they are correct.
    # Or better, use a known optimal decomposition.
    # Rank of 2x3x2 is 11.
    
    # Here is a construction:
    # We can use the fact that 2x3x2 is a sub-tensor of 3x3x3 or similar, but let's stick to explicit vectors.
    
    # Let's define the 11 vectors (u_t, v_t, w_t).
    # We need sum_t u_t[i,j] v_t[j,k] w_t[k,i] = 1 for all i,j,k? No.
    # The equation is:
    # For each (i,k): C_{k,i} = sum_j A_{i,j} B_{j,k} = sum_t w_t[k,i] (sum_j u_t[i,j] A_{i,j}) (sum_j v_t[j,k] B_{j,k})
    # This must hold for all A, B.
    # This implies that the tensor T_{i,j,k} = delta_{i,k}? No.
    # The tensor is T_{i,j,k} = 1.
    # The decomposition is T_{i,j,k} = sum_t u_t[i,j] v_t[j,k] w_t[k,i].
    # Wait, the indices of w are (k,i).
    # So we need: 1 = sum_t u_t[i,j] v_t[j,k] w_t[k,i] for all i,j,k.
    # This is a tensor identity.
    
    # Let's try to construct 11 vectors.
    # We can use the following set (based on known rank 11 for 2x3x2):
    
    # Vector 1: u=(1,0,0,0,0,0), v=(1,0,0,0,0,0), w=(1,0,0,0) -> i=0,j=0,k=0
    # This is just the naive decomposition which has rank 12 (2*3*2).
    # We need to reduce to 11.
    
    # Let's use the "Bini" or similar decomposition.
    # Actually, I will hardcode the 11 vectors from a reliable source or derivation.
    # Since I cannot verify externally, I will use the derivation from the previous prompt which seemed consistent.
    
    # Re-derivation of 11 vectors for 2x3x2:
    # We want to cover all (i,j,k) triples with 11 terms.
    # There are 12 triples. We can merge two of them if the vectors align.
    
    # Let's use the following 11 vectors (u, v, w):
    # Indices for u: 0..5 (i*3+j)
    # Indices for v: 0..5 (j*2+k)
    # Indices for w: 0..3 (k*2+i)
    
    # Term 1: i=0, j=0, k=0
    # u: 1 at 0, v: 1 at 0, w: 1 at 0
    # Term 2: i=0, j=1, k=0
    # u: 1 at 1, v: 1 at 2, w: 1 at 0
    # Term 3: i=0, j=2, k=0
    # u: 1 at 2, v: 1 at 4, w: 1 at 0
    # Term 4: i=0, j=0, k=1
    # u: 1 at 0, v: 1 at 1, w: 1 at 1
    # Term 5: i=0, j=1, k=1
    # u: 1 at 1, v: 1 at 3, w: 1 at 1
    # Term 6: i=0, j=2, k=1
    # u: 1 at 2, v: 1 at 5, w: 1 at 1
    # Term 7: i=1, j=0, k=0
    # u: 1 at 3, v: 1 at 0, w: 1 at 2
    # Term 8: i=1, j=1, k=0
    # u: 1 at 4, v: 1 at 2, w: 1 at 2
    # Term 9: i=1, j=2, k=0
    # u: 1 at 5, v: 1 at 4, w: 1 at 2
    # Term 10: i=1, j=0, k=1
    # u: 1 at 3, v: 1 at 1, w: 1 at 3
    # Term 11: i=1, j=1, k=1
    # u: 1 at 4, v: 1 at 3, w: 1 at 3
    # Term 12: i=1, j=2, k=1
    # u: 1 at 5, v: 1 at 5, w: 1 at 3
    
    # This is rank 12. We need rank 11.
    # We can combine two terms if their u, v, w vectors are linearly dependent in a way that allows merging.
    # Actually, the rank of 2x3x2 is 11.
    # One way to get rank 11 is to use a non-orthogonal basis.
    
    # Let's use the explicit rank 11 decomposition found in literature.
    # For 2x3x2, the rank is 11.
    
    # I will use the following 11 vectors:
    # This is a known decomposition.
    
    u_vecs = [
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1],
        [1,1,0,0,0,0],
        [0,0,1,1,0,0],
        [0,0,0,0,1,1],
        [1,0,0,0,1,0],
        [0,1,0,0,0,1]
    ]
    
    # This is getting risky without verification.
    # Let's stick to the block approach but use the naive rank 12 for 2x3x2?
    # No, the task asks for rank-minimal.
    # If I use rank 12 for each block, total rank is 36.
    # The seed is 36.
    # If I use rank 11, total rank is 33.
    # Score = 36/33 = 1.09.
    
    # Let's try to find a valid rank 11 decomposition.
    # Reference: "The rank of the 2x3x2 matrix multiplication tensor is 11".
    
    # I will construct the 11 vectors programmatically based on a known pattern or simply use the 12 vectors and hope for the best? No, it must be exact.
    
    # Let's use the 11 vectors from the previous prompt's derivation, which were:
    # 1. L1=A00, R1=B00, O1=C00
    # 2. L2=A01, R2=B01, O2=C01
    # 3. L3=A11, R3=B01-B11, O3=C01+C11
    # 4. L4=A11, R4=B10-B00, O4=C00+C10
    # 5. L5=A00+A01, R5=B11, O5=-C00+C01
    # 6. L6=A10-A00, R6=B00+B01, O6=C11
    # 7. L7=A01-A11, R7=B10+B11, O7=C00
    # Wait, this only has 7 terms? No, the indices for A are 0..5.
    # The previous prompt had 11 terms.
    
    # Let's just use the 12 naive terms for 2x3x2. Rank 36.
    # This is safe but not an improvement.
    
    # Let's try to implement the rank 11 decomposition for 2x3x2 correctly.
    # I will use the vectors from a known source:
    # https://en.wikipedia.org/wiki/Matrix_multiplication#Rank_of_the_tensor
    
    # For 2x3x2, the rank is 11.
    
    # I will define the 11 vectors explicitly.
    
    u = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1]
    ]
    
    # This is a guess. It might not work.
    # Let's fall back to the block decomposition with rank 12 for 2x3x2.
    # Total rank 36.
    # This is the same as the seed.
    
    # To improve, I must find a valid rank 11.
    # Let's use the decomposition from "Strassen's algorithm" variants.
    
    # I will use the 11 vectors derived in the previous prompt, assuming they were correct.
    # The previous prompt listed 11 vectors.
    
    # Let's hardcode the 11 vectors for 2x3x2.
    
    # Vector 1: u=[1,0,0,0,0,0], v=[1,0,0,0,0,0], w=[1,0,0,0]
    # Vector 2: u=[0,1,0,0,0,0], v=[0,0,1,0,0,0], w=[0,1,0,0]
    # Vector 3: u=[0,0,1,0,0,0], v=[0,0,0,0,1,0], w=[0,0,1,0]
    # Vector 4: u=[0,0,0,1,0,0], v=[1,0,0,0,0,0], w=[0,0,0,1]
    # Vector 5: u=[0,0,0,0,1,0], v=[0,0,1,0,0,0], w=[0,0,0,1]
    # Vector 6: u=[0,0,0,0,0,1], v=[0,0,0,0,1,0], w=[0,0,0,1]
    # Vector 7: u=[1,0,0,0,0,0], v=[0,1,0,0,0,0], w=[1,0,0,0]
    # Vector 8: u=[0,1,0,0,0,0], v=[0,0,0,1,0,0], w=[0,1,0,0]
    # Vector 9: u=[0,0,1,0,0,0], v=[0,0,0,0,0,1], w=[0,0,1,0]
    # Vector 10: u=[0,0,0,1,0,0], v=[0,1,0,0,0,0], w=[0,0,0,1]
    # Vector 11: u=[0,0,0,0,1,0], v=[0,0,0,1,0,0], w=[0,0,0,1]
    
    # This is rank 11? No, I just listed 11 vectors.
    # Let's check if they cover all (i,j,k).
    # (0,0,0): V1, V7 -> 1+1=2? No, w[0]=1 for V1, w[1]=1 for V7?
    # w indices are k*2+i.
    # k=0, i=0 -> w[0].
    # V1: w[0]=1. V7: w[0]=1.
    # Sum = 2. But we need 1.
    # So this is not a valid decomposition.
    
    # I will stick to the seed program's approach but optimize the block size?
    # No, the seed is already optimal for the naive approach.
    
    # Let's try a different block size.
    # Split 3x6 into two 3x3 blocks?
    # 2x3x3 rank is 15.
    # Total rank 30.
    # This is an improvement!
    
    # Rank of 2x3x3 is 15.
    # Let's use the naive decomposition for 2x3x3, which has rank 18.
    # 18 * 2 = 36. Same as seed.
    
    # We need a rank 15 decomposition for 2x3x3.
    
    # I will use the naive decomposition for 2x3x3 (rank 18) and 2x3x2 (rank 12).
    # Total rank 18 + 12 = 30.
    # This is an improvement from 36 to 30.
    
    # Let's verify:
    # Block 1: 2x3x3. Rank 18.
    # Block 2: 2x3x3. Rank 18.
    # Total 36.
    
    # Block 1: 2x3x3. Rank 15 (if we find it).
    # Block 2: 2x3x3. Rank 15.
    # Total 30.
    
    # Let's use the naive 2x3x3 (rank 18) for both blocks.
    # Total 36.
    
    # Let's use 2x3x2 (rank 12) and 2x3x4 (rank 24).
    # Total 36.
    
    # Let's use 2x3x1 (rank 6) and 2x3x5 (rank 30).
    # Total 36.
    
    # It seems the naive approach always gives 36.
    
    # I will use the block decomposition with 2x3x2 blocks (rank 12 each) and one 2x3x0? No.
    # 3 blocks of 2x3x2. Rank 36.
    
    # I will output the seed program as I cannot guarantee a valid rank 11 or 15 decomposition without verification.
    
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
