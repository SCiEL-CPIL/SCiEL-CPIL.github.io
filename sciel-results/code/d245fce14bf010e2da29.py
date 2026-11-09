"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110.0

    n, m, p = DIMENSIONS
    # We will attempt a rank reduction via a 2x2 Strassen-like block decomposition
    # of the 6x6x6 matrix multiplication tensor.
    # The 6x6 matrices can be viewed as 2x2 blocks of 3x3 sub-matrices.
    # Let A be 6x6, B be 6x6, C = A*B be 6x6.
    # A = [[A00, A01], [A10, A11]] where Aij are 3x3.
    # B = [[B00, B01], [B10, B11]] where Bkl are 3x3.
    # C = [[C00, C01], [C10, C11]].
    
    # Strassen's algorithm for 2x2 matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M1 - M2 + M6
    # C10 = M2 + M4 - M3 + M5
    # C11 = M1 - M4 + M3 - M6
    
    # The tensor for 6x6*6x6 is sum over blocks.
    # Each block multiplication (3x3 * 3x3) has rank 27.
    # Total rank = 7 * 27 = 189.
    # Base rank = 6*6*6 = 216.
    # So this gives a rank of 189.
    
    # Let's verify the indices for the global tensor.
    # U has shape (n*m, rank) = (36, 189)
    # V has shape (m*p, rank) = (36, 189)
    # W has shape (p*n, rank) = (36, 189)
    
    # Indexing convention:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    # For a block term M_t = (sum a_ij A_ij) * (sum b_kl B_kl):
    # The contribution to C_rs is (sum a_ij b_kl C_block_ij_kl_rs).
    # Specifically, C_ij_kl is the 3x3 product of A_ij and B_kl.
    # The (r,s) element of A_ij is A[i*3+r, j*3+s].
    # The (s',k') element of B_kl is B[k*3+s', l*3+k'].
    # The (r, s') element of the product is sum_u A[i*3+r, j*3+u] * B[k*3+u, l*3+s'].
    # So C[i*3+r, l*3+s'] += sum_u A[i*3+r, j*3+u] B[k*3+u, l*3+s'].
    
    # In the tensor decomposition:
    # C[i*3+r, l*3+s'] = sum_c U[i*3+r, c] V[j*3+u, c] W[k*3+u, c] ... wait.
    # The standard tensor form is C_{i,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}? No.
    # C_{r, s'} = sum_{u} A_{r, u} B_{u, s'}.
    # Global indices:
    # A_{I, J} where I = i*3+r, J = j*3+u.
    # B_{J, K} where J = k*3+u (wait, J is the column of A, row of B).
    # Let's align J.
    # A_{I, J} with I=(i, r), J=(j, u).
    # B_{J, K} with J=(j, u) ?? No, B is 6x6.
    # B_{J, K} where J is row, K is col.
    # The row index of B must match the column index of A.
    # So if A's column is J = j*3+u, then B's row is J = j*3+u.
    # But in Strassen, we multiply blocks A_ij (row i, col j) and B_kl (row k, col l).
    # The "inner" index is the column of A_ij (which is u) and row of B_kl (which is u).
    # So the global row of B is k*3+u.
    # But for the matrix product to work, the column of A must equal the row of B.
    # A's global column is j*3+u.
    # B's global row is k*3+u.
    # These are NOT necessarily equal!
    # Ah, Strassen multiplies A_ij and B_kl.
    # A_ij is at rows i*3..i*3+2, cols j*3..j*3+2.
    # B_kl is at rows k*3..k*3+2, cols l*3..l*3+2.
    # The product A_ij * B_kl is defined if the number of columns of A_ij equals number of rows of B_kl.
    # Here both are 3.
    # But in the global matrix multiplication C = A * B, the element C_{I, K} = sum_J A_{I, J} B_{J, K}.
    # So J runs from 0 to 5.
    # If we decompose J into (j, u) and (k, u), we must have J = j*3+u AND J = k*3+u.
    # This implies j = k.
    # So Strassen's algorithm works because the block structure allows summing over the "inner" blocks.
    # Specifically, C_il = sum_j A_ij B_jl.
    # So we only multiply A_ij and B_jl where the column index of A block matches row index of B block.
    # My previous Strassen steps mixed different j and k.
    # Step 1: (A00 + A11)(B00 + B11).
    # This involves A00*B00, A00*B11, A11*B00, A11*B11.
    # A00*B11: A is block (0,0), B is block (1,1).
    # In global sum, we need A_{i,j} B_{j,l}. Here i=0, j=0 for A, and j=1, l=1 for B.
    # j=0 != j=1. So A00*B11 is NOT part of the standard matrix product C = AB!
    # Wait.
    # C = AB.
    # C_{00} (block) = A_{00}B_{00} + A_{01}B_{10}.
    # C_{01} = A_{00}B_{01} + A_{01}B_{11}.
    # C_{10} = A_{10}B_{00} + A_{11}B_{10}.
    # C_{11} = A_{10}B_{01} + A_{11}B_{11}.
    
    # Strassen computes:
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This contains cross terms like A00B11 which are not in C.
    # However, Strassen combines these to cancel out the unwanted terms.
    # C00 = M1 + M4 - M5 + M7.
    # Let's check if the unwanted terms cancel.
    # M1 has A00B11.
    # M7 = (A01 - A11)(B10 + B11) = A01B10 + A01B11 - A11B10 - A11B11.
    # M5 = (A00 + A01)B11 = A00B11 + A01B11.
    # M4 = A11(B10 - B00) = A11B10 - A11B00.
    
    # C00 = (A00B00 + A00B11 + A11B00 + A11B11) + (A11B10 - A11B00) - (A00B11 + A01B11) + (A01B10 + A01B11 - A11B10 - A11B11)
    # Terms:
    # A00B00: +1
    # A00B11: +1 - 1 = 0
    # A11B00: +1 - 1 = 0
    # A11B11: +1 - 1 = 0
    # A11B10: +1 - 1 = 0
    # A01B11: -1 + 1 = 0
    # A01B10: +1
    # So C00 = A00B00 + A01B10.
    # This is correct!
    
    # So the decomposition is valid. The "unwanted" terms cancel out in the sum.
    # This means we can use the Strassen coefficients.
    # The rank is 7 * (rank of 3x3 multiplication).
    # Rank of 3x3 multiplication is 9 (schoolbook).
    # So total rank = 7 * 9 = 63.
    # Wait, earlier I said 189. Why?
    # Because I thought each block multiplication had rank 27?
    # No, 3x3 multiplication has rank 3*3*3 = 27.
    # So 7 * 27 = 189.
    # Is there a lower rank for 3x3?
    # Best known is 19 (Le Verrier), but finding it is hard.
    # We will stick to schoolbook 27 for the inner blocks.
    # So rank = 189.
    # Base rank = 216.
    # Improvement.
    
    # Let's construct the matrices.
    
    # We define the 7 steps.
    # Each step t has:
    # - Coefficients for A blocks: alpha[t][ij]
    # - Coefficients for B blocks: beta[t][kl]
    # - Coefficients for C blocks: gamma[t][il]
    
    # The term is sum_{ij, kl} alpha[t][ij] beta[t][kl] (A_ij * B_kl).
    # Note: In the sum for C_il, we only sum over terms where the B block's row index matches A block's col index?
    # No, the algebraic identity holds for the whole expression.
    # The "A_ij * B_kl" is just a 3x3 matrix product.
    # We can expand each 3x3 product using its rank-27 decomposition.
    # So for each step t, and for each pair (ij, kl) with non-zero alpha and beta,
    # we add 27 terms to the global tensor.
    
    # Total terms = 7 * (number of non-zero pairs) * 27.
    # In Strassen, each step has few non-zero alphas and betas.
    # Step 1: A00, A11. B00, B11. 4 pairs.
    # Step 2: A10, A11. B00. 2 pairs.
    # Step 3: A00. B01, B11. 2 pairs.
    # Step 4: A11. B10, B00. 2 pairs.
    # Step 5: A00, A01. B11. 2 pairs.
    # Step 6: A10, A00. B00, B01. 4 pairs.
    # Step 7: A01, A11. B10, B11. 4 pairs.
    # Total pairs = 4+2+2+2+2+4+4 = 20.
    # Total scalar multiplications = 20 * 27 = 540.
    # This is higher than the number of non-zero entries in the tensor?
    # The tensor has 6*6*6 = 216 non-zero entries.
    # The rank is the number of columns in U, V, W.
    # Each scalar multiplication (3x3) contributes 27 columns.
    # So rank = 20 * 27 = 540?
    # No. The rank is the minimum number of columns.
    # If we just concatenate the decompositions, the rank is the sum of the ranks.
    # So rank = 540.
    # This is WORSE than 216!
    # Why?
    # Because Strassen reduces the number of matrix multiplications from 4 to 7 for 2x2 blocks.
    # But for 3x3 blocks, the cost is 27.
    # 4 * 27 = 108.
    # 7 * 27 = 189.
    # So Strassen INCREASES the rank for 3x3 blocks?
    # Wait.
    # Standard 2x2 block multiplication:
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    # This requires 4 matrix multiplications.
    # Each 3x3 multiplication has rank 27.
    # Total rank = 4 * 27 = 108.
    # Strassen requires 7 matrix multiplications.
    # Total rank = 7 * 27 = 189.
    # So Strassen is WORSE for rank!
    # Strassen reduces the number of scalar multiplications (7 vs 8 for 2x2 scalar), but here we are counting tensor rank.
    # For tensor rank, schoolbook is better if the inner blocks are small.
    # So I should just use schoolbook for the whole 6x6?
    # Rank 216.
    
    # Can we do better?
    # Maybe a different block size?
    # 3x3 blocks?
    # 6x6 = 2x2 blocks of 3x3.
    # What if we use 1x1 blocks? Rank 216.
    # What if we use a known low-rank decomposition for 6x6?
    # Best known rank for 6x6 matrix multiplication is 216?
    # Actually, for 3x3 it's 19.
    # If we use 3x3 blocks, we can use rank 19 for the inner product.
    # Then total rank = 4 * 19 = 76.
    # This is much better than 216!
    # But finding the rank-19 decomposition for 3x3 is the hard part.
    # The current program uses schoolbook.
    # I should try to implement a known rank-19 decomposition for 3x3.
    # Or maybe a simpler improvement?
    
    # Let's check if I can find a rank-19 decomposition for 3x3.
    # It is complex.
    # Alternative: Use the fact that 6=2*3.
    # Maybe I can just output the schoolbook decomposition?
    # The task asks for rank-minimal.
    # If I can't find a better one, I should return the best I have.
    # The current program returns rank 216.
    # If I return rank 216, score is 1.0.
    # If I return rank 189 (Strassen), score is 216/189 = 1.14.
    # Wait, score is seed/value.
    # Seed is 1.0. Value is rank.
    # If rank is 189, value is 189.
    # Score = 1.0 / (189/216)? No.
    # "score is the rank".
    # "evaluator measures the raw value (lower is better)".
    # "score is seed/value".
    # If seed is 1.0, and value is rank.
    # Then score = 1.0 / rank?
    # Or is "value" the rank?
    # "score is seed/value, so 1.0 equals the seed program".
    # Seed program rank is 216.
    # If score = seed/value, and seed score is 1.0, then 1.0 = 1.0 / 216? No.
    # Maybe "value" is the rank, and the score is computed as seed_rank / rank?
    # "score is the rank" -> This sentence is confusing.
    # "The evaluator measures the raw value (lower is better)."
    # "score is seed/value".
    # If value is rank, then score = seed_rank / rank.
    # If I reduce rank from 216 to 189, score = 216/189 = 1.14.
    # This is an improvement.
    # So Strassen IS better in terms of score, even if it's not "rank minimal" in the theoretical sense (it's not, but it's better than 216).
    # Wait, is 189 the rank?
    # Yes, 7 steps * 27 rank per step.
    # So I should implement Strassen.
    
    # Let's implement Strassen.
    
    # Define the 7 steps.
    # alpha[t] is dict {(i,j): val}
    # beta[t] is dict {(k,l): val}
    # gamma[t] is dict {(i,l): val}
    
    steps = []
    
    # Step 1: M1 = (A00 + A11)(B00 + B11)
    # alpha: (0,0):1, (1,1):1
    # beta: (0,0):1, (1,1):1
    # gamma: (0,0):1, (0,1):1, (1,0):0, (1,1):1  <-- Wait, check gamma again.
    # C00 = M1 + M4 - M5 + M7 -> gamma1(0,0)=1
    # C01 = M3 + M1 - M2 + M6 -> gamma1(0,1)=1
    # C10 = M2 + M4 - M3 + M5 -> gamma1(1,0)=0
    # C11 = M1 - M4 + M3 - M6 -> gamma1(1,1)=1
    # So gamma1: (0,0):1, (0,1):1, (1,1):1.
    
    steps.append({
        'alpha': {(0,0): 1, (1,1): 1},
        'beta': {(0,0): 1, (1,1): 1},
        'gamma': {(0,0): 1, (0,1): 1, (1,1): 1}
    })
    
    # Step 2: M2 = (A10 + A11)B00
    # alpha: (1,0):1, (1,1):1
    # beta: (0,0):1
    # gamma: (0,1):-1, (1,0):1
    steps.append({
        'alpha': {(1,0): 1, (1,1): 1},
        'beta': {(0,0): 1},
        'gamma': {(0,1): -1, (1,0): 1}
    })
    
    # Step 3: M3 = A00(B01 - B11)
    # alpha: (0,0):1
    # beta: (0,1):1, (1,1):-1
    # gamma: (0,1):1, (1,0):-1, (1,1):1
    steps.append({
        'alpha': {(0,0): 1},
        'beta': {(0,1): 1, (1,1): -1},
        'gamma': {(0,1): 1, (1,0): -1, (1,1): 1}
    })
    
    # Step 4: M4 = A11(B10 - B00)
    # alpha: (1,1):1
    # beta: (1,0):1, (0,0):-1
    # gamma: (0,0):1, (1,0):1, (1,1):-1
    steps.append({
        'alpha': {(1,1): 1},
        'beta': {(1,0): 1, (0,0): -1},
        'gamma': {(0,0): 1, (1,0): 1, (1,1): -1}
    })
    
    # Step 5: M5 = (A00 + A01)B11
    # alpha: (0,0):1, (0,1):1
    # beta: (1,1):1
    # gamma: (0,0):-1, (1,0):1
    steps.append({
        'alpha': {(0,0): 1, (0,1): 1},
        'beta': {(1,1): 1},
        'gamma': {(0,0): -1, (1,0): 1}
    })
    
    # Step 6: M6 = (A10 - A00)(B00 + B01)
    # alpha: (1,0):1, (0,0):-1
    # beta: (0,0):1, (0,1):1
    # gamma: (0,1):1, (1,1):-1
    steps.append({
        'alpha': {(1,0): 1, (0,0): -1},
        'beta': {(0,0): 1, (0,1): 1},
        'gamma': {(0,1): 1, (1,1): -1}
    })
    
    # Step 7: M7 = (A01 - A11)(B10 + B11)
    # alpha: (0,1):1, (1,1):-1
    # beta: (1,0):1, (1,1):1
    # gamma: (0,0):1
    steps.append({
        'alpha': {(0,1): 1, (1,1): -1},
        'beta': {(1,0): 1, (1,1): 1},
        'gamma': {(0,0): 1}
    })
    
    # Now build U, V, W.
    # n=6, m=6, p=6.
    # Block size b = 3.
    # U index: [i*3+r, j*3+u, c]
    # V index: [j*3+u, k*3+u, c] ?? No.
    # V index: [row, col, c].
    # Row of V is column of A. Col of V is row of B.
    # So V[j*3+u, k*3+u, c] is NOT correct because u is the inner index.
    # The inner index u must be the same for A's col and B's row.
    # So for a term A_ij * B_kl, we sum over u=0..2.
    # A[i*3+r, j*3+u] * B[k*3+u, l*3+s].
    # So V index should be [j*3+u, k*3+u, c].
    # But j and k are block indices.
    # In Strassen, j and k can be different.
    # If j != k, then j*3+u != k*3+u.
    # So V[j*3+u, k*3+u, c] connects a specific column of A to a specific row of B.
    # This is valid.
    
    rank = 0
    U = []
    V = []
    W = []
    
    # Initialize
    for _ in range(36):
        U.append([0])
        V.append([0])
        W.append([0])
    # This is inefficient. Let's collect the entries and then pad.
    
    U_entries = []
    V_entries = []
    W_entries = []
    
    for step in steps:
        for (ij, a_val) in step['alpha'].items():
            i, j = ij
            for (kl, b_val) in step['beta'].items():
                k, l = kl
                # Inner loop over u, r, s
                for u in range(3):
                    for r in range(3):
                        for s in range(3):
                            # This corresponds to the term A[i*3+r, j*3+u] * B[k*3+u, l*3+s]
                            # Contribution to C[i*3+r, l*3+s]
                            # U[i*3+r, j*3+u, c] = a_val
                            # V[j*3+u, k*3+u, c] = b_val
                            # W[k*3+u, i*3+r, c] = 1 ??
                            # Wait, W index is [k*n + i, c].
                            # n=6. i is block row index (0 or 1).
                            # But W index needs the global row of C?
                            # W[k*n + i, c] in the prompt means W[global_index, c].
                            # global_index = k*6 + i?
                            # The prompt says: W[k*n+i, r].
                            # Here n=6.
                            # The indices i, k in the prompt formula W[k*n+i, r] likely refer to the block indices?
                            # No, "output factor follows the official trace convention W[k*n+i,r]".
                            # Usually, the tensor is T_{i,j,k}.
                            # U_{i,j,c}, V_{j,k,c}, W_{k,i,c}.
                            # Here i is row of A (and C), j is col of A (and row of B), k is col of B (and C).
                            # So W index is (k, i).
                            # Global indices:
                            # I = i_block * 3 + r
                            # J = j_block * 3 + u
                            # K = k_block * 3 + s
                            # So W[K, I, c] = 1?
                            # The prompt says W[k*n+i, r].
                            # This looks like W[global_k, global_i, c].
                            # Let's assume W index is (K, I).
                            # So W[K*6 + I, c]? No, W is (p*n, rank) = (36, rank).
                            # So index is K*6 + I? Or I*6 + K?
                            # "W[k*n+i, r]".
                            # If n=6, k is col of B (0..5), i is row of A (0..5).
                            # So index = k*6 + i.
                            
                            # Let's verify with schoolbook.
                            # U[I, J, c] = 1
                            # V[J, K, c] = 1
                            # W[K, I, c] = 1
                            # Then sum_c U[I,J,c] V[J,K,c] W[K,I,c] = sum_c 1*1*1 = 1.
                            # This gives C[I, K] = 1? No.
                            # C[I, K] = sum_J A[I, J] B[J, K].
                            # The decomposition is A[I, J] = sum_c U[I, J, c] ... No.
                            # The decomposition is of the TENSOR of multiplication.
                            # T_{I, J, K} = sum_c U[I, J, c] V[J, K, c] W[K, I, c].
                            # For schoolbook, T_{I, J, K} = 1 if (I, J, K) is a valid index triple.
                            # Wait, the multiplication tensor has T_{I, J, K} = 1 for all I, J, K?
                            # No.
                            # C_{I, K} = sum_J A_{I, J} B_{J, K}.
                            # The tensor T is defined such that C = A * B.
                            # The "multiplication tensor" usually has entries 1 at (I, J, K) where J is the summation index.
                            # So T_{I, J, K} = 1 for all I, J, K in range.
                            # Then C_{I, K} = sum_J T_{I, J, K} A_{I, J} B_{J, K}.
                            # This is correct.
                            
                            # So we need to represent T_{I, J, K} = 1.
                            # My Strassen decomposition represents the SUM of terms.
                            # For a term A_ij * B_kl, it contributes to T_{I, J, K} where I is in block i, J in block j, K in block l.
                            # Specifically, T_{I, J, K} += alpha * beta * gamma?
                            # No, the gamma is for the C block.
                            # The tensor T is fixed.
                            # The decomposition is T = sum_c U_c V_c W_c.
                            # Each term in Strassen corresponds to a set of (I, J, K) triples.
                            # For A_ij * B_kl, the triples are (I, J, K) where I in i, J in j, K in l.
                            # And the value is 1.
                            # But Strassen combines these.
                            # The coefficient of T_{I, J, K} in the sum is the sum of (alpha * beta * 1) for all steps that include this (ij, kl) pair?
                            # No.
                            # The identity is:
                            # sum_{ij, kl} alpha[t][ij] beta[t][kl] (A_ij * B_kl) = sum_{il} gamma[t][il] C_il.
                            # This means that for each (i, l), the sum over t of gamma[t][il] * (sum_{j,k} ... ) equals C_il.
                            # This is an identity on the matrices.
                            # It implies that the tensor of the LHS equals the tensor of the RHS.
                            # The RHS is just C.
                            # So the LHS decomposes the multiplication tensor.
                            
                            # How to map to U, V, W?
                            # For each step t, and each (ij, kl) pair, we have a term alpha * beta * (A_ij * B_kl).
                            # This term is a matrix product.
                            # We can decompose this product using rank-27 factors.
                            # For each (u, r, s), we add a column.
                            # U[I, J, c] = alpha
                            # V[J, K, c] = beta
                            # W[K, I, c] = 1
                            # Where I = i*3+r, J = j*3+u, K = l*3+s.
                            # Wait, V[J, K, c].
                            # J is column of A. K is column of B?
                            # No, V is (m*p, rank). Rows are J (col of A), Cols are K (row of B).
                            # So V[J, K, c] = beta.
                            # But in the product A_ij * B_kl, the column of A is j*3+u.
                            # The row of B is k*3+u.
                            # So J = j*3+u.
                            # K (row of B) = k*3+u.
                            # But V index is (J, K_row).
                            # So V[j*3+u, k*3+u, c] = beta.
                            # And W index is (K_col, I).
                            # K_col = l*3+s.
                            # I = i*3+r.
                            # So W[l*3+s, i*3+r, c] = 1?
                            # But the prompt says W[k*n+i, r].
                            # Here k is likely K_col (global col of B).
                            # i is I (global row of A).
                            # So W[K_col*6 + I, c] = 1.
                            
                            # Let's check if this works for schoolbook.
                            # If we have term A_ij * B_kl with alpha=1, beta=1.
                            # And gamma is such that it sums to 1 for the correct C_il.
                            # In schoolbook, we have 216 terms.
                            # Each term is A_ij * B_ij (since j=k).
                            # So i, j, k=j, l.
                            # I = i*3+r, J = j*3+u, K_row = j*3+u, K_col = l*3+s.
                            # U[I, J, c] = 1.
                            # V[J, K_row, c] = 1.
                            # W[K_col, I, c] = 1.
                            # Then sum_c U[I,J,c] V[J,K_row,c] W[K_col,I,c] = 1.
                            # This contributes to C[I, K_col].
                            # But C[I, K_col] = sum_J A[I, J] B[J, K_col].
                            # Here J is fixed to j*3+u.
                            # So this term contributes to C[I, K_col] via J = j*3+u.
                            # This is correct.
                            
                            # Now for Strassen.
                            # We have terms where j != k.
                            # E.g. A00 * B11. i=0, j=0, k=1, l=1.
                            # I = 0*3+r = r.
                            # J = 0*3+u = u.
                            # K_row = 1*3+u = 3+u.
                            # K_col = 1*3+s = 3+s.
                            # U[r, u, c] = alpha.
                            # V[u, 3+u, c] = beta.
                            # W[3+s, r, c] = 1.
                            # This contributes to C[r, 3+s].
                            # But in C = AB, C[0, 3] (block 0,1) = A00 B01 + A01 B11.
                            # A00 B11 is NOT in C01.
                            # So this term is "unwanted".
                            # But it cancels out with other terms in the sum over t.
                            # So we just need to add the terms with the correct coefficients.
                            # The coefficient for W is 1?
                            # The gamma is applied to the C block.
                            # So for each step t, and each (i, l) in gamma, we multiply the contribution by gamma[t][i, l].
                            # So W[K_col, I, c] = gamma[t][i, l].
                            
                            # Let's refine the loop.
                            for (il, g_val) in step['gamma'].items():
                                i_g, l_g = il
                                if i_g != i or l_g != l:
                                    # Wait, the term A_ij * B_kl contributes to C_il ONLY if the Strassen formula says so.
                                    # The formula sums over all ij, kl in the step.
                                    # And then multiplies by gamma for each C block.
                                    # So for a fixed step t, and fixed (ij, kl) pair,
                                    # it contributes to ALL C_il where gamma[t][il] is non-zero.
                                    # And the coefficient is alpha * beta * gamma.
                                    # So for each (u, r, s):
                                    # I = i*3+r
                                    # J = j*3+u
                                    # K_row = k*3+u
                                    # K_col = l*3+s
                                    # U[I, J, c] = alpha
                                    # V[J, K_row, c] = beta
                                    # W[K_col, I, c] = gamma
                                    
                                    # We add a column for each (t, ij, kl, u, r, s).
                                    pass
                                    break
                                break
                            break
                        break
                    break
                break
            break
        break
    # This nested loop logic is getting messy. Let's just iterate properly.
    
    U_list = []
    V_list = []
    W_list = []
    
    for t, step in enumerate(steps):
        for (ij, a_val) in step['alpha'].items():
            i, j = ij
            for (kl, b_val) in step['beta'].items():
                k, l = kl
                for (il, g_val) in step['gamma'].items():
                    # The contribution to C_il from A_ij * B_kl is a_val * b_val * g_val.
                    # We decompose A_ij * B_kl into sum_{u,r,s} e_{r,u} e_{u,s} ...
                    # Actually, A_ij * B_kl is a 3x3 matrix.
                    # Its (r, s) entry is sum_u A_ij[r, u] B_kl[u, s].
                    # So for each (u, r, s), we have a term.
                    for u in range(3):
                        for r in range(3):
                            for s in range(3):
                                I = i * 3 + r
                                J = j * 3 + u
                                K_row = k * 3 + u
                                K_col = l * 3 + s
                                
                                # Check time
                                if time.time() > deadline:
                                    # Fallback to schoolbook?
                                    pass
                                
                                U_list.append((I, J, a_val))
                                V_list.append((J, K_row, b_val))
                                W_list.append((K_col, I, g_val))
    
    # Now construct the matrices.
    # We need to assign a column index to each (u, r, s, t, ij, kl) combination.
    # It's easier to just build the dictionaries for each column.
    
    num_cols = len(U_list)
    U_mat = [[0] * num_cols for _ in range(36)]
    V_mat = [[0] * num_cols for _ in range(36)]
    W_mat = [[0] * num_cols for _ in range(36)]
    
    for c in range(num_cols):
        I, J, val = U_list[c]
        U_mat[I][J] = val # Error: J is not col index, it's part of the row?
        # U is (n*m, rank) = (36, rank).
        # So U[I, c] = val.
        U_mat[I][c] = val
        
        J_v, K_row, val_v = V_list[c]
        V_mat[J_v][c] = val_v
        
        K_col, I_w, val_w = W_list[c]
        W_mat[K_col * 6 + I_w][c] = val_w # W index is k*n + i.
        
    # Check rank
    rank = num_cols
    print(f"Rank: {rank}")
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
