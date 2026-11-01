"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Block-Kronecker structure:
    # Split columns into 5 groups of 2.
    # For each group of 2 columns, we multiply A (2x3) by B (3x2).
    # A[2,3] * B[3,2] -> C[2,2]
    #
    # For the 2x3 x 3x2 sub-problem, we can use a rank-6 decomposition:
    # C[0,0] = a0*b00 + a1*b10 + a2*b20
    # C[0,1] = a0*b01 + a1*b11 + a2*b21
    # C[1,0] = a0*b00 + a1*b10 + a2*b20  -- wait, no.
    # Actually for 2x3 * 3x2:
    # C[0,0] = a0*b00 + a1*b10 + a2*b20
    # C[0,1] = a0*b01 + a1*b11 + a2*b21
    # C[1,0] = a0*b00 + a1*b10 + a2*b20 -- no, C[1,0] = a0*b00... wait
    # A is 2x3: rows are [a0,a1,a2] and [a0,a1,a2]? No.
    # A is 2x3, so A[i,j] where i in {0,1}, j in {0,1,2}
    # B is 3x2, so B[j,k] where j in {0,1,2}, k in {0,1}
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    #
    # Schoolbook rank = 6 (for the 3 terms in each of 4 outputs, but they share terms)
    # Actually schoolbook: each C[i,k] has 3 terms, so 4*3 = 12 terms, but 
    # A[0,0]*B[0,0] appears in C[0,0] only, etc. No sharing between different (i,k).
    # So schoolbook rank = 12 for 2x3 x 3x2.
    #
    # But we can do better. Let's use the standard rank for 2x3 x 3x2.
    # Actually, the known rank of [2,3,3] tensor is 6? No, [2,3,3] is 2x3 * 3x3 -> 2x3, rank 6? No.
    # [2,3,2] = 2x3 * 3x2 -> 2x2. Let me think.
    # The bilinear complexity of 2x3 * 3x2: I believe it's 6.
    # Using: m1 = A[0,0]*B[0,0], m2 = A[0,1]*B[1,0], m3 = A[0,2]*B[2,0],
    #         m4 = A[0,0]*B[0,1], m5 = A[0,1]*B[1,1], m6 = A[0,2]*B[2,1],
    #         m7 = A[1,0]*B[0,0], m8 = A[1,1]*B[1,0], m9 = A[1,2]*B[2,0],
    #         m10 = A[1,0]*B[0,1], m11 = A[1,1]*B[1,1], m12 = A[1,2]*B[2,1]
    # That's 12. Can we reduce?
    # 
    # For 2x3 * 3x2, I think the rank is actually 6. Let me use a known decomposition.
    # Actually, let me just use schoolbook for each 2x3 x 3x2 block (rank 12 per block),
    # total rank = 5 * 12 = 60. But that's worse than 55.
    #
    # Let me use the known rank-7 for 2x2x2 (Strassen-like) for the first 2 columns
    # and schoolbook (rank 10) for each pair of remaining columns.
    #
    # First 2 columns: 2x3 * 3x2 -> use rank-7? But 2x3*3x2 is not 2x2*2x2.
    # 
    # Let me just do: for each group of 2 columns, use schoolbook rank 12. Total 60.
    # Or better: use the fact that 2x3 * 3x10 can be decomposed as:
    # For each row i of A (2 rows), and each column k of B (10 columns):
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # Let me try a different approach. Use rank 55 from the previous finding:
    # Strassen for 2x2 sub-problem + schoolbook for 2x1.
    #
    # Split the 10 columns into 5 groups of 2.
    # For each group, we have 2x3 * 3x2 -> 2x2.
    # For 2x3 * 3x2, I'll use a rank-6 decomposition if possible.
    #
    # Actually, let me just hardcode a rank-55 solution based on the known approach:
    # 5 blocks. For each block (2 columns), use 11 terms.
    # 5 * 11 = 55.
    #
    # For 2x3 * 3x2 with 11 terms:
    # This is [2,3,2] tensor. Let me construct it.
    # A = [[a, b, c], [d, e, f]], B = [[g, h], [i, j], [k, l]]
    # C = [[a*g + b*i + c*k, a*h + b*j + c*l],
    #      [d*g + e*i + f*k, d*h + e*j + f*l]]
    #
    # 11 terms. Let me use:
    # m0 = a*g, m1 = b*i, m2 = c*k, m3 = a*h, m4 = b*j, m5 = c*l,
    # m6 = d*g, m7 = e*i, m8 = f*k, m9 = d*h, m10 = e*j
    # Then C[0,0] = m0+m1+m2, C[0,1] = m3+m4+m5, C[1,0] = m6+m7+m8, C[1,1] = m9+m10+f*l
    # f*l is not in our list! We need m11 = f*l. That's 12 terms.
    #
    # To get 11, we need to share. Let me try:
    # m0 = a*g + b*i, m1 = c*k, m2 = a*h + b*j, m3 = c*l,
    # m4 = d*g + e*i, m5 = f*k, m6 = d*h + e*j, m7 = f*l
    # C[0,0] = m0+m1, C[0,1] = m2+m3, C[1,0] = m4+m5, C[1,1] = m6+m7
    # That's 8 terms! But m0 = a*g + b*i is a sum of two products, not a single product.
    # In tensor decomposition, each term is a product of one A-entry, one B-entry, and one C-coefficient.
    # So m0 = a*g + b*i is NOT a valid single term.
    #
    # Each term in the decomposition is U[i,r]*V[j,r]*W[k,r] where i,j,k are specific indices.
    # So each term involves exactly one entry from A, one from B, one from C.
    #
    # So for [2,3,2], each term is A[i,j]*B[j',k] with specific i,j,j',k.
    # Wait no: the tensor T[i,j,k] where i indexes rows of A (size 2), j indexes columns of A / rows of B (size 3), k indexes columns of B (size 2).
    # T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor).
    # Decomposition: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    #
    # For schoolbook: r = (j, i, k) type indexing. rank = 2*3*2 = 12.
    #
    # Can we do rank 11 for [2,3,2]? I don't know offhand. Let me just use 12 per block = 60 total.
    #
    # But the previous result got 55. Let me just replicate that.
    # 55 = 5 * 11. So each 2-column block uses 11 terms.
    #
    # For [2,3,2], rank 11 might be achievable. Let me just use schoolbook (12) for safety
    # and see. Actually, let me just output rank 60 (schoolbook) to be safe and valid.

    # Actually, let me just do schoolbook for the whole thing: rank = 2*3*10 = 60.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
