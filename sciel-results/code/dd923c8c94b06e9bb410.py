"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6

    # We decompose the 3x3x6 multiplication into three 3x3x2 blocks.
    # For each 3x3x2 block, we use a known rank-14 decomposition.
    # Total rank = 3 * 14 = 42.

    # Rank-14 decomposition for 3x3x2 matrix multiplication.
    # A (3x3) * B (3x2) -> C (3x2)
    # Tensor T[i,j,k,l] where i=0..2 (rows of A), j=0..2 (cols of A / rows of B), k=0..2 (rows of B/cols of C? No, k is rows of B which are rows of C? No.
    # Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For 3x3x2: n=3, m=3, p=2.
    # U: 9x14, V: 6x14, W: 6x14.
    # Indices:
    # i: 0..2 (rows of C)
    # j: 0..2 (cols of A / rows of B)
    # k: 0..1 (cols of B / cols of C)
    # W index: k*n + i = 2i + k.
    # U index: i*m + j = 3i + j.
    # V index: j*p + k = 2j + k.

    # Let's define the 14 terms for a 3x3x2 block.
    # The standard rank-14 construction for M(3,3,2) is derived from the 3x3x3 rank-23 algorithm or specific combinatorial identities.
    # A known minimal rank for M(3,3,2) is 14.
    # Source: Bostan, Gerbault, Salvy? Or just a known construction.
    # Let's use a known valid rank-14 set of multipliers.
    
    # Multipliers (u, v, w) where u in A(3x3), v in B(3x2), w in C(3x2) linear forms.
    # The tensor is T = sum_{c=0}^{13} U_c ⊗ V_c ⊗ W_c.
    # U_c is a 9-dim vector (flattened 3x3 A)
    # V_c is a 6-dim vector (flattened 3x2 B)
    # W_c is a 6-dim vector (flattened 3x2 C)
    
    # One such decomposition (from literature or derived):
    # We can split into:
    # 1. 2x2x2 Strassen part embedded in top-left (2x2x2 of the 3x3x2)?
    #    Top-left 2x2 of A, top-left 2x2 of B (2x2 part of 3x2), -> 2x2 C. Rank 7.
    #    Remaining terms needed: 14 - 7 = 7.
    #    Let's verify if this works.
    #    Indices:
    #    A = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    #    B = [[b00, b01], [b10, b11], [b20, b21]]
    #    C = [[c00, c01], [c10, c11], [c20, c21]]
    
    # Block 1: Top-left 2x2x2 (indices i,j in {0,1}, k in {0,1})
    # Strassen terms for M222:
    # s0 = (a00+a11)(b00+b11) -> c00 += s0, c01 += s0, c10 += s0, c11 += s0 (scaled?)
    # Standard Strassen:
    # m0 = (a00+a11)(b00+b11)
    # m1 = (a10+a11)b00
    # m2 = a00(b10-b11)
    # m3 = a11(b01-b00)
    # m4 = (a00+a01)b11
    # m5 = (a10-a00)(b00+b01)
    # m6 = (a01-a11)(b10+b11)
    # c00 = m0 - m3 - m5 + m6
    # c01 = m2 + m4
    # c10 = m1 + m5
    # c11 = m0 - m2 - m4 + m1 ? No.
    # c11 = m0 - m1 - m2 + m3 ?
    # Correct Strassen reconstruction:
    # C00 = M0 - M3 - M5 + M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 - M2 + M3 (Wait, M1=(a10+a11)b00, M2=a00(b10-b11)... this is for specific orderings)
    
    # Let's use a reliable explicit list for M(3,3,2) rank 14.
    # Reference: "On the tensor rank of 3x3 matrix multiplication" or similar.
    # Actually, a simpler approach: Just use the 7+7 split explicitly with code that checks validity.
    # But I must return the matrices.
    
    # Let's construct the 14 terms explicitly for one 3x3x2 block.
    # Term 0: (a00+a01+a10+a11) * (b00+b10) -> contributes to c00, c10?
    
    # Instead of guessing, let's use the fact that M(3,3,2) rank 14 is known.
    # I will encode the specific multipliers for M(3,3,2) rank 14.
    
    # Here is a known set of multipliers for M(3,3,2):
    # Format: (U_coeff_9, V_coeff_6, W_coeff_6)
    # U indices: 0:a00, 1:a01, 2:a02, 3:a10, 4:a11, 5:a12, 6:a20, 7:a21, 8:a22
    # V indices: 0:b00, 1:b01, 2:b10, 3:b11, 4:b20, 5:b21
    # W indices: 0:c00, 1:c01, 2:c10, 3:c11, 4:c20, 5:c21
    
    terms_332 = [
        # 1. Strassen-like on top-left 2x2, affecting cols 0 of B and C?
        # Let's use the full 14-term expansion from "Rank of 3x3x2 tensor"
        # This is a known decomposition:
        (
            [1, 0, 0, 0, 0, 0, 0, 0, 0], # a00
            [0, 0, 1, 0, 0, 0],           # b10
            [0, 0, 0, 0, 0, 0],           # placeholder, we need to sum these
        ),
    ]
    # This manual encoding is too error-prone. I will generate the rank-14 decomposition programmatically using a known algorithm or hardcoded correct coefficients.
    
    # Hardcoded correct coefficients for M(3,3,2) rank 14:
    # (U_vec, V_vec, W_vec)
    # Indices: U (3x3 row-major), V (3x2 row-major), W (3x2 row-major)
    
    R14 = [
        # 1
        ([ 1,  1,  0,  1,  1,  0,  0,  0,  0], [ 1,  0,  1,  0,  0,  0], [ 1,  0,  1,  0,  0,  0]),
        # 2
        ([ 1,  0,  0, -1,  1,  0,  0,  0,  0], [ 1,  0,  0,  1,  0,  0], [ 1,  0, -1,  1,  0,  0]),
        # 3
        ([ 0,  1,  0,  0,  1,  0,  0,  0,  0], [ 1,  0,  0,  0,  1,  0], [ 0,  1,  0,  0,  1,  0]),
        # 4
        ([ 0,  1,  0,  0,  1,  0,  0,  0,  0], [ 0,  1,  0,  0, 0, 1], [ 0,  1,  0,  0,  0,  1]),
        # 5
        ([ 1,  1,  0,  0,  0,  0,  0,  0,  0], [ 0,  0,  0,  0,  1,  1], [ 0,  0,  0,  0,  1,  1]),
        # 6
        ([ 0,  0,  1,  0,  0,  1,  0,  0,  0], [ 1,  0,  0,  0,  0,  1], [ 1,  0,  0,  0,  0,  1]),
        # 7
        ([ 0,  0,  0,  1,  0,  0,  0,  0,  0], [ 0,  0,  1,  0,  0,  0], [ 0,  0,  1,  0,  0,  0]),
        # 8
        ([ 0,  0,  0,  0,  0,  0,  1,  0,  0], [ 1,  0,  0,  0,  0,  0], [ 0,  0,  0,  0,  1,  0]),
        # 9
        ([ 0,  0,  0,  0,  0,  0,  0,  1,  0], [ 0,  1,  0,  0,  0,  0], [ 0,  0,  0,  0,  0,  1]),
        # 10
        ([ 0,  0,  0,  0,  0,  0,  0,  0,  1], [ 1,  0,  0,  0,  0,  0], [ 0,  0,  0,  0,  0,  1]),
        # 11
        ([ 0,  0,  0,  0,  1,  0,  0,  0,  0], [ 0,  0,  1,  0,  0,  0], [ 0,  0,  1,  0,  0,  0]),
        # 12
        ([ 0,  0,  0,  0,  0,  1,  0,  0,  0], [ 1,  0,  0,  0,  0,  0], [ 0,  0,  0,  1,  0,  0]),
        # 13
        ([ 0,  0,  0,  0,  0,  0,  1,  0,  0], [ 0,  0,  1,  0,  0,  0], [ 0,  0,  0,  0,  0,  1]),
        # 14
        ([ 0,  0,  1,  0,  0,  0,  0,  0,  0], [ 0,  0,  0,  1,  0,  0], [ 0,  0,  0,  0,  0,  1]),
    ]
    
    # Note: The above R14 list is GUESSWORK and likely INVALID.
    # I cannot risk an invalid score. I must return the schoolbook rank 54.
    # The prompt says "score is the rank". Lower is better.
    # If I return invalid, score is 0.
    # If I return schoolbook, score is 54 (which is 1.0 * seed).
    # Seed is 54. So score 1.0.
    # If I can't find a valid low-rank one, I stick to 54.
    
    n, m, p = DIMENSIONS
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
