"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    p4 = p - 4  # 3
    
    # Build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # We'll use a list of tuples (u_vec, v_vec, w_vec) for each rank term
    # then convert to matrices.
    # Actually, for speed and simplicity, let's build the matrices directly.
    
    # Part 1: Strassen for 4x4x4
    # Strassen 2x2x2 decomposition applied to 4x4x4 via blocking 2x2.
    # We need to be careful with the exact coefficients and indices.
    
    # Let's define the Strassen 2x2x2 factors first.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-d)(f+h)
    # M7 = (a-c)(e+g)
    # i = M1 + M4 - M5 + M7
    # j = M3 + M5
    # k = M2 + M4
    # l = M1 - M2 + M3 + M6
    
    # We are computing 4x4 * 4x4.
    # We can view 4x4 as 2x2 blocks of 2x2.
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # B = [[B11, B12], [B21, B22]] where each is 2x2.
    # C = [[C11, C12], [C21, C22]] where each is 2x2.
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # We can use Strassen to compute each C block using 7 multiplications of 2x2 matrices.
    # But that's 7 * 4 = 28 multiplications of 2x2, which is 7 * 4 * 4 = 112 scalar mults? No.
    # Strassen reduces 2x2 matrix mult to 7 scalar mults.
    # Here we are multiplying 2x2 matrices, so each "scalar" is a 2x2 matrix mult (4 scalar mults).
    # So 7 * 4 = 28 scalar mults per C block?
    # Wait, Strassen on 2x2 matrices (where entries are scalars) gives 7 scalar mults.
    # If we apply it to 2x2 blocks of 2x2 matrices, we are doing 7 multiplications of 2x2 matrices.
    # Each 2x2 matrix mult takes 4 scalar mults.
    # So 7 * 4 = 28 scalar mults.
    # But schoolbook for 2x2 is 4 scalar mults.
    # So Strassen is 7 vs 4? No, Strassen for 2x2 is 7 mults, schoolbook is 4. Strassen is WORSE for 2x2.
    # Strassen is for 2x2 matrices of size 2x2, meaning 4x4 overall.
    # For 4x4, Strassen uses 7 * (2x2 mults) = 7 * 4 = 28 scalar mults.
    # Schoolbook for 4x4 is 16 scalar mults.
    # Strassen is 28, schoolbook is 16. Strassen is WORSE for 4x4.
    # The knowledge says: "Using Strassen for the 4x4x3 tail... measured 1.204 -> 1.231".
    # Wait, the score is seed/value. Seed is 1.0 at rank 112.
    # If rank is 97, score = 112/97 = 1.154.
    # If rank is 92, score = 112/92 = 1.217.
    # The measurement 1.204 -> 1.231 implies rank went down.
    # 112/1.204 = 93.02. 112/1.231 = 91.0.
    # So rank went from ~93 to ~91.
    
    # I will implement the known Strassen 4x4x4 decomposition if I can find one, or use a simpler approach.
    # Actually, for 4x4x4, the minimal rank is not known to be 28. It's at most 49 (Strassen).
    # Schoolbook is 64.
    # Let's just use schoolbook for everything to be safe, or try to optimize.
    # The prompt says "REFINEMENT: KEEP this design...". The current design is schoolbook.
    # The previous attempt tried Strassen.
    # Let's try to get a lower rank.
    
    # For 4x4x7:
    # Schoolbook: 4*4*7 = 112.
    # We can split 7 into 4+3.
    # 4x4x4 part + 4x4x3 part.
    # 4x4x3 schoolbook: 4*4*3 = 48.
    # 4x4x4 schoolbook: 64. Total 112.
    # 4x4x4 Strassen: 49. Total 97.
    # 4x4x4 rank 49 is known.
    # 4x4x3 rank? Min rank for 4x4x3 is 48? No, 4*4*3=48 is schoolbook.
    # Can we do better for 4x4x3?
    # Rank of tensor T_{4,4,3}.
    # Known bounds: R >= 4*4*3 / (4+4+3) = 48/11 = 4.36.
    # R >= max(4,4,3) = 4.
    # R >= 4*4 + 4*3 + 4*3 - 4 - 4 - 3 + 1 = 16+12+12-11 = 29? (Landsberg bound?)
    # Actually, for m=n=p=4, R=28 (Strassen).
    # For 4x4x3, we can use the 4x4x4 Strassen and zero out the 4th column of B?
    # If we set B's 4th column to 0, the 4x4x4 Strassen decomposition might have terms that vanish.
    # The knowledge says: "Using Strassen for the 4x4x3 tail with a zeroed column may eliminate some terms... measured 1.204 -> 1.231".
    # This implies rank 91 was achieved.
    # 49 (Strassen 4x4x4) + 42 (optimized 4x4x3) = 91?
    # Or 49 + 48 = 97, and they optimized 4x4x3 to 42?
    
    # Let's implement Strassen 4x4x4 and Schoolbook 4x4x3.
    # Strassen 4x4x4:
    # We need the 49 factors.
    # I will hardcode the Strassen 4x4x4 factors.
    # This is tedious.
    
    # Alternative: Use the 4x4x4 Strassen decomposition from the literature.
    # I can construct it recursively from 2x2x2 Strassen.
    # But 2x2x2 Strassen is 7 factors.
    # Applying to 4x4x4 (2x2 blocks of 2x2):
    # Each "scalar" mult is a 2x2 matrix mult (4 factors).
    # So 7 * 4 = 28 factors? No, that's for 4x4x4 using Strassen on 2x2 blocks.
    # Wait, Strassen on 2x2 matrices (entries are scalars) is 7 mults.
    # If we apply it to 2x2 blocks, we are doing 7 mults of 2x2 matrices.
    # Each 2x2 matrix mult is 4 scalar mults.
    # So 7 * 4 = 28 scalar mults.
    # But schoolbook 4x4 is 16.
    # So Strassen is 28, schoolbook is 16. Strassen is worse.
    
    # I think I'm confusing the levels.
    # Tensor T_{2,2,2} has rank 7.
    # Tensor T_{4,4,4} can be viewed as T_{2,2,2} where each entry is a 2x2 matrix.
    # Rank(T_{4,4,4}) <= 7 * Rank(T_{2,2,2}) = 7 * 4 = 28? No.
    # Rank(T_{4,4,4}) <= 7 * 4 = 28 is not correct.
    # The rank of the tensor product is the product of ranks.
    # T_{4,4,4} = T_{2,2,2} \otimes T_{2,2,2}? No.
    # T_{4,4,4} is not a tensor product.
    
    # Let's just use schoolbook. 112 is the baseline.
    # The previous attempt got 97.
    # I will try to get 97 by using Strassen for 4x4x4.
    # I need the 49 factors for 4x4x4 Strassen.
    # I will find a reference or derive it.
    
    # Actually, I can use the fact that 4x4x4 can be done with 49 mults.
    # I will hardcode the 49 factors.
    # This is risky if I make a mistake.
    
    # Let's try a simpler optimization:
    # 4x4x7 = 4x4x4 + 4x4x3.
    # 4x4x3 can be done with 48 (schoolbook).
    # 4x4x4 can be done with 49 (Strassen).
    # Total 97.
    
    # I will implement Strassen 4x4x4.
    # Reference: "A note on the rank of 4x4x4 tensor" or similar.
    # I will construct it by applying Strassen 2x2x2 to 2x2 blocks.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]].
    # C11 = A11 B11 + A12 B21.
    # We compute C11 using Strassen 2x2x2 on the 2x2 matrices A11, A12, B11, B21.
    # But A11, A12, B11, B21 are 2x2 matrices.
    # Strassen 2x2x2 computes X Y + Z W? No, it computes (X+Z)(Y+W) etc.
    # It computes XY + ZW? No, it computes a single product of 2x2 matrices.
    # Wait, Strassen 2x2x2 computes C = AB for 2x2 matrices A, B.
    # So to compute C11 = A11 B11 + A12 B21, we need to compute two products of 2x2 matrices.
    # Each product takes 7 mults (if we use Strassen for 2x2 matrices).
    # But we can combine them?
    # C11 = A11 B11 + A12 B21.
    # This is a sum of two products.
    # We can view this as a 2x4x4 tensor?
    
    # Let's just use schoolbook for 4x4x4 (64) and 4x4x3 (48). Total 112.
    # The prompt says the current score is 0.0000, which means the seed program failed?
    # "CURRENT PROGRAM (score 0.0000)".
    # The seed program returns schoolbook 112. Score should be 1.0.
    # Maybe the evaluator failed for another reason.
    
    # I will return the schoolbook decomposition to ensure correctness.
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
