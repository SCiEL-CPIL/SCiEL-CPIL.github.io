import time
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 63
    # But we can do better.
    # For 3x3 * 3x3, rank is 23 (or 22 over reals).
    # Here we have 3x3 * 3x7.
    # Let's try to find a low rank decomposition.
    
    # Strategy:
    # 1. Try to hardcode a known low rank decomposition if possible.
    # 2. If not, use a heuristic search to find a sparse rank-? decomposition.
    # 3. Fallback to schoolbook.
    
    # Known lower bounds:
    # R(3,3,7) <= R(3,3,3) * 7? No.
    # R(3,3,7) <= R(3,3,3) + R(3,3,3) + R(3,3,1)?
    # 3x7 = 3x3 + 3x3 + 3x1.
    # R(3,3,3) <= 23.
    # R(3,3,1) = 3 (since C is 3x1, each entry is a dot product of 3 elements, rank 3? No, rank of M_{3,3,1} is 3?
    # C_i = sum_j A_ij B_j. This is 3 independent bilinear forms. Rank 3.
    # So R(3,3,7) <= 23 + 23 + 3 = 49?
    # Actually, we can group the 7 columns into 3+3+1.
    # The first 3 columns use 23 multiplications.
    # The next 3 columns use 23 multiplications.
    # The last 1 column uses 3 multiplications.
    # Total 49.
    
    # Can we do better?
    # Maybe interleave?
    # Let's try to find a rank 49 decomposition explicitly.
    
    # We need to define the 23 multiplication formulas for 3x3 * 3x3.
    # Standard rank 23 algorithm for 3x3.
    # Reference: "Multiplying 3x3 matrices in 23 multiplications" by Stothers? Or similar.
    # Let's use a simpler construction:
    # Block diagonal? No.
    
    # Let's use the Strassen 2x2 to build 3x3?
    # 3x3 can be padded to 4x4? Rank 49 for 4x4? No, 49 is for 4x4.
    # 3x3 rank is 23.
    
    # Let's try to construct the rank 49 decomposition.
    # Columns 0-2 of B (3x3 part).
    # Columns 3-5 of B (3x3 part).
    # Column 6 of B (3x1 part).
    
    # We need the 23 terms for 3x3.
    # Let's find a valid rank 23 decomposition for 3x3.
    # I will use the one from "A 23-multiplication algorithm for 3x3 matrix multiplication".
    # It's tricky to hardcode from memory.
    
    # Alternative: Use rank 27 (Strassen 3x3 via padding to 4x4 is 49, Strassen 2x2 block decomposition of 3x3?)
    # If we treat 3x3 as 2x2 blocks?
    # A = [[A11 A12], [A21 A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # This leads to rank 27?
    # 2x2 needs 7.
    # Cross terms?
    # This is getting complex.
    
    # Let's try a different approach:
    # Use the fact that R(3,3,7) <= 2 * R(3,3,4) - R(3,3,1)? No.
    
    # Let's just try to find a sparse solution with rank < 63 using random initialization and greedy elimination?
    # Or just return the 49 rank decomposition if I can write it.
    
    # Actually, there is a simple bound:
    # R(n,m,p) <= n*m*p.
    # R(3,3,7) <= 23 * 7 / 3? No.
    
    # Let's use the 49 bound construction.
    # We need the 23 terms for 3x3.
    # I will try to find a valid 23-term algorithm online or derive a 27-term one which is easier.
    # Strassen 2x2 is 7 terms.
    # If we decompose 3x3 into four 2x2, 2x1, 1x2, 1x1 blocks, we can use Strassen recursively?
    # No, Strassen is for square matrices.
    
    # Let's use the 27-term algorithm for 3x3.
    # It's easier to construct.
    # A = [ [a b c], [d e f], [g h i] ]
    # B = [ [j k l], [m n o], [p q r] ]
    # 
    # 1. M1 = (a+d)*(j+m)
    # 2. M2 = (b+e)*(k+n)
    # 3. M3 = (c+f)*(l+o)
    # 4. M4 = (g+i)*(p+r)
    # 5. M5 = a*j
    # 6. M6 = d*m
    # 7. M7 = e*n
    # 8. M8 = i*r
    # 9. M9 = (a+b)*(j+k)
    # 10. M10 = (d+e)*(m+n)
    # 11. M11 = (g+h)*(p+q)
    # 12. M12 = (a+c)*(j+l)
    # 13. M13 = (d+f)*(m+o)
    # 14. M14 = (g+i)*(p+r) -> Wait, M4 is (g+i)(p+r). Is this a duplicate?
    # Let's check standard 27 term algorithms.
    # Actually, rank 23 is the best known for 3x3.
    # Rank 27 is easy to get from Strassen 2x2?
    # If we pad 3x3 to 4x4, rank is 49.
    # If we use the "3x3 in 27" algorithm:
    # It is essentially Strassen's algorithm applied to 2x2 blocks, but with some overlap?
    
    # Let's try to implement a rank 49 decomposition using the 23-term 3x3 algorithm.
    # I will hardcode the 23 terms.
    # Source: "Stothers, R. (2010). Multiplying 3x3 matrices in 23 multiplications."
    # The algorithm is:
    # Let A = (a_ij), B = (b_ij).
    # The 23 products are:
    # 1. m1 = (a11 + a22 + a33)(b11 + b22 + b33)
    # 2. m2 = (a11 - a22)(b11 - b22)
    # 3. m3 = (a11 - a33)(b11 - b33)
    # 4. m4 = (a22 - a33)(b22 - b33)
    # 5. m5 = (a12 + a23)(b12 + b23)
    # 6. m6 = (a13 + a21)(b13 + b21)
    # 7. m7 = (a21 + a32)(b21 + b32)
    # 8. m8 = (a31 + a12)(b31 + b12)
    # 9. m9 = (a12 - a21)(b12 - b21)
    # 10. m10 = (a13 - a31)(b13 - b31)
    # 11. m11 = (a23 - a32)(b23 - b32)
    # 12. m12 = a11*b11
    # 13. m13 = a22*b22
    # 14. m14 = a33*b33
    # 15. m15 = a12*b21
    # 16. m16 = a21*b12
    # 17. m17 = a13*b31
    # 18. m18 = a31*b13
    # 19. m19 = a23*b32
    # 20. m20 = a32*b23
    # 21. m21 = (a11+a12+a13)(b11+b21+b31)
    # 22. m22 = (a21+a22+a23)(b12+b22+b32)
    # 23. m23 = (a31+a32+a33)(b13+b23+b33)
    
    # Wait, I need to verify if this is correct.
    # Actually, I don't trust my memory for the exact 23 formulas.
    # Let's use a safer rank 49 construction with the 27-term algorithm for 3x3?
    # Or just use the schoolbook for the 3x3 parts?
    # Rank 9 for 3x3.
    # 9 + 9 + 3 = 21? No, 9+9+3 = 21 is for the multiplications.
    # But the linear combinations might overlap?
    # If I use schoolbook for the first 3x3 block (9 mults),
    # schoolbook for the second 3x3 block (9 mults),
    # schoolbook for the last 3x1 block (3 mults).
    # Total 21 multiplications.
    # Rank 21!
    # Is this valid?
    # C_0:2 = A * B_0:3. (9 mults)
    # C_3:6 = A * B_3:6. (9 mults)
    # C_6:7 = A * B_6:7. (3 mults)
    # Since the B blocks are disjoint in columns, and A is the same,
    # the multiplications are:
    # A_ij * B_jk.
    # For C_0:2, we need A_ij * B_j0, A_ij * B_j1, A_ij * B_j2.
    # For C_3:6, we need A_ij * B_j3, A_ij * B_j4, A_ij * B_j5.
    # For C_6:7, we need A_ij * B_j6.
    # These are 9 + 9 + 3 = 21 distinct pairs of (i,j,k) indices?
    # Yes, because k is different in each block.
    # So the rank is exactly 21?
    # Wait. The rank of the tensor M_{3,3,7} is the minimum number of rank-1 tensors.
    # If I decompose it into 3 disjoint parts, the rank is the sum of the ranks of the parts.
    # Rank of M_{3,3,3} is 23? Or 9 if schoolbook?
    # Schoolbook rank for 3x3 is 9.
    # So if I use schoolbook for each part, the total rank is 9+9+3 = 21.
    # Is rank 21 valid?
    # Yes, because it's a valid decomposition.
    # Can we do better?
    # R(3,3,7) <= 21.
    # Is R(3,3,7) < 21?
    # Lower bound: n*m*p / max(n,m,p)^? No.
    # Lower bound for 3x3x3 is 19.
    # Lower bound for 3x3x7?
    # Maybe 21 is optimal? Or maybe we can share some terms?
    # The A matrix is shared.
    # But the B columns are distinct.
    # The terms are A_ij * B_jk.
    # If we use Strassen for the 3x3 part, we reduce 9 to 23? No, 23 > 9.
    # Strassen reduces 8 to 7 for 2x2.
    # For 3x3, schoolbook is 9. Strassen-based is 23?
    # Wait, Strassen is faster because it uses fewer multiplications.
    # 2x2: 8 -> 7.
    # 3x3: 9 -> ?
    # If I use Strassen on 2x2 subblocks, I might get more than 9 multiplications?
    # Yes, Strassen 3x3 (via 4x4 padding) uses 49 multiplications.
    # Strassen 3x3 (via 2x2 blocks) uses ?
    # Actually, 9 is the schoolbook count.
    # Can we do 3x3 with less than 9 multiplications?
    # No, the lower bound is 9?
    # Actually, R(3,3,3) >= 19. So 9 is NOT the rank.
    # Wait. The rank of the matrix multiplication tensor M_{n,m,p} is the minimum number of multiplications.
    # For 2x2, rank is 7 (Strassen). Schoolbook is 8.
    # For 3x3, rank is 23 (Stothers). Schoolbook is 9?
    # NO. Schoolbook for 3x3 uses 9 multiplications?
    # C_ii = sum_j A_ij B_ji.
    # 3 rows, 3 cols. 9 entries.
    # Each entry is a dot product of length 3.
    # Total 9 * 3 = 27 multiplications?
    # NO.
    # C = A * B.
    # C_ik = sum_j A_ij B_jk.
    # For each i, k, we sum over j.
    # Total multiplications: n * m * p = 3 * 3 * 3 = 27.
    # Ah! I was confusing rank with number of multiplications.
    # Schoolbook rank is n*m*p.
    # For 3x3x3, schoolbook rank is 27.
    # Strassen rank is 23.
    # So my previous "9" was wrong. It was 27.
    
    # So for 3x3x7:
    # Schoolbook rank = 3*3*7 = 63.
    # My block decomposition:
    # Block 1: 3x3x3. Rank 23 (Stothers) or 27 (Schoolbook).
    # Block 2: 3x3x3. Rank 23.
    # Block 3: 3x3x1. Rank 3?
    # M_{3,3,1}. C_i = sum_j A_ij B_j.
    # This is 3 bilinear forms.
    # Rank 3.
    # So total rank = 23 + 23 + 3 = 49.
    # Or 27 + 27 + 3 = 57.
    
    # Can we do better than 49?
    # Maybe the 3x3x1 part can share with the 3x3x3 parts?
    # Unlikely, because the B columns are distinct.
    # But the A is shared.
    # However, the rank is additive for disjoint B supports.
    # So 49 is a solid upper bound.
    
    # Let's implement the rank 49 decomposition.
    # We need the 23-term algorithm for 3x3.
    # I will use the schoolbook (rank 27) for simplicity and safety, resulting in rank 57.
    # 57 is better than 63.
    
    # Let's try to get 49 if I can hardcode Stothers.
    # I will try to find a valid 23-term set.
    # If I fail, I will use 57.
    
    # Let's use 57 first. It's safe.
    
    rank = 57
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: B columns 0-2
    # Schoolbook for 3x3
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # Block 2: B columns 3-5
    for i in range(3):
        for j in range(3):
            for k in range(3, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # Block 3: B column 6
    for i in range(3):
        for j in range(3):
            k = 6
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
