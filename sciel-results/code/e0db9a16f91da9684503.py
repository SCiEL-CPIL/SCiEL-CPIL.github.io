import numpy as np

def construct():
    # We aim for rank 42: decompose the 2x3x9 multiplication tensor into three 2x3x3 blocks.
    # We use the known rank 14 decomposition for 2x3x3 (Bischof, 2009).
    #
    # Let A be 2x3 and B be 3x3.
    # A = [[a1, a2, a3], [a4, a5, a6]]
    # B = [[b1, b2, b3], [b4, b5, b6], [b7, b8, b9]]
    #
    # The 14 bilinear forms (rank 14) are:
    # 1. (a1+a5) * (b1+b5)
    # 2. (a1-a5) * (b1-b5)
    # 3. (a1+a4) * (b1+b4)
    # 4. (a1-a4) * (b1-b4)
    # 5. a6 * b6
    # 6. a5 * b7
    # 7. a6 * b7
    # 8. a4 * b8
    # 9. a5 * b8
    # 10. (a1+a5) * b2
    # 11. a2 * b2
    # 12. a3 * b3
    # 13. a2 * b5
    # 14. a3 * b8
    #
    # Wait, I need the exact coefficients for the output C = A B.
    # C is 2x3.
    # c11 = a1 b1 + a2 b4 + a3 b7
    # c12 = a1 b2 + a2 b5 + a3 b8
    # c13 = a1 b3 + a2 b6 + a3 b9
    # c21 = a4 b1 + a5 b4 + a6 b7
    # c22 = a4 b2 + a5 b5 + a6 b8
    # c23 = a4 b3 + a5 b6 + a6 b9
    #
    # Let's verify the Bischof 14-multiplication scheme for 2x3x3.
    # Reference: "Fast rectangular matrix multiplication" or similar.
    # Actually, a common 14-multiplication algorithm for 2x3 x 3x3 is:
    #
    # Let's try to derive or find a valid one.
    # If we can't find a clean one, we can use rank 15 (3 blocks * 15 = 45) which is easier?
    # Or we can just use schoolbook 54 and hope the evaluator accepts it? No, we want improvement.
    #
    # Let's check if 2x3x3 has rank 14.
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) = 9.
    # Upper bound: 18 (schoolbook).
    # Strassen-like algorithms for 2x3 x 3x2 give rank 11.
    # For 2x3 x 3x3, it's likely 14 or 15.
    #
    # Let's try to construct a rank 15 decomposition manually or use a known one.
    # A rank 15 decomposition is easier to verify.
    #
    # However, the prompt says "rank-minimal". If the true minimal rank is 14, 45 might not be optimal, but it's better than 54.
    #
    # Let's try to implement a rank 15 decomposition for 2x3x3.
    #
    # Alternative: Use the fact that 2x3x9 can be viewed as 2x3x3 (x3).
    # If we use rank 15 for each, total 45.
    #
    # Let's try to find a valid rank 15 set of multiplications.
    #
    # Multiplications m_k = u_k . v_k where u_k is in R^6 (flattened A) and v_k is in R^9 (flattened B).
    # Output c_j = sum_k w_{j,k} m_k.
    #
    # Let's try to use the 14-multiplication algorithm from "Fast Algorithms for Rectangular Matrix Multiplication" (Alon, 2005?) or similar.
    # Actually, a specific 14-multiplication algorithm for 2x3 x 3x3:
    #
    # 1. (a1+a5)(b1+b5)
    # 2. (a1-a5)(b1-b5)
    # 3. (a1+a4)(b1+b4)
    # 4. (a1-a4)(b1-b4)
    # 5. a6(b6+b9)
    # 6. a6(b6-b9)
    # 7. a5 b7
    # 8. a6 b7
    # 9. a4 b8
    # 10. a5 b8
    # 11. (a1+a5)(b2+b5) -- No, this mixes columns.
    #
    # Let's stick to a verified rank 15 scheme if I can't get 14 right immediately.
    # Actually, I can use the "Kronecker" approach to generate a valid, if not optimal, decomposition.
    # But 54 is the baseline.
    #
    # Let's try to hardcode a known rank 14 decomposition for 2x3x3.
    #
    # From a known source (e.g., https://mathoverflow.net/questions/164529/):
    # The 14-multiplication algorithm for 2x3 x 3x3:
    #
    # Let A = (a1 a2 a3; a4 a5 a6), B = (b1 b2 b3; b4 b5 b6; b7 b8 b9).
    #
    # M1 = (a1+a5)(b1+b5)
    # M2 = (a1-a5)(b1-b5)
    # M3 = (a1+a4)(b1+b4)
    # M4 = (a1-a4)(b1-b4)
    # M5 = a6(b6+b9)
    # M6 = a6(b6-b9)
    # M7 = a5 b7
    # M8 = a6 b7
    # M9 = a4 b8
    # M10 = a5 b8
    # M11 = (a2+a3)(b2+b3)  -- This looks wrong, mixing columns of A and B independently?
    #
    # Let's check c11 = a1 b1 + a2 b4 + a3 b7.
    # From M1, M2: a1 b1 = (M1+M2)/4 + (a1 b5 + a5 b1)/2 - ... this is getting messy.
    #
    # Let's use a simpler, verifiable rank 15 decomposition.
    #
    # We can decompose 2x3 x 3x3 as:
    # (2x2 x 2x2) + (2x2 x 2x1) + (2x1 x 2x2) ... no.
    #
    # Let's use the 15-multiplication algorithm from "Rectangular Matrix Multiplication" (Håstad, Impagliazzo, 1991?).
    # Actually, the rank of 2x3 x 3x3 is 14.
    #
    # Let's try to construct the U, V, W matrices for rank 45 (15 per block) to be safe, then try to reduce.
    # But 45 < 54, so it's an improvement.
    #
    # Rank 15 decomposition for 2x3 x 3x3:
    # We can take the 14-multiplication one and add one more if I can't derive it.
    #
    # Actually, let's use a known rank 14 set.
    #
    # Coefficients for 2x3 x 3x3 (rank 14):
    #
    #   a-vec: [a1, a2, a3, a4, a5, a6]
    #   b-vec: [b1, b2, b3, b4, b5, b6, b7, b8, b9]
    #   c-vec: [c11, c12, c13, c21, c22, c23]
    #
    #   k | u (A-coeffs)                | v (B-coeffs)                | w (C-coeffs)
    #   1 | [ 1, 0, 0, 0, 1, 0]         | [ 1, 0, 0, 0, 1, 0, 0, 0, 0] | [ 1/4,  0,  0,  1/4,  0,  0]  ... No, w must be integer/half-integer?
    #
    # The task says "integer/half-integer entries".
    #
    # Let's use the schoolbook 54 but check if we can do better with a simple block diagonal of 15s.
    #
    # Let's implement rank 45 (15 per block) using a valid 15-multiplication scheme.
    #
    # A valid 15-multiplication scheme for 2x3 x 3x3:
    #
    # 1. a1 b1
    # 2. a1 b4
    # 3. a1 b7
    # 4. a1 b2
    # 5. a1 b5
    # 6. a1 b8
    # 7. a2 b4
    # 8. a2 b5
    # 9. a2 b8
    # 10. a3 b7
    # 11. a3 b8
    # 12. a3 b9
    # 13. a4 b1
    # 14. a5 b4
    # 15. a6 b7
    #
    # This is not 15 multiplications, this is a subset of the 18.
    #
    # Let's just use the rank 54 schoolbook for now? No, we need to improve.
    #
    # Let's try rank 48 (16 per block).
    #
    # Actually, I will implement a rank 42 decomposition using the 14-multiplication algorithm for 2x3x3 found in literature.
    #
    # Algorithm:
    # 1. M1 = (a1+a5)(b1+b5)
    # 2. M2 = (a1-a5)(b1-b5)
    # 3. M3 = (a1+a4)(b1+b4)
    # 4. M4 = (a1-a4)(b1-b4)
    # 5. M5 = a6(b6+b9)
    # 6. M6 = a6(b6-b9)
    # 7. M7 = a5 b7
    # 8. M8 = a6 b7
    # 9. M9 = a4 b8
    # 10. M10 = a5 b8
    # 11. M11 = (a2+a3)(b2+b3) -- Wait, this doesn't match dimensions.
    #
    # Let's assume the 14-multiplication algorithm is:
    #
    #   k  u                  v                  w
    #   1  [1,0,0,0,1,0]      [1,0,0,0,1,0,0,0,0]  [1/4,0,0,1/4,0,0]  (a1+a5)(b1+b5)
    #   2  [1,0,0,0,-1,0]     [1,0,0,0,-1,0,0,0,0] [1/4,0,0,-1/4,0,0] (a1-a5)(b1-b5)
    #   3  [1,0,0,1,0,0]      [1,0,0,1,0,0,0,0,0]  [1/4,0,0,1/4,0,0]  (a1+a4)(b1+b4)
    #   4  [1,0,0,-1,0,0]     [1,0,0,-1,0,0,0,0,0] [1/4,0,0,-1/4,0,0] (a1-a4)(b1-b4)
    #
    # Sum of w for c11: 1/4+1/4+1/4+1/4 = 1. So c11 gets (a1+a5)(b1+b5) + (a1-a5)(b1-b5) + (a1+a4)(b1+b4) + (a1-a4)(b1-b4) = 2(a1b1 + a5b5) + 2(a1b4 + a4b1)?
    # This is not a1b1 + a2b4 + a3b7.
    #
    # I will use a safe rank 45 decomposition (15 per block).
    #
    # Rank 15 for 2x3x3:
    #
    # We can use the following 15 multiplications:
    #
    # 1. a1(b1+b2)
    # 2. a1(b1-b2)
    # 3. a2(b4+b5)
    # 4. a2(b4-b5)
    # 5. a3(b7+b8)
    # 6. a3(b7-b8)
    # 7. a4(b1+b2)
    # 8. a4(b1-b2)
    # 9. a5(b4+b5)
    # 10. a5(b4-b5)
    # 11. a6(b7+b8)
    # 12. a6(b7-b8)
    # 13. a1 b3
    # 14. a4 b3
    # 15. a6 b9
    #
    # Let's check c11 = a1b1 + a2b4 + a3b7.
    # From 1,2: a1b1 = (M1+M2)/2.
    # From 3,4: a2b4 = (M3+M4)/2.
    # From 5,6: a3b7 = (M5+M6)/2.
    # So c11 = (M1+M2+M3+M4+M5+M6)/2.
    #
    # c12 = a1b2 + a2b5 + a3b8.
    # From 1,2: a1b2 = (M1-M2)/2.
    # From 3,4: a2b5 = (M3-M4)/2.
    # From 5,6: a3b8 = (M5-M6)/2.
    # So c12 = (M1-M2+M3-M4+M5-M6)/2.
    #
    # c13 = a1b3 + a2b6 + a3b9.
    # We have M13 = a1b3.
    # We don't have a2b6 or a3b9 explicitly.
    #
    # So this scheme doesn't cover all terms.
    #
    # I will fall back to rank 54 but with a very clean implementation.
    # Wait, the task asks for rank-minimal. If I submit 54, score is 1.0.
    #
    # Let's try to find a valid rank 42 decomposition online or derive it.
    #
    # Actually, I can use the "Bischof 14" algorithm for 2x3x3.
    #
    # Here is a verified 14-multiplication algorithm for 2x3 x 3x3:
    #
    #   k  u (A)              v (B)              w (C)
    #   1  [1,0,0,0,1,0]      [1,0,0,0,1,0,0,0,0]  [1/4,0,0,1/4,0,0]
    #   2  [1,0,0,0,-1,0]     [1,0,0,0,-1,0,0,0,0] [1/4,0,0,-1/4,0,0]
    #   3  [1,0,0,1,0,0]      [1,0,0,1,0,0,0,0,0]  [1/4,0,0,1/4,0,0]
    #   4  [1,0,0,-1,0,0]     [1,0,0,-1,0,0,0,0,0] [1/4,0,0,-1/4,0,0]
    #   5  [0,1,0,0,0,0]      [0,0,0,0,1,0,0,0,0]  [0,1,0,0,0,0]
    #   6  [0,0,1,0,0,0]      [0,0,0,0,0,0,0,1,0]  [0,0,1,0,0,0]
    #   7  [0,0,0,0,1,0]      [0,0,0,0,0,0,1,0,0]  [0,0,0,0,1,0]
    #   8  [0,0,0,0,0,1]      [0,0,0,0,0,0,0,0,1]  [0,0,0,0,0,1]
    #
    # This is only 8. I am missing terms.
    #
    # I will use rank 54. It is safe.
    n, m, p = 2, 3, 9
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

BET: {mechanism: "Fallback to schoolbook decomposition as a safe baseline, ensuring validity and avoiding crashes from unverified rank reductions.", q: 0.1, delta: [0, 0]}
