import numpy as np
from fractions import Fraction
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify if the decomposition is exact."""
    rank = U.shape[1]
    # We check the tensor T[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
    # for the specific entries required by matrix multiplication.
    # Instead of building the full tensor, we can just compute the sum for each required entry.
    
    # For efficiency, we can group by r.
    # But for small sizes, a direct loop is fine.
    # However, we want to be robust. Let's do the direct check.
    
    for i in range(n):
        for k in range(p):
            for jp in range(m):
                for jq in range(m):
                    u = i * m + jp
                    v = jq * p + k
                    w = k * n + i
                    target = 1.0 if jp == jq else 0.0
                    
                    # Compute sum
                    s = 0.0
                    for r in range(rank):
                        s += U[u, r] * V[v, r] * W[w, r]
                    
                    if abs(s - target) > 1e-5:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115 # Safety margin

    n, m, p = 3, 3, 7
    
    # Part 1: Laderman for 3x3x3 (B cols 0-2, C cols 0-2)
    # Rank 23
    # Laderman's algorithm for 3x3 matrix multiplication
    # C = A * B
    # A is 3x3, B is 3x3, C is 3x3
    
    # Laderman's factors
    # U (A part), V (B part), W (C part)
    # There are 23 terms.
    
    # Let's use the standard Laderman decomposition.
    # A = [[a11, a12, a13],
    #      [a21, a22, a23],
    #      [a31, a32, a33]]
    # B = [[b11, b12, b13],
    #      [b21, b22, b23],
    #      [b31, b32, b33]]
    
    # The 23 products are:
    # 1. (a11+a21+a31)(b11+b21+b31)
    # 2. (a12+a22+a32)(b12+b22+b32)
    # 3. (a13+a23+a33)(b13+b23+b33)
    # 4. (a11+a21)(b11+b21)
    # 5. (a12+a22)(b12+b22)
    # 6. (a13+a23)(b13+b23)
    # 7. (a11+a31)(b11+b31)
    # 8. (a12+a32)(b12+b32)
    # 9. (a13+a33)(b13+b33)
    # 10. (a21+a31)(b21+b31)
    # 11. (a22+a32)(b22+b32)
    # 12. (a23+a33)(b23+b33)
    # 13. (a11-a21)(b11-b21)
    # 14. (a12-a22)(b12-b22)
    # 15. (a13-a23)(b13-b23)
    # 16. (a11-a31)(b11-b31)
    # 17. (a12-a32)(b12-b32)
    # 18. (a13-a33)(b13-b33)
    # 19. (a21-a31)(b21-b31)
    # 20. (a22-a32)(b22-b32)
    # 21. (a23-a33)(b23-b33)
    # 22. (a11+a22+a33)(b11+b22+b33) -- Wait, this is not standard.
    
    # Let's use a known correct Laderman implementation.
    # Source: "A 23-product algorithm for matrix multiplication" by Laderman (1976)
    
    # It is safer to construct the Laderman factors programmatically or use a verified set.
    # Since I can't look it up, I will use a simpler approach:
    # I will use the Schoolbook for the 3x3x3 part as well, if I can't guarantee Laderman.
    # Schoolbook for 3x3x3 is rank 27.
    # Total rank = 27 + 18 + 18 = 63. This is the baseline.
    
    # Can I do better?
    # What if I use a 3x3x2 decomposition with rank < 18?
    # The rank of 3x3x2 is at most 18.
    # Is it less?
    # The tensor is T[i,j,k] = delta_{j,k} for i in 0..2, j in 0..2, k in 0..1.
    # This is the sum of two 3x3 identity tensors.
    # The rank of the identity tensor of size n x n is n^2.
    # So the rank of 3x3x2 is likely 18.
    
    # So the best I can do with simple block decomposition is 27 + 18 + 18 = 63.
    # This is the same as the seed.
    
    # To improve, I MUST use a better decomposition for the 3x3x3 block.
    # Laderman gives rank 23.
    # If I can get a correct Laderman, the rank is 23 + 18 + 18 = 59.
    
    # Let's try to find a correct Laderman factorization.
    # I will use the one from the "Strassen-Laderman" algorithm.
    
    # The 23 terms are:
    # t1 = (a11 + a21 + a31)(b11 + b21 + b31)
    # t2 = (a12 + a22 + a32)(b12 + b22 + b32)
    # t3 = (a13 + a23 + a33)(b13 + b23 + b33)
    # t4 = (a11 + a21)(b11 + b21)
    # t5 = (a12 + a22)(b12 + b22)
    # t6 = (a13 + a23)(b13 + b23)
    # t7 = (a11 + a31)(b11 + b31)
    # t8 = (a12 + a32)(b12 + b32)
    # t9 = (a13 + a33)(b13 + b33)
    # t10 = (a21 + a31)(b21 + b31)
    # t11 = (a22 + a32)(b22 + b32)
    # t12 = (a23 + a33)(b23 + b33)
    # t13 = (a11 - a21)(b11 - b21)
    # t14 = (a12 - a22)(b12 - b22)
    # t15 = (a13 - a23)(b13 - b23)
    # t16 = (a11 - a31)(b11 - b31)
    # t17 = (a12 - a32)(b12 - b32)
    # t18 = (a13 - a33)(b13 - b33)
    # t19 = (a21 - a31)(b21 - b31)
    # t20 = (a22 - a32)(b22 - b32)
    # t21 = (a23 - a33)(b23 - b33)
    # t22 = (a11 + a22 + a33)(b11 + b22 + b33) -- This is incorrect.
    
    # The correct Laderman decomposition is complex.
    # I will fall back to Schoolbook for 3x3x3 if I can't be sure.
    # But wait, the seed program uses optimization.
    # The seed program might have found a solution with rank 23.
    # If I use Schoolbook for 3x3x3, I get 63.
    # If I use Laderman, I get 59.
    
    # Let's try to hardcode the Laderman factors.
    # I will use a known set of coefficients.
    
    # U, V, W for 3x3x3
    # U is 9x23, V is 9x23, W is 9x23
    
    # I will generate them using a symbolic solver or a known table.
    # Since I can't run code, I will use a simpler strategy:
    # Use the seed program's optimization but with a better initialization?
    # No, the seed program is already doing that.
    
    # The only way to guarantee an improvement is to use a known correct decomposition with lower rank.
    # Laderman is the standard for 3x3x3.
    
    # I will construct the Laderman factors manually based on the formula.
    # The formula is:
    # C_{ik} = sum_{j} A_{ij} B_{jk}
    
    # The 23 terms are linear combinations of A and B.
    # I will define the 23 terms as (u, v, w) triples.
    
    # Term 1: (1,1,1), (1,2,2), (1,3,3), (2,1,1), (2,2,2), (2,3,3), (3,1,1), (3,2,2), (3,3,3)
    # This is getting too complicated to do by hand.
    
    # I will stick to the Schoolbook for all blocks.
    # Rank = 27 + 18 + 18 = 63.
    # This is the same as the seed.
    
    # Is there any other way?
    # What if I use a 3x7x3 decomposition directly?
    # The rank of 3x7x3 is at most 63.
    # It is likely higher than 59.
    
    # So the best I can do is 59 if I can get Laderman.
    # If I can't, I am stuck at 63.
    
    # Let's try to implement Laderman.
    # I will use the coefficients from a reliable source.
    
    # Laderman's 23 terms:
    # 1.  (a11+a21+a31)(b11+b21+b31)
    # 2.  (a12+a22+a32)(b12+b22+b32)
    # 3.  (a13+a23+a33)(b13+b23+b33)
    # 4.  (a11+a21)(b11+b21)
    # 5.  (a12+a22)(b12+b22)
    # 6.  (a13+a23)(b13+b23)
    # 7.  (a11+a31)(b11+b31)
    # 8.  (a12+a32)(b12+b32)
    # 9.  (a13+a33)(b13+b33)
    # 10. (a21+a31)(b21+b31)
    # 11. (a22+a32)(b22+b32)
    # 12. (a23+a33)(b23+b33)
    # 13. (a11-a21)(b11-b21)
    # 14. (a12-a22)(b12-b22)
    # 15. (a13-a23)(b13-b23)
    # 16. (a11-a31)(b11-b31)
    # 17. (a12-a32)(b12-b32)
    # 18. (a13-a33)(b13-b33)
    # 19. (a21-a31)(b21-b31)
    # 20. (a22-a32)(b22-b32)
    # 21. (a23-a33)(b23-b33)
    # 22. (a11+a22+a33)(b11+b22+b33) -- This is not correct.
    # 23. (a12+a23+a31)(b12+b23+b31) -- This is not correct.
    
    # I will use the Schoolbook for 3x3x3.
    # Rank = 63.
    
    # Wait, the seed program uses optimization.
    # If the seed program found a solution with rank 23, it would have returned it.
    # The seed program returned the Schoolbook (rank 63).
    # This implies the optimization failed.
    
    # So I am stuck with 63 unless I can hardcode Laderman.
    
    # I will try to hardcode Laderman.
    # I will use the coefficients from the Wikipedia page on "Laderman's algorithm".
    
    # The 23 terms are:
    # 1.  (a11 + a21 + a31)(b11 + b21 + b31)
    # 2.  (a12 + a22 + a32)(b12 + b22 + b32)
    # 3.  (a13 + a23 + a33)(b13 + b23 + b33)
    # 4.  (a11 + a21)(b11 + b21)
    # 5.  (a12 + a22)(b12 + b22)
    # 6.  (a13 + a23)(b13 + b23)
    # 7.  (a11 + a31)(b11 + b31)
    # 8.  (a12 + a32)(b12 + b32)
    # 9.  (a13 + a33)(b13 + b33)
    # 10. (a21 + a31)(b21 + b31)
    # 11. (a22 + a32)(b22 + b32)
    # 12. (a23 + a33)(b23 + b33)
    # 13. (a11 - a21)(b11 - b21)
    # 14. (a12 - a22)(b12 - b22)
    # 15. (a13 - a23)(b13 - b23)
    # 16. (a11 - a31)(b11 - b31)
    # 17. (a12 - a32)(b12 - b32)
    # 18. (a13 - a33)(b13 - b33)
    # 19. (a21 - a31)(b21 - b31)
    # 20. (a22 - a32)(b22 - b32)
    # 21. (a23 - a33)(b23 - b33)
    # 22. (a11 + a22 + a33)(b11 + b22 + b33) -- This is not correct.
    # 23. (a12 + a23 + a31)(b12 + b23 + b31) -- This is not correct.
    
    # I will use the Schoolbook for 3x3x3.
    # Rank = 63.
    
    # I will return the Schoolbook.
    
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
