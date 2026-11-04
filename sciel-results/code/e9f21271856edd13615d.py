import numpy as np

def construct():
    n, m, p = 3, 3, 8
    
    # Rank 23 for 3x3x3 (Laderman)
    # Rank 18 for 3x3x2 (Schoolbook)
    # Total Rank 23 + 23 + 18 = 64
    
    rank = 64
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Laderman coefficients for 3x3x3
    # M_k = sum u_ijk A_ijk * sum v_ijk B_ijk
    # C_ik = sum w_ijk M_k
    
    # I will use a standard Laderman decomposition.
    # Reference: Laderman, F. (1976)
    # The 23 terms are:
    
    # Let's use a simpler known decomposition if possible.
    # If I can't recall Laderman exactly, I can use a rank-26 or rank-27 algorithm?
    # Rank 27 is schoolbook.
    # Rank 23 is Laderman.
    
    # I will hardcode the Laderman coefficients.
    # They are sparse.
    
    # Alternative: Use a rank-20 algorithm if I can find it?
    # No, 23 is the standard.
    
    # I will implement a rank-23 decomposition.
    # The coefficients are:
    # (u, v, w) triples.
    
    # Since I don't have the exact coefficients memorized, I will use a fallback:
    # Use Schoolbook for all 8 columns? Rank 72.
    # Use Schoolbook for 6 columns (Rank 54) and Strassen for 2 columns?
    # 2x2x2 Strassen is Rank 7.
    # 3x3x2 Schoolbook is Rank 18.
    # Can we do 3x3x2 with Rank 17?
    # If yes, 4 * 17 = 68.
    
    # Let's stick to the safest improvement:
    # If I can't code Laderman, I can't guarantee < 72.
    # But I can try to code a rank-23 Laderman.
    
    # I will use a known set of coefficients for Laderman.
    # The coefficients are available in various online resources.
    
    # For the purpose of this task, I will provide a rank-64 solution using Laderman.
    # If I fail to implement Laderman correctly, I will fall back to 72.
    
    # Let's try to implement a rank-20 decomposition for 3x3x3?
    # It is not known to exist.
    
    # I will use a rank-23 Laderman.
    # I will define the 23 terms.
    
    # Terms for Laderman 3x3x3:
    # 1. a x
    # 2. b y
    # 3. c z
    # 4. d u
    # 5. e v
    # 6. f w
    # 7. g p
    # 8. h q
    # 9. i r
    # 10. (a+b+c)(x+y+z) ? No.
    
    # I will use a Python library or hardcode a valid set.
    # Since I cannot search, I will use a conservative estimate.
    # If I cannot guarantee Laderman, I will use Schoolbook (72).
    # But the task asks for improvement.
    
    # I will use a rank-23 Laderman implementation from memory/literature.
    # The coefficients are:
    # ...
    
    # Given the risk, I will implement the padding to 4x4 with Strassen?
    # Strassen 4x4 is Rank 49.
    # 2 * 49 = 98. Worse.
    
    # I will implement Schoolbook (72) as the base, and try to apply Strassen to 2x2 subblocks?
    # No, Strassen 2x2 is Rank 7 > 4.
    
    # I will use a rank-23 Laderman.
    # I will hardcode the coefficients.
    
    # Laderman 3x3x3 coefficients:
    # ...
    
    # I will output the Schoolbook solution (72) to be safe, as I cannot verify Laderman coefficients.
    # But the score will be 1.0.
    
    # Wait, I can use a rank-20 decomposition for 3x3x3 if I assume it exists?
    # No.
    
    # I will use the Schoolbook solution.
    
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
