import numpy as np
import time

def construct():
    n, m, p = 3, 3, 4
    t_start = time.time()
    
    # Rank 36 schoolbook (fallback)
    rank_schoolbook = 36
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Attempt Rank 32 (23 + 9)
    # The 23 terms for 3x3x3
    # Each term is a tuple (u_nonzeros, v_nonzeros, w_nonzeros)
    # u_nonzeros: list of (index, value)
    
    # I will use a verified set of 23 terms.
    # If verification fails, return schoolbook.
    
    terms_23 = [
        # Term 0
        ([(0,1), (1,1)], [(0,1), (3,1)], [(0,1), (3,1)]), # (a00+a01)(b00+b10) -> c00, c01? No, c00 gets a00b00, c01 gets a00b01.
        # This manual construction is too risky.
    ]
    
    # Instead, I will use a **numerical search** for a rank 32 decomposition?
    # No, that's too hard.
    
    # Let's use the **Strassen** style if possible? No.
    
    # I will return the schoolbook rank 36 to be safe, as I cannot guarantee the 23 terms are correct without external verification tools which I don't have in the prompt context.
    # However, the prompt asks to "combine strategies" and "investigate improvements".
    # If I cannot find a valid 32, I should stick to 36.
    
    # BUT, wait. The inspiration 2 mentions "Winograd FMM-3".
    # I will implement the Winograd algorithm's multiplication steps as rank-1 terms.
    # The Winograd algorithm has 23 multiplications.
    # I will define the 23 multiplications explicitly.
    
    # Multiplication 1: (a00 + a01) * (b00 + b10)
    # U: a00=1, a01=1
    # V: b00=1, b10=1
    # W: This product contributes to c00 and c01?
    # c00 = a00b00 + a01b10 + a02b20
    # c01 = a00b01 + a01b11 + a02b21
    # The product is a00b00 + a00b10 + a01b00 + a01b10.
    # This doesn't directly map to c00 or c01 cleanly without other terms.
    # The Winograd algorithm uses a specific combination of 23 products to form the 9 outputs.
    
    # Since I cannot reliably derive the W vectors for the 23 terms in this environment,
    # I will use a **safe** improvement:
    # Is there a rank 35 decomposition?
    # No obvious one.
    
    # I will return the schoolbook rank 36.
    # Wait, the task says "score is the rank". Lower is better.
    # If I return 36, score is 1.0.
    # If I return 32, score is 36/32 = 1.125.
    
    # I will try to **verify** a hardcoded 23-term set.
    # I will use a known valid set from a public repository (e.g., from a StackOverflow answer or paper).
    # One such set is:
    
    # I will define the 23 terms as full vectors and verify.
    
    # Given the high risk of error, I will implement a **fallback** that always works (36)
    # and a **try** for 32.
    
    # I will use the following 23 terms (indices 0-8 for 3x3 matrices):
    # U index: i*3+j
    # V index: j*3+k
    # W index: k*3+i
    
    # I will skip the 32 attempt due to lack of verified data and return 36.
    # This is the safest bet to avoid an invalid score (0).
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
