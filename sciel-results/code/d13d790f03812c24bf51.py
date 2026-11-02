"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    # We will construct a rank-30 decomposition for 2x2x10.
    # The tensor is M_{2,2,10}.
    # Indices:
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (20, rank)
    # Output index convention: W[k*n + i, r]
    # Input A index: i (row 0..1), j (col 0..1) -> U index i*m + j
    # Input B index: j (row 0..1), k (col 0..9) -> V index j*p + k
    # Output C index: i (row 0..1), k (col 0..9) -> W index k*n + i
    
    n, m, p = DIMENSIONS
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each output column k, we have a 2x2x1 tensor.
    # Let's use the rank-3 decomposition for 2x2x1.
    # A = [[a00, a01], [a10, a11]]
    # B = [[b0], [b1]]
    # C = [[c00], [c10]]
    # c00 = a00*b0 + a01*b1
    # c10 = a10*b0 + a11*b1
    
    # Rank 3 decomposition:
    # M1 = a00 * b0
    # M2 = a11 * b1
    # M3 = (a01 + a10) * (b0 + b1)  -> a01*b0 + a01*b1 + a10*b0 + a10*b1
    # This doesn't directly give c00 and c10.
    # Wait, c00 needs a01*b1, not a01*b0.
    # Let's use the standard Strassen-like or known rank-3 for 2x2x1.
    # Actually, R(2,2,1) = 3.
    # Let's find the explicit terms.
    # We want to compute:
    # C_0 = A_{00}B_0 + A_{01}B_1
    # C_1 = A_{10}B_0 + A_{11}B_1
    
    # Try:
    # M1 = A_{00} B_0
    # M2 = A_{11} B_1
    # M3 = (A_{00} + A_{11}) (B_0 + B_1) - M1 - M2 ? No, that's not a single multiplication.
    
    # Let's look up R(2,2,1) decomposition.
    # It is known that 2x2x1 has rank 3.
    # Terms:
    # 1: A_{00} B_0
    # 2: A_{11} B_1
    # 3: (A_{01} + A_{10}) (B_0 + B_1) ??
    # Let's check if (A_{01} + A_{10})(B_0 + B_1) can be combined with others.
    # (A_{01} + A_{10})(B_0 + B_1) = A_{01}B_0 + A_{01}B_1 + A_{10}B_0 + A_{10}B_1
    # We need:
    # C_0 = A_{00}B_0 + A_{01}B_1
    # C_1 = A_{10}B_0 + A_{11}B_1
    
    # If we have M1=A_{00}B_0, M2=A_{11}B_1, M3=(A_{01}+A_{10})(B_0+B_1),
    # C_0 = M1 + (M3 - A_{10}B_0 - A_{01}B_0 - A_{10}B_1) ? No.
    
    # Correct rank-3 decomposition for 2x2x1:
    # M1 = A_{00} B_0
    # M2 = A_{01} B_1
    # M3 = (A_{10} + A_{11}) (B_0 + B_1) ??
    # Let's try:
    # M1 = A_{00} B_0
    # M2 = A_{01} B_1
    # M3 = (A_{10} + A_{00}) (B_0 + B_1) ?
    
    # Actually, let's just use the 4-term (schoolbook) for 2x2x1 to be safe?
    # Rank 4 * 10 = 40. The seed is 40.
    # Can we do better?
    # R(2,2,1) is indeed 3.
    # The decomposition is:
    # M1 = A_{00} B_0
    # M2 = A_{11} B_1
    # M3 = (A_{01} + A_{10}) (B_0 + B_1)
    # Then:
    # C_0 = A_{00}B_0 + A_{01}B_1 = M1 + A_{01}B_1
    # C_1 = A_{10}B_0 + A_{11}B_1 = A_{10}B_0 + M2
    # This requires A_{01}B_1 and A_{10}B_0.
    # From M3, we have A_{01}B_0 + A_{01}B_1 + A_{10}B_0 + A_{10}B_1.
    # This doesn't isolate A_{01}B_1 or A_{10}B_0.
    
    # Let's try another set:
    # M1 = A_{00} B_0
    # M2 = A_{01} B_1
    # M3 = (A_{10} + A_{01}) (B_0 + B_1)
    # M3 = A_{10}B_0 + A_{10}B_1 + A_{01}B_0 + A_{01}B_1
    # C_1 = A_{10}B_0 + A_{11}B_1.
    
    # Actually, the known rank 3 for 2x2x1 is:
    # M1 = A_{00} B_0
    # M2 = A_{11} B_1
    # M3 = (A_{00} + A_{11}) (B_0 + B_1)
    # C_0 = M1 + (M3 - A_{11}B_0 - A_{00}B_1 - A_{11}B_1) ? No.
    
    # Let's stick to the 4-term decomposition for each k to ensure correctness and avoid bugs,
    # unless I can verify the 3-term.
    # 4 terms * 10 = 40. Same as seed.
    # Is there a way to share terms between different k?
    # No, because B_{j,k} are independent.
    
    # However, we can use the rank-7 Strassen for 2x2x2 and embed it?
    # No, that would be higher rank.
    
    # Let's try to find the rank-3 decomposition online or derive it.
    # We want to express the tensor:
    # T = A_{00}B_0 C_0 + A_{01}B_1 C_0 + A_{10}B_0 C_1 + A_{11}B_1 C_1
    # as sum of 3 rank-1 tensors.
    # (u1,v1,w1) + (u2,v2,w2) + (u3,v3,w3)
    
    # Let w1 = C_0, w2 = C_1, w3 = C_0 + C_1
    # u1 = A_{00} + A_{01}, v1 = B_0 + B_1
    # u1 v1 = A_{00}B_0 + A_{00}B_1 + A_{01}B_0 + A_{01}B_1
    # u2 = A_{10} + A_{11}, v2 = B_0 + B_1
    # u2 v2 = A_{10}B_0 + A_{10}B_1 + A_{11}B_0 + A_{11}B_1
    
    # Sum w1 u1 v1 + w2 u2 v2 = C_0(A_{00}B_0 + ... + A_{01}B_1) + C_1(A_{10}B_0 + ... + A_{11}B_1)
    # We need: C_0(A_{00}B_0 + A_{01}B_1) + C_1(A_{10}B_0 + A_{11}B_1)
    # The cross terms are A_{00}B_1, A_{01}B_0, A_{10}B_1, A_{11}B_0.
    # We can use a third term to cancel these.
    # Let u3 = A_{00} - A_{11}, v3 = B_1 - B_0, w3 = C_0 - C_1 ?
    # u3 v3 = (A_{00} - A_{11})(B_1 - B_0) = A_{00}B_1 - A_{00}B_0 - A_{11}B_1 + A_{11}B_0
    # This cancels A_{00}B_1, A_{11}B_0, but introduces -A_{00}B_0 - A_{11}B_1.
    
    # Let's try a specific known decomposition:
    # M1 = A_{00} B_0
    # M2 = A_{11} B_1
    # M3 = (A_{01} + A_{10}) (B_0 + B_1)
    # C_0 = M1 + M3 - A_{10}B_0 - A_{01}B_0 - A_{10}B_1 ?
    # This is getting complicated.
    
    # Given the time limit and risk of error, I will implement the 4-term (schoolbook) per k.
    # Rank = 40.
    # Wait, the seed is 40. If I return 40, score is 1.0.
    # I need to beat 40.
    
    # Let's use the rank-3 decomposition for 2x2x1:
    # M1 = A_{00} B_0
    # M2 = A_{01} B_1
    # M3 = (A_{10} + A_{00}) (B_0 + B_1) ??
    
    # Correct Rank 3 for 2x2x1:
    # 1. A_{00} B_0
    # 2. A_{01} B_1
    # 3. (A_{10} + A_{11}) (B_0 + B_1) - A_{11}B_1 - A_{10}B_0 ??
    
    # Actually, R(2,2,1) = 3.
    # Terms:
    # M1 = A_{00} B_0
    # M2 = A_{11} B_1
    # M3 = (A_{00} + A_{11}) (B_0 + B_1)
    # C_0 = M1 + (M3 - A_{11}B_0 - A_{00}B_1 - A_{11}B_1) ?
    
    # Let's just use the 4-term. It's safer.
    # But I must improve.
    # Is it possible that R(2,2,10) < 40?
    # Yes, R(2,2,1) = 3. So 3*10 = 30.
    
    # Decomposition for 2x2x1:
    # M1 = A_{00} B_0
    # M2 = A_{11} B_1
    # M3 = (A_{01} + A_{10}) (B_0 + B_1)
    # C_0 = M1 + A_{01}B_1
    # C_1 = M2 + A_{10}B_0
    # From M3, A_{01}B_1 + A_{10}B_0 = M3 - A_{01}B_0 - A_{10}B_1
    # This doesn't isolate them.
    
    # Correct decomposition:
    # M1 = A_{00} B_0
    # M2 = A_{01} B_1
    # M3 = (A_{10} + A_{00}) (B_0 + B_1)
    # C_1 = A_{10}B_0 + A_{11}B_1
    # C_0 = A_{00}B_0 + A_{01}B_1
    
    # Let's try:
    # M1 = A_{00} B_0
    # M2 = A_{01} B_1
    # M3 = (A_{10} + A_{11}) (B_0 + B_1)
    # C_1 = M3 - A_{11}B_1 - A_{10}B_1 + A_{11}B_1 = M3 - A_{10}B_1
    # No.
    
    # I will use the 4-term schoolbook to be safe, as I cannot verify the 3-term in time.
    # Rank 40.
    rank = 40
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
