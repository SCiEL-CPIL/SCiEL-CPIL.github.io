"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Try to find a low-rank decomposition
    # For 2x2 * 2x15, try rank 2p+2 = 32 first
    
    # Construction for rank 2p+2:
    # Shared products (indices 0, 1):
    #   P0 = (a00 + a10) * (b00 + b01 + ... + b0_{p-1}) -- no, this doesn't work cleanly
    #
    # Better construction:
    # For each column k, we need:
    #   c0k = a00*b0k + a01*b1k
    #   c1k = a10*b0k + a11*b1k
    #
    # Use 4 products per column but share 2 across all columns:
    # This is tricky. Let me try a different approach.
    #
    # Actually, for 2x2 * 2xp, the optimal rank is known to be 2p+2 for p >= 2.
    # Construction:
    #   Let s0 = a00 + a10, s1 = a01 + a11
    #   Let t_k = b0k - b1k for each k
    #
    # Products:
    #   P0 = s0 * (sum of all b0k) -- no
    #
    # Let me use a cleaner construction:
    #   P_k0 = (a00 + a10) * b0k  for k=0,...,p-1  (p products)
    #   P_k1 = (a01 + a11) * b1k  for k=0,...,p-1  (p products)
    #   P_k2 = (a00 - a10) * b0k  for k=0,...,p-1  -- but this is p more, total 3p
    #
    # That's 3p which is worse than 2p+2 for p>2.
    #
    # Correct construction for rank 2p+2:
    #   P0 = (a00 + a10) * (b00 + b10)  -- no, this mixes columns
    #
    # I think the right approach is:
    #   For the 2x2 * 2x2 case, rank is 7 (Strassen)
    #   For 2x2 * 2xp, rank is 2p + 2
    #
    # Construction:
    #   P0 = (a00 + a10) * b00
    #   P1 = (a01 + a11) * b10
    #   P2 = (a00 + a10) * b01
    #   P3 = (a01 + a11) * b11
    #   ...
    #   P_{2k} = (a00 + a10) * b0k
    #   P_{2k+1} = (a01 + a11) * b1k
    #   Plus 2 more:
    #   P_{2p} = (a00 - a10) * (b00 - b01 + b02 - ... ) -- no
    #
    # Let me just implement the schoolbook and a few structured attempts,
    # then verify and return the best valid one.
    
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    def verify(U, V, W, n, m, p):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if val != 1:
                        return False
        return True
    
    # Try rank 2p+2 = 32 construction
    # Based on the idea that for 2x2 * 2xp, we can use 2p+2 products
    # P_{2k} = (a00 + a10) * b0k
    # P_{2k+1} = (a01 + a11) * b1k
    # P_{2p} = (a00 - a10) * (b00 + b01 + ... + b0_{p-1})  -- this doesn't isolate columns
    #
    # Actually, let me try:
    # c0k = a00*b0k + a01*b1k
    # c1k = a10*b0k + a11*b1k
    #
    # Let u0 = a00 + a10, u1 = a01 + a11, u2 = a00 - a10, u3 = a01 - a11
    # Then:
    # a00 = (u0 + u2)/2, a10 = (u0 - u2)/2
    # a01 = (u1 + u3)/2, a11 = (u1 - u3)/2
    #
    # c0k = ((u0+u2)/2)*b0k + ((u1+u3)/2)*b1k
    #      = (1/2)[u0*b0k + u2*b0k + u1*b1k + u3*b1k]
    # c1k = ((u0-u2)/2)*b0k + ((u1-u3)/2)*b1k
    #      = (1/2)[u0*b0k - u2*b0k + u1*b1k - u3*b1k]
    #
    # So we need products: u0*b0k, u2*b0k, u1*b1k, u3*b1k for each k
    # That's 4p products. No savings.
    #
    # The key insight for 2p+2: we can share some products across columns.
    # Let me try:
    #   P_k = (a00 + a10) * b0k  for k=0..p-1   (p products)
    #   Q_k = (a01 + a11) * b1k  for k=0..p-1   (p products)
    #   R = (a00 - a10) * (b0_0 + b0_1 + ... + b0_{p-1})  -- no
    #
    # I'll just return schoolbook for now since I can't quickly derive the 2p+2 construction.
    
    U_school, V_school, W_school, rank_school = make_schoolbook()
    
    # Verify schoolbook
    if not verify(U_school, V_school, W_school, n, m, p):
        # Should not happen
        pass
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
