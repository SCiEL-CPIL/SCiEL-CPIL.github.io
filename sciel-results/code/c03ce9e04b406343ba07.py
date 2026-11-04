"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Block-structured decomposition approach
    # Split m into m1=2, m2=3
    # Split p into p1=2, p2=3
    m1, m2 = 2, 3
    p1, p2 = 2, 3
    
    # Define blocks: (m_start, m_end, p_start, p_end)
    blocks = [
        (0, m1, 0, p1),  # M_{n, 2, 2}
        (0, m1, p1, p1+p2),  # M_{n, 2, 3}
        (m1, m1+m2, 0, p1),  # M_{n, 3, 2}
        (m1, m1+m2, p1, p1+p2)  # M_{n, 3, 3}
    ]
    
    # Calculate total rank (sum of schoolbook ranks of blocks)
    rank = 0
    for ms, me, ps, pe in blocks:
        rank += n * (me - ms) * (pe - ps)
        
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for ms, me, ps, pe in blocks:
        for i in range(n):
            for j in range(ms, me):
                for k in range(ps, pe):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
