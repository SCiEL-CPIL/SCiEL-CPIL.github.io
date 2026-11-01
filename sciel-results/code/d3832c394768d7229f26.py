"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 decomposition using Strassen for 5 blocks
    rank = 35
    
    # U shape (n*m, rank) = (4, 35)
    # V shape (m*p, rank) = (20, 35)
    # W shape (p*n, rank) = (20, 35)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions for a single 2x2x2 block
    # Each term is a tuple of (u_indices, u_vals, v_indices_offset, v_vals, w_indices_offset, w_vals)
    # Indices are relative to the block's local indices or global?
    # We will compute global indices inside the loop.
    
    # Local indices for A (U): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for B (V) part of block t: 
    #   j=0, k=0 -> global 10*0 + 2t = 2t
    #   j=0, k=1 -> global 10*0 + 2t+1 = 2t+1
    #   j=1, k=0 -> global 10*1 + 2t = 10+2t
    #   j=1, k=1 -> global 10*1 + 2t+1 = 11+2t
    # Local indices for C (W) part of block t:
    #   k=0, i=0 -> global 2*2t + 0 = 4t
    #   k=0, i=1 -> global 2*2t + 1 = 4t+1
    #   k=1, i=0 -> global 2*(2t+1) + 0 = 4t+2
    #   k=1, i=1 -> global 2*(2t+1) + 1 = 4t+3
    
    for t in range(5):
        col_base = t * 7
        
        # Term 1: P1 = (a+d)(e+h)
        col = col_base
        U[0][col] = 1; U[3][col] = 1
        V[2*t][col] = 1; V[11+2*t][col] = 1
        W[4*t][col] = 1; W[4*t+3][col] = 1
        
        # Term 2: P2 = (c+d)e
        col = col_base + 1
        U[2][col] = 1; U[3][col] = 1
        V[2*t][col] = 1
        W[4*t+1][col] = 1; W[4*t+3][col] = -1
        
        # Term 3: P3 = a(f-h)
        col = col_base + 2
        U[0][col] = 1
        V[2*t+1][col] = 1; V[11+2*t][col] = -1
        W[4*t+2][col] = 1; W[4*t+3][col] = 1
        
        # Term 4: P4 = d(g-e)
        col = col_base + 3
        U[3][col] = 1
        V[10+2*t][col] = 1; V[2*t][col] = -1
        W[4*t][col] = 1; W[4*t+1][col] = 1
        
        # Term 5: P5 = (a+b)h
        col = col_base + 4
        U[0][col] = 1; U[1][col] = 1
        V[11+2*t][col] = 1
        W[4*t][col] = -1; W[4*t+2][col] = 1
        
        # Term 6: P6 = (c-a)(e+f)
        col = col_base + 5
        U[2][col] = 1; U[0][col] = -1
        V[2*t][col] = 1; V[2*t+1][col] = 1
        W[4*t+3][col] = 1
        
        # Term 7: P7 = (b-d)(g+h)
        col = col_base + 6
        U[1][col] = 1; U[3][col] = -1
        V[10+2*t][col] = 1; V[11+2*t][col] = 1
        W[4*t][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
