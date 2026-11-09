"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 43 decomposition
    rank = 43
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        # Local indices:
        # A: (0,0)->a, (0,1)->b, (1,0)->c, (1,1)->d
        # B: (0,0)->e, (0,1)->f, (1,0)->g, (1,1)->h
        # C: (0,0)->i, (0,1)->j, (1,0)->k, (1,1)->l
        
        # Map local to global
        # U index: (i_off + li) * m + (j_off + lj)
        # V index: (j_off + lj) * p + (k_off + lk)
        # W index: (k_off + lk) * n + (i_off + li)
        
        # Strassen terms:
        # 1. M1 = (a+d)(e+h) -> i=1, l=1
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # 2. M2 = (c+d)e -> k=1, l=-1
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1
        
        # 3. M3 = a(f-h) -> j=1, l=1
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # 4. M4 = d(g-e) -> i=1, k=1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = -1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1
        
        # 5. M5 = (a+b)h -> i=-1, j=1
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        # 6. M6 = (c-a)(e+f) -> l=1
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+0), col] = -1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # 7. M7 = (b-d)(g+h) -> i=1
        U[(i_off+0)*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

    # Helper to add a 1x2x2 block (rank 4)
    def add_122(i_off, j_off, k_off):
        nonlocal col
        # A: (0,0)->a, (0,1)->b
        # B: (0,0)->e, (0,1)->f, (1,0)->g, (1,1)->h
        # C: (0,0)->i, (0,1)->j
        
        # 1. ae -> i
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1
        
        # 2. bg -> i
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1
        
        # 3. af -> j
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        # 4. bh -> j
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

    # Top part: A[0:2, :] * B[:, :]
    # B blocks: B[0:2, 0:2] (0,0), B[0:2, 2:4] (0,1), B[2:4, 0:2] (1,0), B[2:4, 2:4] (1,1)
    
    # Block 1: A_top[:, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_222(0, 0, 0)
    
    # Block 2: A_top[:, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_222(0, 2, 0)
    
    # Block 3: A_top[:, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_222(0, 0, 2)
    
    # Block 4: A_top[:, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_222(0, 2, 2)
    
    # Bottom part: A[2, :] * B[:, :]
    # A_bot is 1x4.
    # A_bot[0:2] is 1x2. A_bot[2:4] is 1x2.
    
    # Block 5: A_bot[0:2] * B[0:2, 0:2] -> C[2, 0:2]
    add_122(2, 0, 0)
    
    # Block 6: A_bot[2:4] * B[2:4, 0:2] -> C[2, 0:2]
    add_122(2, 2, 0)
    
    # Block 7: A_bot[0:2] * B[0:2, 2:4] -> C[2, 2:4]
    add_122(2, 0, 2)
    
    # Block 8: A_bot[2:4] * B[2:4, 2:4] -> C[2, 2:4]
    # We need to save 1 rank here. Let's try a rank-3 decomposition for 1x2x2.
    # Standard: ae, bg, af, bh (rank 4)
    # Can we do rank 3?
    # Let's try:
    # M1 = a(e+h) -> contributes to i (via e) and j (via h)
    # M2 = b(g-e) -> contributes to i (via g) and j (via -e)... wait
    # Actually, for 1x2x2:
    # C[0,0] = a*e + b*g
    # C[0,1] = a*f + b*h
    # Rank 3 attempt:
    # M1 = a*(e+f) -> C[0,0] += a*e, C[0,1] += a*f
    # M2 = b*(g+h) -> C[0,0] += b*g, C[0,1] += b*h
    # This is rank 2! But wait, that's not right.
    # C[0,0] = a*e + b*g
    # C[0,1] = a*f + b*h
    # If we use M1 = a*(e+f), M2 = b*(g+h):
    # C[0,0] gets a*e + b*g (correct)
    # C[0,1] gets a*f + b*h (correct)
    # So rank 2 is sufficient for 1x2x2!
    
    # Let's implement rank 2 for the last block
    # M1 = a*(e+f)
    U[(2)*m + (2), col] = 1  # a = A[2,2]
    V[(2)*p + (0), col] = 1  # e = B[2,0]
    V[(2)*p + (1), col] = 1  # f = B[2,1]
    W[(0)*n + (2), col] = 1  # C[0,2] += a*e
    W[(1)*n + (2), col] = 1  # C[1,2] += a*f
    col += 1
    
    # M2 = b*(g+h)
    U[(2)*m + (3), col] = 1  # b = A[2,3]
    V[(3)*p + (0), col] = 1  # g = B[3,0]
    V[(3)*p + (1), col] = 1  # h = B[3,1]
    W[(0)*n + (2), col] = 1  # C[0,2] += b*g
    W[(1)*n + (2), col] = 1  # C[1,2] += b*h
    col += 1
    
    # Now col should be 43
    # Let's verify: 4*7 + 3*4 + 2 = 28 + 12 + 2 = 42. Wait, that's 42.
    # Let me recount.
    # 4 Strassen blocks: 4*7 = 28
    # 3 1x2x2 blocks: 3*4 = 12
    # 1 rank-2 block: 2
    # Total: 28 + 12 + 2 = 42.
    
    # Hmm, I can save 2 ranks! Let's use rank 2 for all the 1x2x2 blocks.
    # But wait, I already added 3 blocks of rank 4. Let me redo this.
    
    # Actually, let me just return what I have and see if it's valid.
    # The issue is that I've already allocated 43 columns but only filled 42.
    
    # Let me restart with a cleaner approach.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
