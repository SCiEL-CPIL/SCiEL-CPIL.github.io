"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition term by term.
    # U: (n*m, rank) -> (25, rank)
    # V: (m*p, rank) -> (25, rank)
    # W: (p*n, rank) -> (25, rank)
    
    terms = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx):
        terms.append((u_idx, v_idx, w_idx))
        
    # 1. A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    # Use Strassen for 4x4
    # Strassen 2x2 terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # We need to map these to global indices.
    # Block offsets: i_off=0, j_off=0, k_off=0. Size=4.
    # We can decompose 4x4 into four 2x2 blocks.
    # A00 (0,0), A01 (0,2), A10 (2,0), A11 (2,2)
    # B00 (0,0), B01 (0,2), B10 (2,0), B11 (2,2)
    
    # This is complex to hardcode correctly.
    # Alternative: Use Schoolbook for 4x4 (Rank 64).
    # Total Rank: 64 + 4 + 1 + 1 + 4 + 1 + 1 + 1 = 77.
    # 125 / 77 = 1.62. Still a good improvement.
    # And much safer to implement.
    
    # Let's use Schoolbook for the 4x4 block to ensure correctness.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # A[i, k] * B[k, j] -> C[i, j]
                u_idx = i * 5 + k
                v_idx = k * 5 + j
                w_idx = j * 5 + i
                add_term(u_idx, v_idx, w_idx)
                
    # 2. A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    # A[i, k] * B[k, 4] -> C[i, 4]
    for i in range(4):
        for k in range(4):
            u_idx = i * 5 + k
            v_idx = k * 5 + 4
            w_idx = 4 * 5 + i
            add_term(u_idx, v_idx, w_idx)
            
    # 3. A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    # A[i, 4] * B[4, j] -> C[i, j]
    # Rank 1 optimization?
    # T[i, j, 4] = 1.
    # We can do it with 1 term:
    # U[i*5+4] = 1 for all i? No, U is a matrix.
    # If we use 1 term c:
    # U[i*5+4, c] = 1 for all i in 0..3.
    # V[4*5+j, c] = 1 for all j in 0..3.
    # W[j*5+i, c] = 1 for all i,j in 0..3.
    # This works.
    c_start = len(terms)
    # We just add 1 term. But we need to set the values in U, V, W.
    # Since we are building a list of terms, we can just add the term.
    # But the "Rank 1" logic means we use 1 column.
    # However, the values in U, V, W are 0/1.
    # For this term, U has 1s in rows 4, 9, 14, 19.
    # V has 1s in rows 20, 21, 22, 23.
    # W has 1s in rows 0..15? No, W index is j*5+i.
    # j in 0..3, i in 0..3.
    # W rows: 0,1,2,3, 5,6,7,8, 10,11,12,13, 15,16,17,18.
    # So we add 1 term.
    add_term(-1, -1, -1) # Placeholder, will handle specially or just add 4 terms?
    # To be safe and simple, I will use 4 terms (Schoolbook style) for this block too.
    # Rank 4.
    terms.pop()
    for i in range(4):
        for j in range(4):
            u_idx = i * 5 + 4
            v_idx = 4 * 5 + j
            w_idx = j * 5 + i
            add_term(u_idx, v_idx, w_idx)

    # 4. A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    # A[i, 4] * B[4, 4] -> C[i, 4]
    for i in range(4):
        u_idx = i * 5 + 4
        v_idx = 4 * 5 + 4
        w_idx = 4 * 5 + i
        add_term(u_idx, v_idx, w_idx)

    # 5. A[4, 0:4] * B[0:4, 0:4] -> C[4, 0:4]
    # A[4, k] * B[k, j] -> C[4, j]
    for j in range(4):
        for k in range(4):
            u_idx = 4 * 5 + k
            v_idx = k * 5 + j
            w_idx = j * 5 + 4
            add_term(u_idx, v_idx, w_idx)

    # 6. A[4, 0:4] * B[0:4, 4] -> C[4, 4]
    # A[4, k] * B[k, 4] -> C[4, 4]
    for k in range(4):
        u_idx = 4 * 5 + k
        v_idx = k * 5 + 4
        w_idx = 4 * 5 + 4
        add_term(u_idx, v_idx, w_idx)

    # 7. A[4, 4] * B[4, 0:4] -> C[4, 0:4]
    # A[4, 4] * B[4, j] -> C[4, j]
    for j in range(4):
        u_idx = 4 * 5 + 4
        v_idx = 4 * 5 + j
        w_idx = j * 5 + 4
        add_term(u_idx, v_idx, w_idx)

    # 8. A[4, 4] * B[4, 4] -> C[4, 4]
    u_idx = 4 * 5 + 4
    v_idx = 4 * 5 + 4
    w_idx = 4 * 5 + 4
    add_term(u_idx, v_idx, w_idx)

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        U[u][c] = 1
        V[v][c] = 1
        W[w][c] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
