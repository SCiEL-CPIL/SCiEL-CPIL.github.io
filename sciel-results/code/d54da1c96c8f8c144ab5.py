"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # The tensor M_{2,3,3} has entries T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # We want T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # Known: rank >= 16 (lower bound from tensor rank theory for this size)
    # Let's try to find rank 16.
    
    # Strategy: The 2x2x2 core (j in {0,1}, k in {0,1}) has rank 7 (Strassen).
    # The 10 boundary terms need to be handled. If we can merge one boundary term
    # with a core term, we get rank 16.
    
    # Let me try a direct construction for rank 16.
    
    # Idea: Use a modified Strassen where one of the 7 core terms is modified
    # to also handle one boundary entry.
    
    # Actually, let me try a completely different approach:
    # Use the fact that we can write the 2x3 * 3x3 multiplication as:
    # Split A into [A_left (2x2) | A_right (2x1)]
    # Split B into [[B_top (2x3)], [B_bottom (1x3)]]
    # Then C = A_left * B_top + A_right * B_bottom
    # A_left * B_top is 2x2 * 2x3, which can be done with rank 7 (Strassen for 2x2*2x2 gives 4 outputs, 
    #   but we need 2x3 outputs, so we need more)
    # Actually, 2x2 * 2x3: C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] for k=0,1,2
    # This is 6 output entries from 4*2 = 8 input pairs.
    
    # Let me just try to construct rank 16 directly.
    
    # The 18 entries are (i,j,k) for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    
    # Let me try: 7 Strassen terms for the j in {0,1}, k in {0,1} part (8 entries),
    # and 9 terms for the remaining 10 entries, with one term handling 2 entries.
    
    # Can one rank-1 term handle 2 entries? Only if they share the same i, j, or k pattern
    # in a way that allows factoring.
    
    # Entry (0,0,2) and (1,0,2): both have j=0, k=2. 
    # u would need to have entries at (0,0) and (1,0), v at (0,2), w at (2,0) and (2,1).
    # But a rank-1 term u⊗v⊗w has u[i,j]*v[j,k]*w[k,i] for each (i,j,k).
    # For (0,0,2): u[0,0]*v[0,2]*w[2,0]
    # For (1,0,2): u[1,0]*v[0,2]*w[2,1]
    # These are different (i,j,k) triples, so a single rank-1 term can only contribute
    # to one specific (i,j,k) triple (with value u[i,j]*v[j,k]*w[k,i]).
    # Wait, no! A rank-1 term contributes to ALL (i,j,k) where u[i,j], v[j,k], w[k,i] are nonzero.
    # So if u has entries at (0,0) and (1,0), v has entry at (0,2), w has entries at (2,0) and (2,1),
    # then the term contributes to:
    # (0,0,2): u[0,0]*v[0,2]*w[2,0]
    # (1,0,2): u[1,0]*v[0,2]*w[2,1]
    # (0,0,?) : no, w is only at (2,0) and (2,1), so k must be 2.
    # Actually, w[k,i] is nonzero for (k,i) = (2,0) and (2,1). So i can be 0 or 1, k=2.
    # u[i,j] nonzero for (i,j) = (0,0) and (1,0). So i in {0,1}, j=0.
    # v[j,k] nonzero for (j,k) = (0,2). So j=0, k=2.
    # Combined: i in {0,1}, j=0, k=2. So it contributes to (0,0,2) and (1,0,2).
    # Yes! One rank-1 term can handle both!
    
    # So: term with u[0,0]=a, u[1,0]=b, v[0,2]=c, w[2,0]=d, w[2,1]=e
    # contributes a*c*d to (0,0,2) and b*c*e to (1,0,2).
    # To get both equal to 1: a*c*d = 1, b*c*e = 1.
    # We can set c=1, a=1, d=1, b=1, e=1.
    
    # Great! So we can merge (0,0,2) and (1,0,2) into one term.
    # Similarly, can we merge (0,1,2) and (1,1,2)?
    # u[0,1]=a, u[1,1]=b, v[1,2]=c, w[2,0]=d, w[2,1]=e
    # contributes a*c*d to (0,1,2) and b*c*e to (1,1,2).
    # Yes!
    
    # And (0,2,0) and (1,2,0)?
    # u[0,2]=a, u[1,2]=b, v[2,0]=c, w[0,0]=d, w[0,1]=e
    # contributes a*c*d to (0,2,0) and b*c*e to (1,2,0).
    # Yes!
    
    # And (0,2,1) and (1,2,1)?
    # u[0,2]=a, u[1,2]=b, v[2,1]=c, w[1,0]=d, w[1,1]=e
    # Yes!
    
    # And (0,2,2) and (1,2,2)?
    # u[0,2]=a, u[1,2]=b, v[2,2]=c, w[2,0]=d, w[2,1]=e
    # Yes!
    
    # So all 10 boundary entries can be handled by 5 rank-1 terms!
    # Total: 7 (core) + 5 (boundary) = 12? That seems too good.
    
    # Wait, but the core Strassen terms also contribute to some of these entries!
    # Let me check: the core terms only involve j in {0,1} and k in {0,1}.
    # So they don't contribute to any entry with j=2 or k=2.
    # So the 7 core terms handle the 8 entries with j in {0,1}, k in {0,1}.
    # The 5 boundary terms handle the 10 entries with j=2 or k=2.
    # Total: 12 terms!
    
    # But wait, can the 5 boundary terms interfere with the core entries?
    # Boundary term 1: u at (0,0),(1,0); v at (0,2); w at (2,0),(2,1)
    # This only has k=2, so it doesn't affect core entries (which have k in {0,1}).
    # Boundary term 2: u at (0,1),(1,1); v at (1,2); w at (2,0),(2,1)
    # k=2 only. Good.
    # Boundary term 3: u at (0,2),(1,2); v at (2,0); w at (0,0),(0,1)
    # j=2 only. Good.
    # Boundary term 4: u at (0,2),(1,2); v at (2,1); w at (1,0),(1,1)
    # j=2 only. Good.
    # Boundary term 5: u at (0,2),(1,2); v at (2,2); w at (2,0),(2,1)
    # j=2 and k=2. Good.
    
    # So total rank = 12? Let me verify this more carefully.
    
    # Actually, I need to double-check. Let me construct this and verify.
    
    rank = 12
    U = [[0] * rank for _ in range(n * m)]  # (6, 12)
    V = [[0] * rank for _ in range(m * p)]  # (9, 12)
    W = [[0] * rank for _ in range(p * n)]  # (6, 12)
    
    def set_u(i, j, col, val):
        U[i * m + j][col] = val
        
    def set_v(j, k, col, val):
        V[j * p + k][col] = val
        
    def set_w(k, i, col, val):
        W[k * n + i][col] = val
    
    # Core: 7 Strassen terms for j in {0,1}, k in {0,1}
    # M1 = (A00+A11)(B00+B11)
    col = 0
    set_u(0, 0, col, 1)
    set_u(1, 1, col, 1)
    set_v(0, 0, col, 1)
    set_v(1, 1, col, 1)
    set_w(0, 0, col, 1)
    set_w(1, 1, col, 1)
    
    # M2 = (A10+A11)B00
    col = 1
    set_u(1, 0, col, 1)
    set_u(1, 1, col, 1)
    set_v(0, 0, col, 1)
    set_w(0, 1, col, 1)
    set_w(1, 1, col, -1)
    
    # M3 = A00(B01-B11)
    col = 2
    set_u(0, 0, col, 1)
    set_v(0, 1, col, 1)
    set_v(1, 1, col, -1)
    set_w(1, 0, col, 1)
    set_w(1, 1, col, 1)
    
    # M4 = A11(B10-B00)
    col = 3
    set_u(1, 1, col, 1)
    set_v(1, 0, col, 1)
    set_v(0, 0, col, -1)
    set_w(0, 0, col, 1)
    set_w(0, 1, col, 1)
    
    # M5 = (A00+A01)B11
    col = 4
    set_u(0, 0, col, 1)
    set_u(0, 1, col, 1)
    set_v(1, 1, col, 1)
    set_w(0, 0, col, -1)
    set_w(1, 0, col, 1)
    
    # M6 = (A10-A00)(B00+B01)
    col = 5
    set_u(1, 0, col, 1)
    set_u(0, 0, col, -1)
    set_v(0, 0, col, 1)
    set_v(0, 1, col, 1)
    set_w(1, 1, col, 1)
    
    # M7 = (A01-A11)(B10+B11)
    col = 6
    set_u(0, 1, col, 1)
    set_u(1, 1, col, -1)
    set_v(1, 0, col, 1)
    set_v(1, 1, col, 1)
    set_w(0, 0, col, 1)
    
    # Boundary term 1: handles (0,0,2) and (1,0,2)
    col = 7
    set_u(0, 0, col, 1)
    set_u(1, 0, col, 1)
    set_v(0, 2, col, 1)
    set_w(2, 0, col, 1)
    set_w(2, 1, col, 1)
    
    # Boundary term 2: handles (0,1,2) and (1,1,2)
    col = 8
    set_u(0, 1, col, 1)
    set_u(1, 1, col, 1)
    set_v(1, 2, col, 1)
    set_w(2, 0, col, 1)
    set_w(2, 1, col, 1)
    
    # Boundary term 3: handles (0,2,0) and (1,2,0)
    col = 9
    set_u(0, 2, col, 1)
    set_u(1, 2, col, 1)
    set_v(2, 0, col, 1)
    set_w(0, 0, col, 1)
    set_w(0, 1, col, 1)
    
    # Boundary term 4: handles (0,2,1) and (1,2,1)
    col = 10
    set_u(0, 2, col, 1)
    set_u(1, 2, col, 1)
    set_v(2, 1, col, 1)
    set_w(1, 0, col, 1)
    set_w(1, 1, col, 1)
    
    # Boundary term 5: handles (0,2,2) and (1,2,2)
    col = 11
    set_u(0, 2, col, 1)
    set_u(1, 2, col, 1)
    set_v(2, 2, col, 1)
    set_w(2, 0, col, 1)
    set_w(2, 1, col, 1)
    
    # Now verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1) > max_err:
                    max_err = abs(val - 1)
    
    if max_err < 1e-10:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to rank 17
        rank = 17
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def set_u(i, j, col, val):
            U[i * m + j][col] = val
            
        def set_v(j, k, col, val):
            V[j * p + k][col] = val
            
        def set_w(k, i, col, val):
            W[k * n + i][col] = val

        col = 0
        set_u(0, 0, col, 1); set_u(1, 1, col, 1)
        set_v(0, 0, col, 1); set_v(1, 1, col, 1)
        set_w(0, 0, col, 1); set_w(1, 1, col, 1)

        col = 1
        set_u(1, 0, col, 1); set_u(1, 1, col, 1)
        set_v(0, 0, col, 1)
        set_w(0, 1, col, 1); set_w(1, 1, col, -1)

        col = 2
        set_u(0, 0, col, 1)
        set_v(0, 1, col, 1); set_v(1, 1, col, -1)
        set_w(1, 0, col, 1); set_w(1, 1, col, 1)

        col = 3
        set_u(1, 1, col, 1)
        set_v(1, 0, col, 1); set_v(0, 0, col, -1)
        set_w(0, 0, col, 1); set_w(0, 1, col, 1)

        col = 4
        set_u(0, 0, col, 1); set_u(0, 1, col, 1)
        set_v(1, 1, col, 1)
        set_w(0, 0, col, -1); set_w(1, 0, col, 1)

        col = 5
        set_u(1, 0, col, 1); set_u(0, 0, col, -1)
        set_v(0, 0, col, 1); set_v(0, 1, col, 1)
        set_w(1, 1, col, 1)

        col = 6
        set_u(0, 1, col, 1); set_u(1, 1, col, -1)
        set_v(1, 0, col, 1); set_v(1, 1, col, 1)
        set_w(0, 0, col, 1)

        col = 7; set_u(0, 0, col, 1); set_v(0, 2, col, 1); set_w(2, 0, col, 1)
        col = 8; set_u(0, 1, col, 1); set_v(1, 2, col, 1); set_w(2, 0, col, 1)
        col = 9; set_u(1, 0, col, 1); set_v(0, 2, col, 1); set_w(2, 1, col, 1)
        col = 10; set_u(1, 1, col, 1); set_v(1, 2, col, 1); set_w(2, 1, col, 1)
        col = 11; set_u(0, 2, col, 1); set_v(2, 0, col, 1); set_w(0, 0, col, 1)
        col = 12; set_u(0, 2, col, 1); set_v(2, 1, col, 1); set_w(1, 0, col, 1)
        col = 13; set_u(0, 2, col, 1); set_v(2, 2, col, 1); set_w(2, 0, col, 1)
        col = 14; set_u(1, 2, col, 1); set_v(2, 0, col, 1); set_w(0, 1, col, 1)
        col = 15; set_u(1, 2, col, 1); set_v(2, 1, col, 1); set_w(1, 1, col, 1)
        col = 16; set_u(1, 2, col, 1); set_v(2, 2, col, 1); set_w(2, 1, col, 1)

        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
