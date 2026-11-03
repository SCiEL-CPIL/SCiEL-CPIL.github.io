"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 3 groups of 2 columns
    for g in range(3):
        k1 = 2 * g
        k2 = 2 * g + 1
        
        # Part 1: Strassen 2x2x2
        # A_left: rows 0,1; cols 0,1
        # B_top: rows 0,1; cols k1, k2
        
        # M1 = (A00+A11)(B0k1+B0k2) ... wait, Strassen uses specific combos.
        # A = [[A00, A01], [A10, A11]]
        # B = [[B0k1, B0k2], [B1k1, B1k2]]
        
        # M1 = (A00+A11) * (B0k1+B1k2) ? No.
        # Standard Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # a=A00, b=A01, c=A10, d=A11
        # e=B0k1, f=B0k2, g=B1k1, h=B1k2
        
        # We need to add these to U, V, W.
        # U indices: i*m+j
        # V indices: j*p+k
        # W indices: k*n+i
        
        def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
            nonlocal col
            U[u_idx][col] = u_val
            V[v_idx][col] = v_val
            W[w_idx][col] = w_val
            col += 1

        # M1
        # (A00+A11) * (B0k1+B1k2)
        # U: A00 (0,0), A11 (1,1)
        # V: B0k1 (0,k1), B1k2 (1,k2)
        # W: C0k1 (k1,0), C1k2 (k2,1)
        # This is not a single rank-1 term. It's a product of sums.
        # We need to expand or use the tensor structure.
        # The term is (A00+A11) * (B0k1+B1k2).
        # This contributes to C0k1, C1k2, etc?
        # In Strassen, M1 is a scalar.
        # C00 += M1, C11 += M1, etc.
        # So we need to add M1 to the appropriate W components.
        
        # But U and V must be linear in A and B.
        # U for M1 has A00=1, A11=1.
        # V for M1 has B0k1=1, B1k2=1.
        # W for M1 has C0k1=1, C1k2=1?
        # No, W is the output coefficient.
        # The product U*V contributes to W.
        # If U has A00=1, A11=1 and V has B0k1=1, B1k2=1,
        # The product is (A00+A11)(B0k1+B1k2).
        # This scalar is added to W.
        # W indices are k*n+i.
        # M1 contributes to C0k1 and C1k2?
        # Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for M1, W has 1 at C0k1 (k1,0) and 1 at C1k2 (k2,1).
        
        # Let's implement a helper to add a Strassen term.
        def add_strassen_term(a_indices, b_indices, w_indices, sign=1):
            nonlocal col
            # a_indices: list of (i, j)
            # b_indices: list of (j, k)
            # w_indices: list of (k, i)
            for i, j in a_indices:
                U[i * m + j][col] = sign
            for j, k in b_indices:
                V[j * p + k][col] = sign
            for k, i in w_indices:
                W[k * n + i][col] = sign
            col += 1

        # M1: (A00+A11)(B0k1+B1k2) -> C0k1, C1k2
        # Wait, B indices in Strassen are e,f,g,h.
        # e=B0k1, f=B0k2, g=B1k1, h=B1k2.
        # M1 = (a+d)(e+h) = (A00+A11)(B0k1+B1k2).
        # W: C0k1 (k1,0), C1k2 (k2,1).
        add_strassen_term([(0,0), (1,1)], [(0,k1), (1,k2)], [(k1,0), (k2,1)])
        
        # M2: (c+d)e = (A10+A11)B0k1 -> C1k1
        # W: C1k1 (k1,1).
        add_strassen_term([(1,0), (1,1)], [(0,k1)], [(k1,1)])
        
        # M3: a(f-h) = A00(B0k2-B1k2) -> C0k2
        # W: C0k2 (k2,0).
        add_strassen_term([(0,0)], [(0,k2), (1,k2)], [(k2,0)], sign=1)
        # Note: V has B0k2=1, B1k2=-1.
        # My helper sets sign for all.
        # I need to handle signs in V.
        
        # Let's rewrite add_term to accept explicit values.
        def add_term_explicit(u_vals, v_vals, w_vals):
            nonlocal col
            for idx, val in u_vals:
                U[idx][col] += val
            for idx, val in v_vals:
                V[idx][col] += val
            for idx, val in w_vals:
                W[idx][col] += val
            col += 1

        # Reset col to 0? No, we are in a loop.
        # I should define the terms explicitly.
        
        # M1
        add_term_explicit(
            [(0*m+0, 1), (1*m+1, 1)],
            [(0*p+k1, 1), (1*p+k2, 1)],
            [(k1*n+0, 1), (k2*n+1, 1)]
        )
        
        # M2
        add_term_explicit(
            [(1*m+0, 1), (1*m+1, 1)],
            [(0*p+k1, 1)],
            [(k1*n+1, 1)]
        )
        
        # M3
        add_term_explicit(
            [(0*m+0, 1)],
            [(0*p+k2, 1), (1*p+k2, -1)],
            [(k2*n+0, 1)]
        )
        
        # M4
        add_term_explicit(
            [(1*m+1, 1)],
            [(1*p+k1, 1), (0*p+k1, -1)],
            [(k1*n+0, 1), (k1*n+1, 1)]
        )
        
        # M5
        add_term_explicit(
            [(0*m+0, 1), (0*m+1, 1)],
            [(1*p+k2, 1)],
            [(k2*n+0, 1), (k2*n+1, 1)]
        )
        
        # M6
        add_term_explicit(
            [(1*m+0, 1), (0*m+0, -1)],
            [(0*p+k1, 1), (0*p+k2, 1)],
            [(k2*n+1, 1)]
        )
        
        # M7
        add_term_explicit(
            [(0*m+1, 1), (1*m+1, -1)],
            [(1*p+k1, 1), (1*p+k2, 1)],
            [(k1*n+0, 1), (k2*n+1, 1)]
        )
        
        # Part 2: 2x1x2
        # A_right: col 2. A02, A12.
        # B_bot: row 2. B2k1, B2k2.
        
        # T1: A02 * B2k1 -> C0k1
        add_term_explicit(
            [(0*m+2, 1)],
            [(2*p+k1, 1)],
            [(k1*n+0, 1)]
        )
        
        # T2: A02 * B2k2 -> C0k2
        add_term_explicit(
            [(0*m+2, 1)],
            [(2*p+k2, 1)],
            [(k2*n+0, 1)]
        )
        
        # T3: A12 * B2k1 -> C1k1
        add_term_explicit(
            [(1*m+2, 1)],
            [(2*p+k1, 1)],
            [(k1*n+1, 1)]
        )
        
        # T4: A12 * B2k2 -> C1k2
        add_term_explicit(
            [(1*m+2, 1)],
            [(2*p+k2, 1)],
            [(k2*n+1, 1)]
        )

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
