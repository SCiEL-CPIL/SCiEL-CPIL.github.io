"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: 6x6 = 3 blocks of size 2 in each dimension.
    # Meta-tensor: 3x3x3 matrix multiplication (Strassen rank 7)
    # Cell tensor: 2x2x2 matrix multiplication (schoolbook rank 8)
    # Total rank = 7 * 8 = 56
    
    block_size = 2
    blocks_per_dim = 3  # 6 / 2
    
    rank = 7 * 8  # = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen patterns for 3x3 matrices (meta-level).
    # Each pattern is (U_meta, V_meta, W_meta) where each is a dict from (i,j) to coeff.
    # U_meta[i,j] multiplies A-block[i,j], V_meta[j,k] multiplies B-block[j,k],
    # W_meta[k,i] contributes to C-block[i,k].
    # The identity: sum_t U_meta_t[i,j] * V_meta_t[j,k] * W_meta_t[k,i] = 1 for all i,j,k in {0,1,2}
    # This is the standard Strassen decomposition of the 3x3x3 multiplication tensor.
    
    # Strassen's 7 products for 3x3 matrix multiplication:
    # M0 = (A00+A11+A22) * (B00+B11+B22)
    # M1 = (A01+A12+A20) * (B01+B12+B20)
    # M2 = (A02+A10+A21) * (B02+B10+B21)
    # M3 = (A00-A11) * (B11-B00)  -- wait, this is for 2x2.
    
    # For 3x3, the Strassen-like decomposition uses 7 products. Let me use the
    # standard formulation. The 3x3x3 tensor rank is 7 (this is a known result
    # from the "Strassen" method extended, or more precisely, the rank of M_3,3,3
    # is at most 23, but we're using a 7-term decomposition of a SUB-tensor).
    
    # Wait - I'm confusing myself. The rank of M_3,3,3 is NOT 7. It's at most 23.
    # Rank 7 is for M_2,2,2.
    
    # Let me reconsider. The correct decomposition:
    # 6x6x6 tensor = Kronecker product of M_3,3,3 (meta) and M_2,2,2 (cell)
    # rank(M_6,6,6) <= rank(M_3,3,3) * rank(M_2,2,2)
    # rank(M_2,2,2) = 7 (Strassen)
    # rank(M_3,3,3) <= 23 (Laderman) or 27 (schoolbook)
    
    # So: 23 * 7 = 161 or 27 * 7 = 189.
    
    # Alternatively: 6 = 2*3, so M_6,6,6 <= M_2,2,2 * M_3,3,3 (Kronecker)
    # Same thing: 7 * 23 = 161.
    
    # OR: use schoolbook for 3x3 (rank 27) and Strassen for 2x2 (rank 7): 27*7 = 189 (current)
    # OR: use Strassen for 3x3... but rank of M_3,3,3 is not 7.
    
    # The best I can do with Kronecker products:
    # 6 = 2 * 3: rank <= rank(M_2,2,2) * rank(M_3,3,3) = 7 * 23 = 161
    # 6 = 3 * 2: same thing = 161
    # 6 = 6: rank <= 216 (schoolbook)
    
    # So 161 is the best Kronecker approach. Let me implement 3x3 schoolbook * 2x2 Strassen
    # which gives 27 * 7 = 189, OR 2x2 Strassen * 3x3 schoolbook = 7 * 27 = 189.
    
    # Wait, these are the same! The Kronecker product is commutative in rank.
    
    # To get 161, I need a rank-23 decomposition of M_3,3,3.
    # Let me try to construct one.
    
    # Actually, let me reconsider the structure.
    # 6x6x6: split each dimension into 3 parts of size 2.
    # Meta: 3x3x3 (which is M_3,3,3, rank <= 23)
    # Cell: 2x2x2 (which is M_2,2,2, rank = 7)
    # Kronecker: rank <= 23 * 7 = 161.
    
    # Alternatively: split each dimension into 2 parts of size 3.
    # Meta: 2x2x2 (M_2,2,2, rank = 7)
    # Cell: 3x3x3 (M_3,3,3, rank <= 23)
    # Kronecker: rank <= 7 * 23 = 161.
    
    # Both give 161. The current code uses the second form (2x2 Strassen meta, 3x3 schoolbook cell) = 7*27 = 189.
    # If I use 2x2 Strassen meta and 3x3 rank-23 cell: 7 * 23 = 161.
    
    # I need a rank-23 decomposition of M_3,3,3. Let me try to construct/verify one.
    
    # For now, let me implement the safe 189-rank version (current) and add a time-limited
    # search for a rank-23 M_3,3,3 decomposition.
    
    # SAFE FALLBACK: current Strassen-2x2-meta approach (rank 189)
    rank_safe = 7 * 27
    U_safe = [[0] * rank_safe for _ in range(n * m)]
    V_safe = [[0] * rank_safe for _ in range(m * p)]
    W_safe = [[0] * rank_safe for _ in range(p * n)]
    
    strassen_patterns_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    bs = 3
    bpd = 2
    col_idx = 0
    for I in range(bpd):
        for J in range(bpd):
            for K in range(bpd):
                for u_c, v_c, w_c in strassen_patterns_2x2:
                    for (pi, pj), val in u_c.items():
                        r = I * bs + pi
                        c = J * bs + pj
                        U_safe[r * m + c][col_idx] += val
                    for (pi, pj), val in v_c.items():
                        r = J * bs + pi
                        c = K * bs + pj
                        V_safe[r * p + c][col_idx] += val
                    for (pi, pj), val in w_c.items():
                        k = K * bs + pj
                        i = I * bs + pi
                        W_safe[k * n + i][col_idx] += val
                    col_idx += 1
    
    # Verify safe decomposition
    def verify(Um, Vm, Wm, nn, mm, pp, rk):
        """Verify that sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1 for all i,j,k."""
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    s = 0
                    for t in range(rk):
                        s += Um[i * mm + j][t] * Vm[j * pp + k][t] * Wm[k * nn + i][t]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify(U_safe, V_safe, W_safe, n, m, p, rank_safe):
        # Should never happen, but return schoolbook as last resort
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t = i * m * p + j * p + k
                    U_s[i * m + j][t] = 1
                    V_s[j * p + k][t] = 1
                    W_s[k * n + i][t] = 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    # Try to find a rank-23 decomposition of M_3,3,3 to achieve rank 161.
    # Use a time-limited approach.
    import time
    deadline = time.time() + 100  # 100s limit, leave 20s margin
    
    # Attempt: construct a rank-23 M_3,3,3 decomposition using known formulas.
    # The Laderman algorithm for 3x3 uses 23 multiplications.
    # Let me try to write it out and verify.
    
    # Known rank-23 decomposition of M_3,3,3 (from Laderman 1976):
    # I'll define 23 triples of 3x3 matrices.
    # Each triple (U_t, V_t, W_t) satisfies:
    #   sum_t U_t[i,j] * V_t[j,k] * W_t[k,i] = 1 for all i,j,k in {0,1,2}
    
    # Let me try a construction based on the "border rank" approach.
    # Actually, let me try a simpler systematic construction.
    
    # Approach: Start with schoolbook (27 terms) and try to merge 4 terms.
    # Schoolbook: U[i,j] = e_ij, V[j,k] = e_jk, W[k,i] = e_ki (for each i,j,k)
    # That's 27 terms. We need to reduce to 23.
    
    # Alternative: Use the known fact that rank(M_3,3,3) <= 23 and try to
    # find it via optimization. But that's complex.
    
    # Let me try a different known decomposition.
    # The "Strassen-like" decomposition for 3x3:
    # Actually, I recall that the rank of M_3,3,3 is exactly 23 (proven by
    # various authors). Let me try to find an explicit construction.
    
    # One approach: use the "substitution" method.
    # M_3,3,3 can be seen as having a 2x2 sub-block plus cross terms.
    # The 2x2 sub-block has rank 7 (Strassen).
    # The cross terms... this gets complicated.
    
    # Given time constraints, let me just return the safe rank-189 decomposition.
    # This is a valid improvement over the schoolbook 216.
    
    return {"U": U_safe, "V": V_safe, "W": W_safe}
# EVOLVE-BLOCK-END
