import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 16
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # We decompose C = A * B where A is 2x4, B is 4x4, C is 2x4
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Strategy: 16 rank-1 terms
    # Terms 0-7: handle contributions from B rows 0,1 (top half)
    # Terms 8-15: handle contributions from B rows 2,3 (bottom half)
    #
    # For top half: C_top[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a 2x2x4 tensor. We use 8 terms (one per (i,k) pair won't work, 
    # need to be smarter)
    #
    # Actually, let's use a direct construction:
    # For each (i, k) in {0,1} x {0,1,2,3}, C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Let's use a rank-8 decomposition for each half:
    # Top: C_top[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # We can write this with 8 terms using a 2x2x4 decomposition
    #
    # For a 2x2x4 matrix multiplication, the rank is at most 8 (naive) 
    # but can be less. Let's use rank 8 for each half = 16 total.
    #
    # Top half (rows 0,1 of B):
    # C_top[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # 
    # Term structure for top half (terms 0-7):
    # We need: for each i in {0,1}, k in {0,1,2,3}:
    #   C[i,k] gets A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # Let's define 8 terms where each term handles one (i, j_top) pair:
    # Actually, let's use a simpler direct approach.
    #
    # Direct rank-8 for 2x2x4:
    # For i in {0,1}, j in {0,1}, k in {0,1,2,3}:
    #   C[i,k] += A[i,j] * B[j,k]
    # That's 2*2*4 = 16 terms naively. We need to reduce to 8.
    #
    # Use Strassen-like: for 2x2x2, rank 7. For 2x2x4, we can do better than 16.
    # Actually, 2x2x4 can be done in rank 8 by treating it as two 2x2x2 products
    # but sharing some structure.
    #
    # Let me just use a valid (possibly non-minimal) decomposition and verify.
    # I'll use rank 16 with a straightforward construction.
    
    # Let's use a different approach: exploit linearity
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # Define 16 terms as follows:
    # For a in {0,1}, b in {0,1}, c in {0,1}, d in {0,1}:
    #   Term index = a*4 + b*2 + c*2 + d  (0 to 15)
    #   U[i, term] = A-related: depends on a, b
    #   V[j, term] = B-related: depends on b, c  
    #   W[k, term] = C-related: depends on c, d
    #
    # This is getting too abstract. Let me just hardcode a known working decomposition.
    
    # I'll use the following rank-16 construction:
    # Split into 4 groups of 4 terms each, handling different (row, col) combinations
    
    # Group 1 (terms 0-3): Handle B[0,:] and B[1,:] contributions to C[:,0:2]
    # Group 2 (terms 4-7): Handle B[2,:] and B[3,:] contributions to C[:,0:2]
    # Group 3 (terms 8-11): Handle B[0,:] and B[1,:] contributions to C[:,2:4]
    # Group 4 (terms 12-15): Handle B[2,:] and B[3,:] contributions to C[:,2:4]
    
    # For each group, we have a 2x2x2 subproblem (2 rows of A, 2 rows of B, 2 cols of C)
    # Each 2x2x2 can be done in rank 7, but we're using 4 terms per group.
    # This won't work directly. Let me use a different split.
    
    # Alternative: Use 16 terms where each term is a rank-1 component
    # Let me just try to construct it and verify numerically.
    
    # I'll define the decomposition explicitly:
    # U[i*4+j, t], V[j*4+k, t], W[k*2+i, t] for t in 0..15
    # such that sum_t U[i*4+j,t]*V[j*4+k,t]*W[k*2+i,t] = delta (always 1 for valid i,j,k)
    # Wait, no. The tensor is T[i,j,k] = 1 for all i,j,k (it's the multiplication tensor).
    # Actually T[i,j,k] represents the coefficient: C[i,k] = sum_j A[i,j]*B[j,k]*T[i,j,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # We need: sum_{t=0}^{15} U[i*4+j, t] * V[j*4+k, t] * W[k*2+i, t] = 1 for all i,j,k
    
    # Let me try a tensor product structure:
    # U[i*4+j, t] = u1[i, t1] * u2[j, t2] where t = t1*4 + t2, t1 in {0,1,2,3}, t2 in {0,1,2,3}
    # This gives 16 terms.
    
    # Let u1 be 2x4 and u2 be 4x4 such that the product works.
    # Actually, let's just use:
    # U[i*4+j, t] = U1[i, t//4] * U2[j, t%4]
    # V[j*4+k, t] = V1[j, t//4] * V2[k, t%4]  
    # W[k*2+i, t] = W1[k, t//4] * W2[i, t%4]
    
    # Then sum_t U*V*W = sum_{t1,t2} U1[i,t1]*U2[j,t2]*V1[j,t1]*V2[k,t2]*W1[k,t1]*W2[i,t2]
    # = sum_{t1,t2} (U1[i,t1]*V1[j,t1]*W1[k,t1]) * (U2[j,t2]*V2[k,t2]*W2[i,t2])
    
    # This is a sum over t1 and t2 of products. We need this to equal 1 for all i,j,k.
    
    # This is complex. Let me just use a known rank-16 decomposition if one exists,
    # or fall back to rank 28.
    
    # Let me try a simpler direct construction with 16 terms:
    # Term t: 
    #   Let t = 4*a + 2*b + c where a,b,c in {0,1}
    #   U[i*4+j, t] = 1 if (i == a and j == b) else 0  -- no, this gives rank 8 max
    
    # I'll use the following valid rank-16 decomposition:
    # For each pair (j, k) in {0,1,2,3} x {0,1,2,3}, that's 16 pairs.
    # But we only have 16 terms, so one term per (j,k) pair? No, that doesn't work.
    
    # Let me just implement rank 28 as a fallback and try to improve.
    # Actually, let me try rank 16 with a specific construction.
    
    # Construction: Use the fact that for fixed j, C[i,k] = A[i,j]*B[j,k]
    # Sum over j. So T[i,j,k] = 1.
    # 
    # Rank-16: Let's use terms indexed by (a, b) where a in {0,1,2,3}, b in {0,1,2,3}
    # U[i*4+j, a*4+b] = Ua[i, a] * Ub[j, b]
    # V[j*4+k, a*4+b] = Va[j, a] * Vb[k, b]
    # W[k*2+i, a*4+b] = Wa[k, a] * Wb[i, b]
    
    # We need sum_{a,b} Ua[i,a]*Ub[j,b]*Va[j,a]*Vb[k,b]*Wa[k,a]*Wb[i,b] = 1
    
    # = sum_a (Ua[i,a]*Va[j,a]*Wa[k,a]) * sum_b (Ub[j,b]*Vb[k,b]*Wb[i,b])
    
    # Let X(i,j,k) = sum_a Ua[i,a]*Va[j,a]*Wa[k,a]
    # Let Y(i,j,k) = sum_b Ub[j,b]*Vb[k,b]*Wb[i,b]
    # We need X(i,j,k)*Y(i,j,k) = 1 for all i,j,k.
    
    # This is hard to satisfy. Let me abandon this approach.
    
    # I'll use a verified rank-28 decomposition for now, but let me try one more thing.
    # 
    # Actually, for 2x4x4, the known upper bound on rank is 28 (from Strassen).
    # The lower bound is at least 2*4 = 8 (trivial) or higher.
    # Let me check if rank 16 is actually achievable.
    
    # Given the complexity, I'll implement the rank-28 Strassen decomposition which is known to work.
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_term(u_vec, v_vec, w_vec, col_idx):
        for idx, coeff in u_vec:
            U[idx, col_idx] += coeff
        for idx, coeff in v_vec:
            V[idx, col_idx] += coeff
        for idx, coeff in w_vec:
            W[idx, col_idx] += coeff
            
    def strassen_terms():
        terms = []
        # M1 = (X00+X11)(Y00+Y11)
        u = {0: 1, 3: 1}
        v = {0: 1, 3: 1}
        w = {0: 1, 3: 1}
        terms.append((u, v, w))
        # M2 = (X10+X11)Y00
        u = {2: 1, 3: 1}
        v = {0: 1}
        w = {2: 1, 3: -1}
        terms.append((u, v, w))
        # M3 = X00(Y01-Y11)
        u = {0: 1}
        v = {1: 1, 3: -1}
        w = {1: 1, 3: 1}
        terms.append((u, v, w))
        # M4 = X11(Y10-Y00)
        u = {3: 1}
        v = {2: 1, 0: -1}
        w = {0: 1, 2: 1}
        terms.append((u, v, w))
        # M5 = (X00+X01)Y11
        u = {0: 1, 1: 1}
        v = {3: 1}
        w = {0: -1, 1: 1}
        terms.append((u, v, w))
        # M6 = (X10-X00)(Y00+Y01)
        u = {2: 1, 0: -1}
        v = {0: 1, 1: 1}
        w = {3: 1}
        terms.append((u, v, w))
        # M7 = (X01-X11)(Y10+Y11)
        u = {1: 1, 3: -1}
        v = {2: 1, 3: 1}
        w = {0: 1}
        terms.append((u, v, w))
        return terms

    base_terms = strassen_terms()
    
    def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx
            u_global = {}
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_offset * 2 + c
                g_idx = g_r * 4 + g_c
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            v_global = {}
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_row_offset * 2 + r
                g_c = b_col_offset * 2 + c
                g_idx = g_r * 4 + g_c
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            w_global = {}
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = c_col_offset * 2 + c
                g_idx = g_c * 2 + g_r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col)

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(1, 1, 0, 0, 7)
    apply_strassen(0, 0, 1, 1, 14)
    apply_strassen(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
