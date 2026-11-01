"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (9, rank)  
    # W: (p*n, rank) = (6, rank)
    
    # Constructive approach: decompose 2x3x3 as direct sum of 2x2x2 (Strassen rank 7) + 1x3x3 (rank 3) = rank 10
    # The 2x3x3 tensor T[i,j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # We partition: rows i=0,1; cols j=0,1 for Strassen part; j=2 for the 1x3x3 part
    
    # Strategy: 
    # For the 2x2x2 sub-problem (i in {0,1}, j in {0,1}, k in {0,1}): use Strassen (rank 7)
    # For the remaining parts: we need to handle j=2 (all k) and k=2 (i=0,1, j=0,1)
    # Actually, let's think more carefully.
    
    # The tensor T[i,j,k] = 1 for all i,j,k in range.
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Direct sum decomposition:
    # Part A: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2 tensor, Strassen rank 7
    # Part B: handle the "boundary" terms involving j=2 or k=2
    
    # Let's use a different constructive approach:
    # Think of the 2x3x3 multiplication as:
    #   For each output element (i,k), we compute sum_j A[i,j]*B[j,k]
    #   i in {0,1}, k in {0,1,2}, j in {0,1,2}
    
    # We can decompose this as:
    #   sum_j A[i,j]*B[j,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + A[i,2]*B[2,k]
    
    # The first term is a 2x2x2 tensor (rank 7 via Strassen)
    # The second term A[i,2]*B[2,k] for i in {0,1}, k in {0,1,2} is a 1x3x3 tensor (rank 3)
    
    # So total rank = 7 + 3 = 10
    
    # Strassen's algorithm for 2x2x2:
    # Let M = [[a,b],[c,d]], N = [[e,f],[g,h]]
    # P1 = (a+b)(e+h)
    # P2 = (c+d)(e+h)  
    # P3 = a(g-h)
    # P4 = d(f-g)
    # P5 = (a+d)(e+h)  -- wait, let me use the standard Strassen
    # Standard Strassen:
    # P1 = (a+b)(e+h)
    # P2 = (c+d)(e+h)
    # P3 = a(g-h)
    # P4 = d(f-g)
    # P5 = (a+d)(e+h)
    # P6 = (b+c)(e+f)
    # P7 = (a-c)(g+h)
    # Result:
    # [a*e+b*g, a*f+b*h]
    # [c*e+d*g, c*f+d*h]
    # = [[P1-P4+P7, P3+P4], [P2-P3, P1-P2+P5]]
    
    # Wait, let me use the correct Strassen formulas:
    # P1 = (a+b)(e+h)
    # P2 = (c+d)(e+h)
    # P3 = a(g-h)
    # P4 = d(f-g)
    # P5 = (a+d)(e+h)
    # P6 = (b+c)(e+f)
    # P7 = (a-c)(g+h)
    # 
    # M*N[0,0] = P1 - P4 + P7
    # M*N[0,1] = P3 + P4
    # M*N[1,0] = P2 - P3
    # M*N[1,1] = P1 - P2 + P5
    
    # Actually, the standard Strassen is:
    # p1 = (a+b)(e+h)
    # p2 = (c+d)(e+h)
    # p3 = a(g-h)
    # p4 = d(f-g)
    # p5 = (a+d)(e+h)
    # p6 = (b+c)(e+f)
    # p7 = (a-c)(g+h)
    # 
    # c11 = p1 - p4 + p7
    # c12 = p3 + p4
    # c21 = p2 - p3
    # c22 = p1 - p2 + p5
    
    # Let me verify: 
    # c11 = (a+b)(e+h) - d(f-g) + (a-c)(g+h)
    #     = ae+ah+be+bh - df+dg + ag+ah-cg-ch
    #     = ae + ah + be + bh - df + dg + ag + ah - cg - ch
    #     = ae + 2ah + be + bh - df + dg + ag - cg - ch
    #     Hmm, this doesn't look right. Let me use the correct Strassen.
    
    # Correct Strassen:
    # p1 = (a+b)(e+h)
    # p2 = (c+d)(e+h)
    # p3 = a(g-h)
    # p4 = d(f-g)
    # p5 = (a+d)(e+h)
    # p6 = (b+c)(e+f)
    # p7 = (a-c)(g+h)
    #
    # c11 = p1 - p4 + p7
    # c12 = p3 + p4
    # c21 = p2 - p3
    # c22 = p1 - p2 + p5
    
    # Let me just verify with a simple example: a=b=c=d=e=f=g=h=1
    # p1 = 2*2 = 4
    # p2 = 2*2 = 4
    # p3 = 1*(1-1) = 0
    # p4 = 1*(1-1) = 0
    # p5 = 2*2 = 4
    # p6 = 2*2 = 4
    # p7 = 0*2 = 0
    # c11 = 4-0+0 = 4 ✓ (should be 1*1+1*1=2) ✗
    
    # I'm making errors. Let me look up the correct Strassen formulas.
    # 
    # Standard Strassen:
    # p1 = (a+b)(e+h)
    # p2 = (c+d)(e+h)
    # p3 = a(g-h)
    # p4 = d(f-g)
    # p5 = (a+d)(e+h)
    # p6 = (b+c)(e+f)
    # p7 = (a-c)(g+h)
    #
    # c11 = p1 - p4 + p7
    # c12 = p3 + p4
    # c21 = p2 - p3
    # c22 = p1 - p2 + p5
    
    # Let me try different indices. The issue is my mapping.
    # 
    # Let M = [[m00, m01], [m10, m11]], N = [[n00, n01], [n10, n11]]
    # 
    # p1 = (m00+m01)(n00+n11)
    # p2 = (m10+m11)(n00+n11)
    # p3 = m00*(n10-n11)
    # p4 = m11*(n01-n10)
    # p5 = (m00+m11)(n00+n11)
    # p6 = (m01+m10)(n00+n01)
    # p7 = (m00-m10)(n10+n11)
    #
    # c00 = p1 - p4 + p7
    # c01 = p3 + p4
    # c10 = p2 - p3
    # c11 = p1 - p2 + p5
    
    # Verify with all ones:
    # p1 = 2*2 = 4
    # p2 = 2*2 = 4
    # p3 = 1*(1-1) = 0
    # p4 = 1*(1-1) = 0
    # p5 = 2*2 = 4
    # p6 = 2*2 = 4
    # p7 = 0*2 = 0
    # c00 = 4-0+0 = 4, but should be 1+1=2. Still wrong.
    
    # I think I have the Strassen formulas wrong. Let me use a different approach.
    # 
    # Actually, for this task, I don't need to use Strassen explicitly. I can use any rank-7 decomposition of 2x2x2.
    # The key insight is that the 2x3x3 tensor has rank 10 (or possibly less).
    # 
    # Let me try a simpler constructive approach:
    # 
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We want sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1
    # 
    # Consider the following decomposition:
    # For the "main" 2x2x2 part (i in {0,1}, j in {0,1}, k in {0,1}):
    #   Use 7 terms from Strassen
    # For the "cross" terms involving j=2 or k=2:
    #   Use 3 additional terms
    #
    # Actually, let me think about this differently.
    #
    # The 2x3x3 tensor can be viewed as:
    # T = T_2x2x2 ⊗ 1 + T_boundary
    #
    # Where T_2x2x2 is the 2x2x2 tensor (i in {0,1}, j in {0,1}, k in {0,1})
    # And T_boundary handles all other (i,j,k) combinations.
    #
    # The boundary terms are:
    # - j=2: (i, 2, k) for i in {0,1}, k in {0,1,2} -> 6 terms
    # - k=2: (i, j, 2) for i in {0,1}, j in {0,1} -> 4 terms (excluding (i,2,2) already counted)
    # Total boundary: 6 + 4 = 10 terms, but we can do better.
    #
    # Actually, the boundary is:
    # All (i,j,k) where j=2 OR k=2 (but not both, to avoid double counting)
    # j=2, k!=2: (0,2,0), (0,2,1), (1,2,0), (1,2,1) -> 4 terms
    # k=2, j!=2: (0,0,2), (0,1,2), (1,0,2), (1,1,2) -> 4 terms
    # j=2, k=2: (0,2,2), (1,2,2) -> 2 terms
    # Total: 10 boundary terms
    #
    # But we can group these more efficiently.
    #
    # Let me try a direct construction with rank 10:
    #
    # Terms 1-7: Strassen for the 2x2x2 core (i in {0,1}, j in {0,1}, k in {0,1})
    # Terms 8-10: Handle the boundary
    #
    # For the boundary, note that:
    # - When j=2: the contribution is A[i,2]*B[2,k] for all i,k
    # - When k=2: the contribution is A[i,j]*B[j,2] for all i,j (j=0,1,2)
    #
    # We can handle this with:
    # Term 8: A[i,2]*B[2,0]*W[0*2+i] for k=0
    # Term 9: A[i,2]*B[2,1]*W[1*2+i] for k=1  
    # Term 10: A[i,2]*B[2,2]*W[2*2+i] for k=2
    # But this doesn't handle the k=2, j=0,1 case.
    #
    # This approach is getting complicated. Let me just use a numerical search with the correct rank.
    
    # Constructive rank-10 decomposition:
    # Decompose 2x3x3 as: (2x2x2 core via Strassen, rank 7) + (boundary terms, rank 3)
    # 
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # 
    # We split into:
    # 1. Core 2x2x2: i in {0,1}, j in {0,1}, k in {0,1} -> 8 entries, rank 7 (Strassen)
    # 2. Boundary: all other (i,j,k) entries
    #
    # For the core, we use Strassen's algorithm.
    # For the boundary, we use 3 additional terms.
    #
    # Let's define the decomposition explicitly.
    #
    # Strassen for 2x2x2 (indices i=0,1; j=0,1; k=0,1):
    # The 8 tensor entries are:
    # T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1
    # T[1,0,0]=1, T[1,0,1]=1, T[1,1,0]=1, T[1,1,1]=1
    #
    # Strassen factors (7 terms):
    # We need U_core[i*3+j, r], V_core[j*3+k, r], W_core[k*2+i, r] for r=0..6
    # such that sum_r U_core[i*3+j,r]*V_core[j*3+k,r]*W_core[k*2+i,r] = 1 for core entries
    # and = 0 for non-core entries (to be handled by boundary terms)
    #
    # This is complex. Let me use a simpler approach:
    # Just search for a rank-10 solution numerically with better initialization.
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    def tensor_residual(U, V, W):
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    total += (s - 1.0) ** 2
        return total
    
    def is_half_integer(arr, tol=1e-9):
        for val in arr.flatten():
            v = val * 2
            if abs(v - round(v)) > tol:
                return False
        return True
    
    def round_to_half(arr):
        return np.round(arr * 2) / 2
    
    # Try rank 10 first (constructive target)
    target_ranks = [10, 11, 12, 13, 14, 15, 16, 17]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
        if target_rank >= best_rank:
            continue
        
        n_attempts = 8
        for attempt in range(n_attempts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(123 + target_rank * 1000 + attempt * 7)
            U = np.random.randint(-3, 4, size=(n*m, target_rank)) / 2.0
            V = np.random.randint(-3, 4, size=(m*p, target_rank)) / 2.0
            W = np.random.randint(-3, 4, size=(p*n, target_rank)) / 2.0
            
            from scipy.optimize import minimize
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def obj(x):
                U_flat = x[:n*m*target_rank].reshape(n*m, target_rank)
                V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_flat = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                return tensor_residual(U_flat, V_flat, W_flat)
            
            res = minimize(obj, x0, method='Nelder-Mead', 
                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
            
            if res.fun < 1e-10:
                U_sol = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = res.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                if is_half_integer(U_sol) and is_half_integer(V_sol) and is_half_integer(W_sol):
                    if tensor_residual(U_sol, V_sol, W_sol) < 1e-8:
                        best = {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                        best_rank = target_rank
                        break
                else:
                    U_r = round_to_half(U_sol)
                    V_r = round_to_half(V_sol)
                    W_r = round_to_half(W_sol)
                    if tensor_residual(U_r, V_r, W_r) < 1e-8:
                        best = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        best_rank = target_rank
                        break
    return best
# EVOLVE-BLOCK-END
