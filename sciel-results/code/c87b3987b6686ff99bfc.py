"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _build_tensor(n, m, p):
    """Build the (n*m) x (m*p) x (p*n) tensor T where T[a,b,c]=1 iff consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-8):
    """Verify that sum_r U[:,r]*V[:,r]*W[:,r] == T exactly (within tol)."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return bool(np.allclose(result, T, atol=tol))

def _schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition (always valid)."""
    n_a, n_b, n_c = n*m, m*p, p*n
    R = n*m*p
    U = np.zeros((n_a, R)); V = np.zeros((n_b, R)); W = np.zeros((n_c, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _rank17(n, m, p):
    """Verified Strassen(2x2x2) + direct decomposition of rank 17."""
    n_a, n_b, n_c = n*m, m*p, p*n
    R = 17
    U = np.zeros((n_a, R)); V = np.zeros((n_b, R)); W = np.zeros((n_c, R))

    def set_term(r, u, v, w):
        for a, val in u.items(): U[a, r] = val
        for b, val in v.items(): V[b, r] = val
        for c, val in w.items(): W[c, r] = val

    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2,0:2]
    set_term(0, {0:1,4:1}, {0:1,4:1}, {0:1,3:1})   # M1=(a0+a4)(b0+b4)
    set_term(1, {3:1,4:1}, {0:1},       {1:1,3:-1}) # M2=(a3+a4)b0
    set_term(2, {0:1},       {1:1,4:-1}, {2:1,3:1}) # M3=a0(b1-b4)
    set_term(3, {4:1},       {3:1,0:-1}, {0:1,1:1}) # M4=a4(b3-b0)
    set_term(4, {0:1,1:1},   {4:1},      {0:-1,2:1})# M5=(a0+a1)b4
    set_term(5, {3:1,0:-1},  {0:1,1:1},  {3:1})     # M6=(a3-a0)(b0+b1)
    set_term(6, {1:1,4:-1},  {3:1,4:1},  {0:1})     # M7=(a1-a4)(b3+b4)
    # Part 2: A1(2x2) x B1_right(2x1) -> C[:,2]
    set_term(7,  {0:1}, {2:1}, {4:1})
    set_term(8,  {1:1}, {5:1}, {4:1})
    set_term(9,  {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    # Part 3: A2(2x1) x B2(1x3) -> C[0:2,0:3]
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    return U, V, W

def _try_reduced(T, U, V, W, R, deadline):
    """
    Bounded search for a strictly smaller rank decomposition.
    Only returns a solution if it is EXACTLY verified; otherwise returns None.
    """
    n_a, n_b, n_c = T.shape
    rank = U.shape[1]
    best = None

    def matmul_of_terms(keep, extra):
        cols = list(keep)
        Ux = np.zeros((n_a, len(cols) + len(extra)))
        Vx = np.zeros((n_b, len(cols) + len(extra)))
        Wx = np.zeros((n_c, len(cols) + len(extra)))
        for ci, ti in enumerate(cols):
            Ux[:, ci] = U[:, ti]; Vx[:, ci] = V[:, ti]; Wx[:, ci] = W[:, ti]
        for ei, (uu, vv, ww) in enumerate(extra):
            Ux[:, len(cols)+ei] = uu; Vx[:, len(cols)+ei] = vv; Wx[:, len(cols)+ei] = ww
        return Ux, Vx, Wx

    grid = [-2, -1, 0, 1, 2]

    # (a) single-term drop
    for drop in range(rank):
        if time.time() > deadline: break
        keep = [t for t in range(rank) if t != drop]
        Ux, Vx, Wx = matmul_of_terms(keep, [])
        if Ux.shape[1] < rank and _verify_decomposition(Ux, Vx, Wx, T):
            best = (Ux, Vx, Wx)
            break

    if best is not None:
        return best

    # (b) drop two terms, add one new rank-1 term
    for d1 in range(rank):
        if time.time() > deadline: break
        for d2 in range(d1+1, rank):
            if time.time() > deadline: break
            keep = [t for t in range(rank) if t not in (d1, d2)]
            for ca in grid:
                for cb in grid:
                    if ca == 0 and cb == 0:
                        continue
                    uu = ca*U[:, d1] + cb*U[:, d2]
                    vv = ca*V[:, d1] + cb*V[:, d2]
                    ww = ca*W[:, d1] + cb*W[:, d2]
                    Ux, Vx, Wx = matmul_of_terms(keep, [(uu, vv, ww)])
                    if Ux.shape[1] < rank and _verify_decomposition(Ux, Vx, Wx, T):
                        return (Ux, Vx, Wx)
    return best

def _rank16_tensor_completion(n, m, p, deadline):
    """
    Tensor completion approach for rank-16:
    Fix a structured sparsity pattern derived from the Strassen core,
    then solve a linear system for the free variables to check exact recovery.
    
    Strategy: Start from the rank-17 decomposition. The 7 Strassen terms handle
    the 2x2x2 core. The remaining 10 terms handle cross-products. We try to
    merge pairs of the 10 "direct" terms into single terms with new coefficients,
    or restructure the entire factorization with 16 terms.
    
    We formulate: given fixed supports (which entries of U,V,W are nonzero),
    find values such that sum_r U[:,r] V[:,r] W[:,r] = T.
    """
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    T_flat = T.flatten()
    
    # The 18 nonzero entries of T
    # T[a,b,c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    # For n=2,m=3,p=3:
    # i=0,j=0,k=0: a=0,b=0,c=0
    # i=0,j=0,k=1: a=0,b=1,c=2
    # i=0,j=0,k=2: a=0,b=2,c=4
    # i=0,j=1,k=0: a=3,b=3,c=0
    # i=0,j=1,k=1: a=3,b=4,c=2
    # i=0,j=1,k=2: a=3,b=5,c=4
    # i=0,j=2,k=0: a=6,b=6,c=0  -- wait, n=2 so a=i*m+j, i in [0,2), j in [0,3)
    # Let me recompute: n=2, m=3, p=3
    # a = i*m + j, i in {0,1}, j in {0,1,2} -> a in {0,1,2,3,4,5}
    # b = j*p + k, j in {0,1,2}, k in {0,1,2} -> b in {0,...,8}
    # c = k*n + i, k in {0,1,2}, i in {0,1} -> c in {0,...,5}
    # Nonzero entries:
    # (i,j,k) -> (a,b,c)
    # (0,0,0)->(0,0,0), (0,0,1)->(0,1,2), (0,0,2)->(0,2,4)
    # (0,1,0)->(3,3,0), (0,1,1)->(3,4,2), (0,1,2)->(3,5,4)
    # (0,2,0)->(6,6,0) -- WRONG, a=i*m+j=0*3+2=2, not 6
    # Let me redo: a = i*m + j
    # i=0: a=j, so a in {0,1,2}
    # i=1: a=3+j, so a in {3,4,5}
    # b = j*p + k
    # j=0: b=k, b in {0,1,2}
    # j=1: b=3+k, b in {3,4,5}
    # j=2: b=6+k, b in {6,7,8}
    # c = k*n + i = 2k + i
    # k=0: c=i, c in {0,1}
    # k=1: c=2+i, c in {2,3}
    # k=2: c=4+i, c in {4,5}
    
    # So the 18 nonzero entries are:
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                nonzero_indices.append((a, b, c))
    
    # For rank-16 tensor completion, we try a structured approach:
    # Use the 7 Strassen terms for the 2x2x2 core (rows 0-1, cols 0-3, outputs 0-3)
    # and find a rank-9 decomposition for the remaining part.
    # 
    # Actually, let's try a different structured approach:
    # Fix 16 terms with specific sparsity patterns and solve for values.
    
    # Approach: Try to find 16 rank-1 terms where each term has at most 2 nonzeros
    # in each factor (similar to Strassen structure).
    
    # Let's try a specific structured pattern:
    # Terms 0-6: Strassen-like for the 2x2x2 block (A[0:2,0:2] * B[0:2,0:2])
    # Terms 7-15: Handle the remaining 9 output entries with 9 terms,
    #   but try to merge some.
    
    # For now, let's try a linear algebra approach:
    # Assume we fix which (a,b,c) triples each term contributes to.
    # Each term r contributes to T[a,b,c] via U[a,r]*V[b,r]*W[c,r].
    # If we fix the support of each term (which a,b,c indices are nonzero),
    # we get a system of equations.
    
    # This is complex, so let's try a simpler structured search:
    # Try all ways to drop 1 term from rank-17 and add a "hybrid" term.
    
    U17, V17, W17 = _rank17(n, m, p)
    R17 = U17.shape[1]
    
    # Try dropping one term and replacing with a linear combination of two other terms
    for drop in range(R17):
        if time.time() > deadline:
            return None
        keep = [t for t in range(R17) if t != drop]
        # Try to add back a term that's a linear combo of two kept terms
        for i in range(len(keep)):
            for j in range(i+1, len(keep)):
                for alpha in [-2,-1,-0.5,0.5,1,2]:
                    for beta in [-2,-1,-0.5,0.5,1,2]:
                        t1, t2 = keep[i], keep[j]
                        uu = alpha*U17[:,t1] + beta*U17[:,t2]
                        vv = alpha*V17[:,t1] + beta*V17[:,t2]
                        ww = alpha*W17[:,t1] + beta*W17[:,t2]
                        # Check if this new term plus kept terms (minus t1,t2) gives rank 16
                        # Actually we want to keep all except drop, and add new term
                        # That gives rank 17 still. We need to also drop t1 or t2.
                        pass
        # Simpler: just check if dropping one term works (already done in _try_reduced)
    
    # Try a more systematic approach: fix a rank-16 pattern and solve
    # Use the fact that the tensor has only 18 nonzero entries.
    # Each rank-1 term U[:,r]V[:,r]W[:,r] has at most (nnz_U)*(nnz_V)*(nnz_W) nonzeros.
    # If each factor has 2 nonzeros, each term has at most 8 nonzeros.
    # 16 terms * 8 = 128 potential nonzeros, but we only need 18.
    
    # Let's try: fix the 7 Strassen terms (they handle 8 outputs in the 2x2x2 core)
    # and find 9 more terms for the remaining 10 outputs (one output is shared).
    
    # The 2x2x2 core uses A[0:2,0:2] (rows 0,1,3,4) * B[0:2,0:2] (cols 0,1,3,4)
    # -> C[0:2,0:2] (outputs 0,1,3,4)
    # Wait, C[0:2,0:2] means c indices where k in {0,1} and i in {0,1}
    # c = 2k+i: k=0,i=0->0; k=0,i=1->1; k=1,i=0->2; k=1,i=1->3
    # So core handles c in {0,1,2,3}, i.e., outputs 0,1,2,3
    
    # Remaining outputs: c=4,5 (k=2, i=0,1)
    # c=4: k=2,i=0 -> (i,j,k) with i=0,k=2, j in {0,1,2} -> 3 entries
    # c=5: k=2,i=1 -> (i,j,k) with i=1,k=2, j in {0,1,2} -> 3 entries
    # Total remaining: 6 entries
    
    # But the Strassen terms also affect other entries! Let me check.
    # Term 0: U has nonzeros at a=0,4; V at b=0,4; W at c=0,3
    # So it contributes to T[a,b,c] for a in {0,4}, b in {0,4}, c in {0,3}
    # The actual T entries: T[0,0,0]=1, T[0,4,0]=?, T[4,0,0]=?, T[4,4,0]=?
    # T[0,4,0]: i=0,j=0,k=0? No, b=4 means j=1,k=1. So T[0,4,0] would need i=0,j=1,k=1 -> c=2*1+0=2, not 0. So T[0,4,0]=0.
    # The Strassen terms are designed so that cross-terms cancel.
    
    # This is getting complex. Let's just try the linear system approach directly.
    
    # For rank 16, we need to find U (6x16), V (9x16), W (6x16) such that
    # sum_r U[:,r] V[:,r] W[:,r] = T
    # 
    # This is a system of 6*9*6 = 324 equations in 16*(6+9+6) = 360 unknowns.
    # But it's nonlinear (trilinear). 
    #
    # Linearize: fix U and V, solve for W linearly. Or fix two, solve for third.
    # This is an alternating minimization / tensor completion approach.
    
    # Initialize from rank-17 by zeroing one column
    for drop in range(R17):
        if time.time() > deadline:
            return None
        
        U0 = U17[:, [t for t in range(R17) if t != drop]].copy()
        V0 = V17[:, [t for t in range(R17) if t != drop]].copy()
        W0 = W17[:, [t for t in range(R17) if t != drop]].copy()
        
        # Now U0 is 6x16, V0 is 9x16, W0 is 6x16
        # We want to adjust these to exactly match T.
        # Use alternating least squares: fix two, solve for third.
        
        U_alt = U0.copy()
        V_alt = V0.copy()
        W_alt = W0.copy()
        
        for iteration in range(50):
            if time.time() > deadline:
                return None
            
            # Solve for W: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For each c: T[:,:,c] = sum_r U[:,r] V[:,r] W[c,r]
            # This is: T[:,:,c] = (U * W[c,:]) * V^T ... not quite.
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Let's define M_c = sum_r U[:,r] V[:,r] W[c,r]
            # We want M_c = T[:,:,c] for each c.
            # This is linear in W[:,c] if we fix U and V.
            # T[:,:,c] = sum_r (U[:,r] V[:,r]) W[c,r]
            # Let A_r = U[:,r] V[:,r] (outer product, 6x9)
            # T[:,:,c] = sum_r A_r * W[c,r]
            # So for fixed c, we have: T[:,:,c] = sum_r W[c,r] A_r
            # This is a linear system: [A_0, A_1, ..., A_15] * W[:,c] = T[:,:,c].flatten()
            # where [A_0,...,A_15] is a 54x16 matrix.
            
            # Build the linear system for W
            A_mat = np.zeros((n_a * n_b, 16))
            for r in range(16):
                A_mat[:, r] = (U_alt[:, r] * V_alt[:, r]).flatten()
            
            rhs = T_flat  # This doesn't work directly because T[:,:,c] for different c
            # Actually, for each c, T[:,:,c] is a 6x9 matrix = 54 entries.
            # And W[:,c] is a 16-vector.
            # So we have 6 separate 54x16 systems, one for each c.
            
            W_new = np.zeros((n_c, 16))
            for c in range(n_c):
                rhs_c = T[:, :, c].flatten()
                # Solve A_mat * W[:,c] = rhs_c
                try:
                    W_new[c, :] = np.linalg.lstsq(A_mat, rhs_c, rcond=None)[0]
                except:
                    return None
            
            # Check convergence
            if np.allclose(W_new, W_alt, atol=1e-10):
                W_alt = W_new
                break
            W_alt = W_new
            
            # Solve for U similarly
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed a: T[a,:,:] = sum_r U[a,r] V[:,r] W[:,r]^T ... 
            # T[a,b,c] = sum_r U[a,r] (V[b,r] W[c,r])
            # Let B_r = V[:,r] W[:,r]^T (9x6 outer product)
            # T[a,:,:] = sum_r U[a,r] B_r
            # So [B_0,...,B_15] * U[:,a] = T[a,:,:].flatten()
            
            B_mat = np.zeros((n_b * n_c, 16))
            for r in range(16):
                B_mat[:, r] = (V_alt[:, r] * W_alt[:, r]).flatten()
            
            U_new = np.zeros((n_a, 16))
            for a in range(n_a):
                rhs_a = T[a, :, :].flatten()
                try:
                    U_new[a, :] = np.linalg.lstsq(B_mat, rhs_a, rcond=None)[0]
                except:
                    return None
            
            if np.allclose(U_new, U_alt, atol=1e-10):
                U_alt = U_new
                break
            U_alt = U_new
            
            # Solve for V
            # T[a,b,c] = sum_r V[b,r] (U[a,r] W[c,r])
            # Let C_r = U[:,r] W[:,r]^T (6x6)
            # T[:,b,:] = sum_r V[b,r] C_r
            # [C_0,...,C_15] * V[:,b] = T[:,b,:].flatten()
            
            C_mat = np.zeros((n_a * n_c, 16))
            for r in range(16):
                C_mat[:, r] = (U_alt[:, r] * W_alt[:, r]).flatten()
            
            V_new = np.zeros((n_b, 16))
            for b in range(n_b):
                rhs_b = T[:, b, :].flatten()
                try:
                    V_new[b, :] = np.linalg.lstsq(C_mat, rhs_b, rcond=None)[0]
                except:
                    return None
            
            if np.allclose(V_new, V_alt, atol=1e-10):
                V_alt = V_new
                break
            V_alt = V_new
        
        # Verify the result
        if _verify_decomposition(U_alt, V_alt, W_alt, T):
            return (U_alt, V_alt, W_alt)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # explicit safety margin (< 120s wall-clock)

    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    # Baseline: verified rank-17 (Strassen + direct).
    U17, V17, W17 = _rank17(n, m, p)
    if _verify_decomposition(U17, V17, W17, T):
        base = (U17, V17, W17)
    else:
        base = _schoolbook(n, m, p)

    # Try rank-16 tensor completion
    rank16_result = _rank16_tensor_completion(n, m, p, deadline)
    if rank16_result is not None:
        U, V, W = rank16_result
    else:
        # Bounded refinement: attempt a strictly smaller, EXACTLY verified rank.
        reduced = _try_reduced(T, base[0], base[1], base[2], base[0].shape[1], deadline)
        if reduced is not None:
            U, V, W = reduced
        else:
            U, V, W = base

    # Final exact verification; fall back to schoolbook on any failure.
    if not _verify_decomposition(U, V, W, T):
        U, V, W = _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
