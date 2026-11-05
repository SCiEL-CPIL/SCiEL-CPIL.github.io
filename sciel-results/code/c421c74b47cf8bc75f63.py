"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

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
    T_flat = T.flatten()

    # (a) single-term drop: try removing each term
    for drop in range(rank):
        if time.time() > deadline: break
        keep = [t for t in range(rank) if t != drop]
        Ux = np.zeros((n_a, len(keep))); Vx = np.zeros((n_b, len(keep))); Wx = np.zeros((n_c, len(keep)))
        for ci, ti in enumerate(keep):
            Ux[:, ci] = U[:, ti]; Vx[:, ci] = V[:, ti]; Wx[:, ci] = W[:, ti]
        if Ux.shape[1] < rank and _verify_decomposition(Ux, Vx, Wx, T):
            best = (Ux, Vx, Wx)
            break

    if best is not None:
        return best

    # (b) drop two terms, add one new rank-1 term from a small structured grid.
    grid = [-3, -2, -1, 0, 1, 2, 3]
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
                    Ux = np.zeros((n_a, len(keep)+1)); Vx = np.zeros((n_b, len(keep)+1)); Wx = np.zeros((n_c, len(keep)+1))
                    for ci, ti in enumerate(keep):
                        Ux[:, ci] = U[:, ti]; Vx[:, ci] = V[:, ti]; Wx[:, ci] = W[:, ti]
                    Ux[:, len(keep)] = uu; Vx[:, len(keep)] = vv; Wx[:, len(keep)] = ww
                    if Ux.shape[1] < rank and _verify_decomposition(Ux, Vx, Wx, T):
                        return (Ux, Vx, Wx)
    
    # (c) More targeted randomized search for rank 16
    # Start from the rank-17 structure and try to merge terms
    np.random.seed(42)
    
    target_ranks = [16, 15]
    max_restarts_per_rank = {16: 3000, 15: 1000}
    
    for target_rank in target_ranks:
        if time.time() > deadline: break
        
        max_restarts = max_restarts_per_rank.get(target_rank, 100)
        
        for restart in range(max_restarts):
            if time.time() > deadline: break
            
            if target_rank == 16 and restart < 100:
                # Strategy 1: Start from rank-17, drop 2 terms, add 1 new term
                # Try different pairs to drop
                d1 = restart % rank
                d2 = (restart + 1) % rank
                if d2 == d1:
                    d2 = (d1 + 1) % rank
                keep = [t for t in range(rank) if t not in (d1, d2)]
                
                U_cand = np.zeros((n_a, target_rank))
                V_cand = np.zeros((n_b, target_rank))
                W_cand = np.zeros((n_c, target_rank))
                
                for ci, ti in enumerate(keep):
                    U_cand[:, ci] = U[:, ti]
                    V_cand[:, ci] = V[:, ti]
                    W_cand[:, ci] = W[:, ti]
                
                # Add one new random term
                r = len(keep)
                num_comps = np.random.randint(2, 7)
                comps = np.random.choice(12, num_comps, replace=False)
                
                u_vec = np.zeros(n_a)
                v_vec = np.zeros(n_b)
                w_vec = np.zeros(n_c)
                
                for comp in comps:
                    i = comp // 9
                    j = (comp % 9) // 3
                    k = comp % 3
                    
                    a = i*3 + j
                    b = j*3 + k
                    c = k*2 + i
                    
                    u_vec[a] += np.random.choice([-2, -1, 1, 2])
                    v_vec[b] += np.random.choice([-2, -1, 1, 2])
                    w_vec[c] += np.random.choice([-2, -1, 1, 2])
                
                U_cand[:, r] = u_vec
                V_cand[:, r] = v_vec
                W_cand[:, r] = w_vec
            else:
                # Strategy 2: Pure random initialization
                U_cand = np.zeros((n_a, target_rank))
                V_cand = np.zeros((n_b, target_rank))
                W_cand = np.zeros((n_c, target_rank))
                
                for r in range(target_rank):
                    num_comps = np.random.randint(2, 7)
                    comps = np.random.choice(12, num_comps, replace=False)
                    
                    u_vec = np.zeros(n_a)
                    v_vec = np.zeros(n_b)
                    w_vec = np.zeros(n_c)
                    
                    for comp in comps:
                        i = comp // 9
                        j = (comp % 9) // 3
                        k = comp % 3
                        
                        a = i*3 + j
                        b = j*3 + k
                        c = k*2 + i
                        
                        u_vec[a] += np.random.choice([-2, -1, 1, 2])
                        v_vec[b] += np.random.choice([-2, -1, 1, 2])
                        w_vec[c] += np.random.choice([-2, -1, 1, 2])
                    
                    U_cand[:, r] = u_vec
                    V_cand[:, r] = v_vec
                    W_cand[:, r] = w_vec
            
            # Construct matrix A where A[:, r] is the flattened outer product
            A = np.zeros((324, target_rank))
            for r in range(target_rank):
                outer = np.einsum('a,b,c->abc', U_cand[:, r], V_cand[:, r], W_cand[:, r])
                A[:, r] = outer.flatten()
                
            # Solve A c = T_flat
            try:
                c, residuals, rank_s, s = np.linalg.lstsq(A, T_flat, rcond=None)
                # Check if the solution is exact
                if np.allclose(A @ c, T_flat, atol=1e-8):
                    U_sol = U_cand.copy()
                    for r in range(target_rank):
                        U_sol[:, r] *= c[r]
                    
                    if _verify_decomposition(U_sol, V_cand, W_cand, T):
                        return (U_sol, V_cand, W_cand)
            except np.linalg.LinAlgError:
                pass
            
    return best

def _combinatorial_search(T, deadline, max_rank=16):
    """
    Systematic combinatorial search over sparse support patterns.
    For each rank r from 14 to max_rank, try all combinations of 'active' tensor entries
    that could form a rank-1 term, and solve for coefficients.
    This is a heuristic approach: we don't enumerate ALL supports (too many), but we
    enumerate structured supports based on the Strassen pattern and random perturbations.
    """
    n_a, n_b, n_c = T.shape
    T_flat = T.flatten()
    
    # Pre-compute all possible rank-1 outer products for a small set of candidate vectors
    # We'll use a grid of integer vectors with small entries
    # To keep it tractable, we limit the number of non-zeros per vector
    
    # Candidate vectors for U, V, W
    # We'll generate them on the fly in the search loop
    
    for target_rank in range(14, max_rank + 1):
        if time.time() > deadline:
            break
        
        # For this target rank, try to find a decomposition
        # We'll use a randomized approach with structured initialization
        
        np.random.seed(12345 + target_rank)
        
        for restart in range(500):
            if time.time() > deadline:
                break
            
            # Initialize with random sparse vectors
            U_cand = np.zeros((n_a, target_rank))
            V_cand = np.zeros((n_b, target_rank))
            W_cand = np.zeros((n_c, target_rank))
            
            for r in range(target_rank):
                # Random sparse support
                n_u = np.random.randint(1, 4)
                n_v = np.random.randint(1, 4)
                n_w = np.random.randint(1, 4)
                
                u_idx = np.random.choice(n_a, n_u, replace=False)
                v_idx = np.random.choice(n_b, n_v, replace=False)
                w_idx = np.random.choice(n_c, n_w, replace=False)
                
                for idx in u_idx:
                    U_cand[idx, r] = np.random.choice([-2, -1, 1, 2])
                for idx in v_idx:
                    V_cand[idx, r] = np.random.choice([-2, -1, 1, 2])
                for idx in w_idx:
                    W_cand[idx, r] = np.random.choice([-2, -1, 1, 2])
            
            # Build the linear system
            A = np.zeros((324, target_rank))
            for r in range(target_rank):
                outer = np.einsum('a,b,c->abc', U_cand[:, r], V_cand[:, r], W_cand[:, r])
                A[:, r] = outer.flatten()
            
            # Solve for coefficients
            try:
                c, residuals, rank_s, s = np.linalg.lstsq(A, T_flat, rcond=None)
                if np.allclose(A @ c, T_flat, atol=1e-8):
                    U_sol = U_cand.copy()
                    for r in range(target_rank):
                        U_sol[:, r] *= c[r]
                    
                    if _verify_decomposition(U_sol, V_cand, W_cand, T):
                        return (U_sol, V_cand, W_cand)
            except np.linalg.LinAlgError:
                pass
        
        # Also try a more structured approach: start from rank-17 and merge terms
        if target_rank == 16:
            # Try all pairs of terms from rank-17 to merge
            U17, V17, W17 = _rank17(2, 3, 3)
            for d1 in range(17):
                if time.time() > deadline:
                    break
                for d2 in range(d1 + 1, 17):
                    if time.time() > deadline:
                        break
                    
                    keep = [t for t in range(17) if t not in (d1, d2)]
                    U_cand = np.zeros((n_a, 16))
                    V_cand = np.zeros((n_b, 16))
                    W_cand = np.zeros((n_c, 16))
                    
                    for ci, ti in enumerate(keep):
                        U_cand[:, ci] = U17[:, ti]
                        V_cand[:, ci] = V17[:, ti]
                        W_cand[:, ci] = W17[:, ti]
                    
                    # The merged term is a linear combination of the two dropped terms
                    # Try a grid of coefficients
                    for alpha in [-3, -2, -1, 0, 1, 2, 3]:
                        for beta in [-3, -2, -1, 0, 1, 2, 3]:
                            if alpha == 0 and beta == 0:
                                continue
                            
                            u_merged = alpha * U17[:, d1] + beta * U17[:, d2]
                            v_merged = alpha * V17[:, d1] + beta * V17[:, d2]
                            w_merged = alpha * W17[:, d1] + beta * W17[:, d2]
                            
                            U_cand[:, 15] = u_merged
                            V_cand[:, 15] = v_merged
                            W_cand[:, 15] = w_merged
                            
                            if _verify_decomposition(U_cand, V_cand, W_cand, T):
                                return (U_cand, V_cand, W_cand)
    
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

    # Bounded refinement: attempt a strictly smaller, EXACTLY verified rank.
    # This never hurts: it only replaces the baseline if verified.
    reduced = _try_reduced(T, base[0], base[1], base[2], base[0].shape[1], deadline)
    if reduced is not None:
        U, V, W = reduced
    else:
        # Try the combinatorial search if time remains
        if time.time() < deadline - 5:
            comb = _combinatorial_search(T, deadline, max_rank=16)
            if comb is not None:
                U, V, W = comb
            else:
                U, V, W = base
        else:
            U, V, W = base

    # Final exact verification; fall back to schoolbook on any failure.
    if not _verify_decomposition(U, V, W, T):
        U, V, W = _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
