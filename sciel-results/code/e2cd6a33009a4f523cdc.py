import numpy as np
import time
import itertools
import random
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # 1. Generate Baseline (Schoolbook Rank 72)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_sb

    def verify(U, V, W, n, m, p):
        """Verify exact decomposition with integer/half-integer tolerance."""
        rank = len(U[0]) if U else 0
        if rank == 0:
            return False
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def solve_sparse_pattern(pattern, rank, time_limit):
        """
        Given a set of 'monomial' patterns (indices (i,j,k) for each term c),
        solve for the coefficients u_c, v_c, w_c such that:
        sum_c u_c v_c w_c delta(i_c, i) delta(j_c, j) delta(k_c, k) = 1
        
        This expands to a system of equations. Since the supports are fixed and sparse,
        the variables are decoupled in a way that allows for efficient solving.
        
        Actually, for monomial patterns, the equation for a specific (i,j,k) is:
        sum_{c : i_c=i, j_c=j, k_c=k} (u_c * v_c * w_c) = 1
        
        Let x_c = u_c * v_c * w_c. We need sum_{c in S_{ijk}} x_c = 1 for all (i,j,k).
        This is a linear system for x_c.
        However, we need to decompose x_c into u_c, v_c, w_c.
        If we assume u_c, v_c, w_c are from a small set (e.g., +/- 1, +/- 0.5), 
        we can search. But a more robust method is:
        
        1. Solve for x_c using least squares (or exact if possible).
        2. If a solution exists, check if it can be factored.
        
        A better approach for "Strassen-like" structures:
        Often the coefficients are integers or half-integers.
        We can set up a linear system for the products P_c = u_c v_c w_c.
        Let A be the incidence matrix where A[t, c] = 1 if term c covers (i,j,k) at index t.
        We want A @ P = 1.
        
        If the system A @ P = 1 is consistent, we have candidate products P_c.
        We then need to find u_c, v_c, w_c such that u_c v_c w_c = P_c.
        This is always possible over reals (e.g., u=sqrt(P), v=sqrt(P), w=1/sqrt(P)), 
        but we need integer/half-integer entries.
        
        So, we look for integer/half-integer solutions to u*v*w = P.
        If P is an integer or half-integer, we can often find such factors.
        
        Strategy:
        1. Generate candidate patterns (supports).
        2. Solve A @ P = 1 for P.
        3. Check if P_c allows for integer/half-integer factorization.
        4. If yes, construct U, V, W.
        """
        if time.time() > time_limit:
            return None
            
        # pattern is a list of (i, j, k) tuples of length rank
        # Build incidence matrix A (n*m*p, rank)
        num_eq = n * m * p
        A_rows = []
        A_cols = []
        A_vals = []
        
        for c, (i, j, k) in enumerate(pattern):
            idx = i * m * p + j * p + k
            A_rows.append(idx)
            A_cols.append(c)
            A_vals.append(1.0)
            
        A = csr_matrix((A_vals, (A_rows, A_cols)), shape=(num_eq, rank))
        b = np.ones(num_eq)
        
        # Solve A @ P = b
        # Use least squares as the system might be underdetermined or overdetermined
        try:
            P, residuals, rank_sol, s = lsqr(A, b, atol=1e-12, btol=1e-12, itermax=1000)
        except Exception:
            return None
            
        # Check residual
        if residuals is not None and residuals[0] > 1e-9:
            return None
            
        # Now we have P_c such that sum_{c in S_ijk} P_c = 1.
        # We need to find u_c, v_c, w_c in {integers, half-integers} such that u_c v_c w_c = P_c.
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Candidate factors for small numbers
        # Since P_c might be small, we check a grid of half-integers
        # P_c is likely +/- 1, +/- 0.5, etc.
        
        for c in range(rank):
            p_val = P[c]
            # If p_val is close to 0, skip? No, if p_val is 0, then u,v,w can be 0.
            # But usually in minimal rank decompositions, no term is zero.
            
            found_factors = False
            # Search space for u, v, w
            # Limit search to reasonable values
            # If P_c is an integer, factors are likely integers.
            # If P_c is x.5, factors might be half-integers.
            
            # Simple heuristic: try to set w_c = 1, and solve u_c * v_c = P_c.
            # Or w_c = 0.5, etc.
            
            # Let's try a small grid
            # Range: -4 to 4 step 0.5
            candidates = [x * 0.5 for x in range(-8, 9)]
            
            for u in candidates:
                if abs(u) < 1e-9:
                    continue
                # v * w = P_c / u
                target_vw = p_val / u
                for v in candidates:
                    if abs(v) < 1e-9:
                        continue
                    w = target_vw / v
                    # Check if w is a half-integer
                    if abs(w * 2 - round(w * 2)) < 1e-9:
                        w = round(w * 2) / 2
                        # Verify
                        if abs(u * v * w - p_val) < 1e-9:
                            i, j, k = pattern[c]
                            U[i * m + j, c] = u
                            V[j * p + k, c] = v
                            W[k * n + i, c] = w
                            found_factors = True
                            break
                if found_factors:
                    break
            
            if not found_factors:
                # Try setting one variable to 1 to simplify?
                # If P_c is integer, u=1, v=1, w=P_c works if P_c is half-integer.
                if abs(p_val * 2 - round(p_val * 2)) < 1e-9:
                    i, j, k = pattern[c]
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = p_val
                    found_factors = True
                else:
                    return None # Cannot factorize with half-integers easily
            
        if found_factors:
            # Final verification
            if verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return U, V, W
                
        return None

    def generate_patterns(rank, n, m, p, time_limit):
        """
        Generate candidate patterns for given rank.
        We focus on patterns where each (i,j,k) is covered by exactly 1 term?
        No, that's rank 72.
        We want overlapping terms.
        
        Heuristic: 
        1. Start with schoolbook (rank 72).
        2. Try to merge terms.
        3. Randomly sample subsets of the full index set, ensuring coverage.
        
        A better heuristic for Strassen-like:
        Look for patterns where the "weight" of each index is balanced.
        
        Let's use a randomized greedy construction of patterns.
        1. Initialize a set of covered indices with count 0.
        2. Add 'rank' random (i,j,k) tuples.
        3. Check if every (i,j,k) is covered at least once.
        4. If yes, yield this pattern.
        """
        if time.time() > time_limit:
            return
        
        # Total indices
        total_indices = n * m * p
        # We need to cover all total_indices with 'rank' terms.
        # If rank < total_indices, some terms must cover multiple? 
        # No, in monomial patterns, each term covers exactly one (i,j,k).
        # So rank must be >= total_indices? 
        # WAIT. 
        # The tensor T_{ijk} = 1.
        # If term c is monomial at (i_c, j_c, k_c), it contributes to T_{i_c, j_c, k_c} only.
        # So sum_{c: i_c=i, j_c=j, k_c=k} P_c = 1.
        # This requires that for every (i,j,k), there is at least one c such that (i_c, j_c, k_c) == (i,j,k).
        # Therefore, the set of patterns {(i_c, j_c, k_c)} must be a superset of all (i,j,k).
        # Since there are n*m*p distinct (i,j,k) pairs, and each term c corresponds to exactly one pair,
        # we MUST have rank >= n*m*p.
        
        # THIS MEANS MONOMIAL PATTERNS CANNOT BEAT SCHOOLBOOK RANK.
        # Strassen uses NON-monomial patterns.
        # In Strassen, a term c has u_c[i,j] non-zero for multiple (i,j), etc.
        
        # So my "monomial" assumption is wrong for beating schoolbook.
        # I need to allow "block" or "dense" patterns.
        
        # Revised Strategy:
        # Use patterns where each term c has a small support size.
        # E.g., support size 2 or 4.
        # This is a combinatorial explosion.
        
        # Alternative: Use the "Strassen-like" structure directly.
        # For 2x2x2, rank 7.
        # For 3x4x6, we can pad to 4x4x8? Or 8x8x8?
        # Or use block decomposition.
        
        # Let's try a different approach:
        # Use the known Strassen decomposition for 2x2x2 and extend it.
        # Or, use a generic "sparse random" initialization but with a smarter solver.
        
        pass

    # Since monomial patterns don't work, let's try a different structural approach.
    # We will use a "Block Strassen" approach.
    # Decompose 3x4x6 into blocks.
    # 3x4 can be seen as blocks of 2x2 and 1x2?
    # 4x6 can be seen as blocks of 2x2?
    
    # Let's try to construct a rank 60-70 solution using a deterministic search over "Strassen-like" templates.
    # Template: A set of terms where each term is a product of sums of a few basis vectors.
    
    # Given the time, I will implement a specific heuristic:
    # 1. Try to find a decomposition by solving a linear system for a FIXED structure.
    # Structure: Each term c has u_c, v_c, w_c with at most 2 non-zero entries.
    # This is still hard.
    
    # Let's fall back to a more robust numerical method: 
    # "Tensor Train" or "MPS" decomposition? 
    # Or just better ALS with multiple restarts and annealing.
    
    # The previous program used ALS but maybe it got stuck in local minima.
    # I will implement a "Simulated Annealing" style search on the RANK itself, 
    # using a penalty for large values, and trying to push the solution towards sparsity.
    
    def optimize_with_sa(rank, seed, time_limit):
        """
        Simulated Annealing / Gradient Descent hybrid for tensor decomposition.
        Objective: Minimize || T - sum U V W ||^2 + lambda * ||U||_0 + ...
        But L0 is non-differentiable.
        
        Instead, we minimize the error, and periodically prune small coefficients.
        """
        if time.time() > time_limit:
            return None
            
        np.random.seed(seed)
        
        # Initialize with random sparse vectors
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Randomly assign non-zero entries
        for c in range(rank):
            # Pick random indices
            ui = np.random.randint(0, n * m)
            vj = np.random.randint(0, m * p)
            wk = np.random.randint(0, p * n)
            U[ui, c] = 1.0
            V[vj, c] = 1.0
            W[wk, c] = 1.0
            
        # Scale
        U *= 0.5
        V *= 0.5
        W *= 0.5
        
        # Gradient Descent with restarts
        lr = 0.01
        for it in range(200):
            if time.time() > time_limit:
                return None
                
            # Compute gradient
            # Error E_ijk = 1 - sum_c U_ic V_jc W_kc
            # dE/du_ic = - V_jc W_kc (summed over j,k? No, E depends on i,j,k)
            # We want to minimize sum_{ijk} E_ijk^2
            # d/dU_ic sum_{j,k} (1 - sum_c' U_i'c' V_jc' W_kc')^2
            # = -2 sum_{j,k} E_ijk V_jc W_kc
            
            # This is expensive to compute naively.
            # Use ALS instead, it's more stable.
            pass
            
        return None

    # Since complex heuristics are risky, I will stick to the ALS approach 
    # but with a much more aggressive initialization and "sparsification" step.
    # Key insight: The optimal solution for matrix multiplication tensors is often 
    # "generic" (dense) but can be approximated by sparse ones if the rank is high enough.
    # However, we want MINIMAL rank.
    
    # Let's try a specific known construction for 3x4x6.
    # If we can't find one, we return the schoolbook.
    
    # Actually, there is a known result: Rank(3x4x6) <= 64?
    # Let's try to find a rank 64 solution using a structured search.
    
    # I will implement a simple "Genetic Algorithm" on the support patterns.
    
    def evolve_patterns(pop_size, gen_count, time_limit):
        if time.time() > time_limit:
            return None
            
        # Representation: List of (i,j,k) for each term? 
        # No, we need to allow non-monomial.
        # Let's represent each term as a tuple of (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals).
        # This is too complex for a GA in 100s.
        
        # Alternative: Use the "Strassen" decomposition for 2x2x2 and embed it.
        # 3x4x6 can be embedded in 4x4x8?
        # 4x4x8 rank is 4^3 * 8 / something?
        
        # Let's try a simple "Block" decomposition.
        # Split 3x4 into [2x4] and [1x4].
        # Split 4x6 into [4x4] and [4x2].
        # This doesn't help directly.
        
        # I will return the schoolbook if I can't find a better one quickly.
        # But I must try.
        
        # Let's try rank 70 with ALS and multiple seeds.
        pass

    # Given the constraints, I will use the ALS method from the seed program 
    # but improve the initialization by using "Strassen-like" random blocks.
    
    def optimize_with_als_improved(rank, seed, time_limit):
        if time.time() > time_limit:
            return None
            
        np.random.seed(seed)
        
        # Initialize with random values, but add a "Strassen" bias:
        # Some terms will have 0s in specific patterns.
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # ALS Steps
        def solve_U(V, W):
            num_eq = n * m * p
            num_unk = n * m * rank
            A = np.zeros((num_eq, num_unk))
            b = np.ones(num_eq)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_base = i * m + j
                        for c in range(rank):
                            A[eq_idx, u_base * rank + c] = V[j * p + k, c] * W[k * n + i, c]
                        eq_idx += 1
            try:
                U_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                return U_flat.reshape(n * m, rank)
            except:
                return None

        def solve_V(U, W):
            num_eq = n * m * p
            num_unk = m * p * rank
            A = np.zeros((num_eq, num_unk))
            b = np.ones(num_eq)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        v_base = j * p + k
                        for c in range(rank):
                            A[eq_idx, v_base * rank + c] = U[i * m + j, c] * W[k * n + i, c]
                        eq_idx += 1
            try:
                V_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                return V_flat.reshape(m * p, rank)
            except:
                return None

        def solve_W(U, V):
            num_eq = n * m * p
            num_unk = p * n * rank
            A = np.zeros((num_eq, num_unk))
            b = np.ones(num_eq)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        w_base = k * n + i
                        for c in range(rank):
                            A[eq_idx, w_base * rank + c] = U[i * m + j, c] * V[j * p + k, c]
                        eq_idx += 1
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                return W_flat.reshape(p * n, rank)
            except:
                return None

        for it in range(100):
            if time.time() > time_limit:
                return None
                
            U = solve_U(V, W)
            if U is None: continue
            V = solve_V(U, W)
            if V is None: continue
            W = solve_W(U, V)
            if W is None: continue
            
            # Check error
            errors = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        errors += (s - 1.0) ** 2
                        
            if errors < 1e-6:
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                    return U_rounded, V_rounded, W_rounded
                    
        return None

    # Try ranks from 71 down to 60
    ranks_to_try = list(range(71, 59, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        found = False
        for seed in [42, 123, 999, 7, 13, 55, 77, 88]:
            if time.time() > deadline:
                break
            if found:
                break
                
            t_limit = time.time() + 2  # 2s per attempt
            if t_limit > deadline:
                t_limit = deadline
                
            res = optimize_with_als_improved(r, seed, t_limit)
            if res is not None:
                U_r, V_r, W_r = res
                if r < best_rank:
                    best_U = U_r.tolist()
                    best_V = V_r.tolist()
                    best_W = W_r.tolist()
                    best_rank = r
                found = True
                break

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
