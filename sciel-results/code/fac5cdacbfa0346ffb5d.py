import time
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def _build_critical_aps(n, min_len=4):
    """
    Identify the most critical Arithmetic Progressions (APs).
    We focus on APs with length >= min_len, as short APs (length 1-3) have
    bounded discrepancy (<=3) and are easier to satisfy.
    We return the indices for these APs and a mapping from index to affected APs.
    """
    ap_indices = []
    index_to_aps = [[] for _ in range(n)]
    
    ap_id = 0
    for step in range(1, n + 1):
        for start in range(n):
            indices = np.arange(start, n, step, dtype=np.int32)
            if len(indices) < min_len:
                continue
            
            ap_indices.append(indices)
            
            for pos, idx in enumerate(indices):
                index_to_aps[idx].append((ap_id, pos))
            
            ap_id += 1
            
    return ap_indices, index_to_aps

def _solve_admm_bqp(n, ap_indices, deadline):
    """
    Solve the Binary Quadratic Program using ADMM.
    
    Objective: Minimize sum_{ap} (sum_{i in ap} x_i)^2
    Subject to: x_i in {-1, 1}
    
    We relax x_i to R and use ADMM with the constraint set C = {-1, 1}^n.
    The update for x involves solving a linear system:
    (2 * A^T A + rho * I) x = 2 * A^T (z - u)
    where A is the AP incidence matrix (rows are APs, cols are indices).
    """
    num_aps = len(ap_indices)
    
    if num_aps == 0:
        return np.ones(n)
        
    # Build sparse incidence matrix A
    # A[k, i] = 1 if index i is in AP k
    A = lil_matrix((num_aps, n), dtype=np.float64)
    for k, indices in enumerate(ap_indices):
        for idx in indices:
            A[k, idx] = 1.0
            
    A = A.tocsr()
    
    # Pre-compute the linear system matrix M = 2 * A^T A + rho * I
    # We use a fixed rho for simplicity, or adaptive. Fixed is safer for speed.
    rho = 10.0 
    AT = A.T
    ATA = (AT @ A).toarray()
    M = 2.0 * ATA + rho * np.eye(n)
    
    # Pre-compute Cholesky factorization for fast solving
    # M is symmetric positive definite
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        # Fallback to LU if Cholesky fails (should not happen)
        from scipy.linalg import lu
        lu_piv, lu_l, lu_u = lu(M)
        use_lu = True
    else:
        use_lu = False
        
    # Initialize
    x = np.zeros(n)
    z = np.ones(n) # Start with all +1
    u = np.zeros(n)
    
    # ADMM Iterations
    max_iters = 100
    start_time = time.time()
    
    # Pre-compute 2 * A^T for the RHS calculation
    # RHS = 2 * A^T (z - u)
    
    for _ in range(max_iters):
        if time.time() > deadline - 2:
            break
            
        # 1. x-update: Solve M x_new = 2 * A^T (z - u)
        rhs = 2.0 * (AT @ (z - u))
        
        if not use_lu:
            # Solve L y = rhs, then L^T x = y
            y = np.linalg.solve(L, rhs)
            x_new = np.linalg.solve(L.T, y)
        else:
            x_new = np.linalg.solve(M, rhs)
            
        # 2. z-update: Projection onto {-1, 1}
        # z_new = sign(x_new + u)
        z_new = np.sign(x_new + u)
        # Handle zeros in sign: if x_new + u == 0, sign is 0. 
        # We need {-1, 1}. We can break ties randomly or based on previous z.
        # Let's keep previous z for zeros to maintain stability, or flip randomly.
        # For BQP, sign(0) is undefined. Let's use: if val >= 0 then 1 else -1.
        z_new = np.where(x_new + u >= 0, 1, -1)
        
        # 3. u-update
        u = u + x_new - z_new
        
        # Check convergence
        # If ||x_new - z_new|| is small, we might be converged
        # But we also want to ensure x is close to binary.
        # For now, run fixed iterations.
        
        x = x_new
        z = z_new
        
    # Final binary solution
    x_final = np.where(x + u >= 0, 1, -1)
    
    return x_final

def _local_search(signs, ap_indices, index_to_aps, deadline):
    """
    Fast local search to polish the integer solution.
    Uses vectorized numpy operations for speed.
    """
    n = len(signs)
    signs = np.array(signs, dtype=np.int8)
    
    # Compute initial prefix sums and max disc for each AP
    # We store the full prefix sum array for each AP to allow O(1) flip evaluation?
    # No, flip affects suffix.
    # Let's store the prefix sums.
    prefix_sums = []
    for indices in ap_indices:
        subseq = signs[indices]
        ps = np.cumsum(subseq)
        prefix_sums.append(ps)
        
    ap_max_disc = np.array([np.max(np.abs(ps)) for ps in prefix_sums], dtype=np.int64)
    current_global_max = np.max(ap_max_disc) if len(ap_max_disc) > 0 else 0
    
    best_signs = signs.copy()
    best_global_max = current_global_max
    
    max_ls_iters = 50
    ls_iter = 0
    rng = np.random.default_rng(42)
    
    while ls_iter < max_ls_iters and time.time() < deadline - 2:
        improved = False
        indices_to_try = list(range(n))
        rng.shuffle(indices_to_try)
        
        for i in indices_to_try:
            if time.time() > deadline - 2:
                break
                
            old_val = signs[i]
            delta = -2 * old_val
            affected_aps = index_to_aps[i]
            
            # Evaluate the flip
            # We need to check if flipping i reduces the max discrepancy.
            # A flip at position i in AP k changes the prefix sums from pos_i onwards by delta.
            # New max for AP k is max(max_prefix_before_i, max(|suffix_after_i + delta|))
            
            max_new_aff = 0
            feasible = True
            
            for ap_id, pos in affected_aps:
                ps = prefix_sums[ap_id]
                if pos > 0:
                    max_pref = np.max(np.abs(ps[:pos]))
                else:
                    max_pref = 0
                
                # Suffix part
                suffix = ps[pos:]
                new_suffix = suffix + delta
                max_suff = np.max(np.abs(new_suffix))
                
                new_ap_max = max(max_pref, max_suff)
                
                if new_ap_max > current_global_max:
                    feasible = False
                    break
                
                if new_ap_max > max_new_aff:
                    max_new_aff = new_ap_max
            
            if not feasible:
                continue
                
            # Apply flip tentatively
            signs[i] = -old_val
            for ap_id, pos in affected_aps:
                prefix_sums[ap_id][pos:] += delta
                new_m = np.max(np.abs(prefix_sums[ap_id]))
                ap_max_disc[ap_id] = new_m
                
            new_global_max = np.max(ap_max_disc)
            
            if new_global_max < current_global_max:
                current_global_max = new_global_max
                improved = True
                if current_global_max < best_global_max:
                    best_global_max = current_global_max
                    best_signs = signs.copy()
            else:
                # Revert
                signs[i] = old_val
                for ap_id, pos in affected_aps:
                    prefix_sums[ap_id][pos:] -= delta
                    ap_max_disc[ap_id] = np.max(np.abs(prefix_sums[ap_id]))
                    
        ls_iter += 1
        if not improved:
            break
            
    return best_signs, best_global_max

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # 1. Pre-compute structure
    # We use min_len=4 to focus on the harder constraints.
    # Singletons (len 1) have sum +/-1. Pairs (len 2) have sum -2,0,2. Triples -3..3.
    # The bottleneck is usually longer APs.
    ap_indices, index_to_aps = _build_critical_aps(n, min_len=4)
    
    best_signs = None
    best_score = float('inf')
    
    # 2. Solve ADMM BQP
    if time.time() < deadline - 10:
        x_admm = _solve_admm_bqp(n, ap_indices, deadline)
        
        # Polish with local search
        polished_signs, polished_score = _local_search(x_admm, ap_indices, index_to_aps, deadline)
        
        # Note: _local_search returns score based on ap_indices (len >= 4).
        # We need to verify the score on ALL APs including short ones.
        # However, short APs are bounded by their length, so if max_disc for long APs is D,
        # the global max is max(D, 3).
        # Since we are minimizing D, and D is likely > 3, this is a good proxy.
        # But to be safe, let's calculate the true score on all APs for the final candidate.
        
        if polished_score < best_score:
            best_score = polished_score
            best_signs = polished_signs
    
    # 3. Fallback / Multi-start if ADMM failed or time remains
    if time.time() < deadline - 15 or best_signs is None:
        # Use Legendre symbol as a strong heuristic start
        def _generate_legendre_sequence(n):
            def is_prime(k):
                if k < 2: return False
                for i in range(2, int(np.sqrt(k)) + 1):
                    if k % i == 0: return False
                return True
            p = n
            while not is_prime(p): p += 1
            seq = []
            for i in range(1, n + 1):
                ls = pow(i, (p - 1) // 2, p)
                if ls == 0: seq.append(1)
                else: seq.append(1 if ls == 1 else -1)
            return seq
        
        leg_seq = _generate_legendre_sequence(n)
        polished_signs_leg, polished_score_leg = _local_search(leg_seq, ap_indices, index_to_aps, deadline)
        
        if polished_score_leg < best_score:
            best_score = polished_score_leg
            best_signs = polished_signs_leg
            
        # Random start
        rng = np.random.default_rng(0)
        rand_seq = rng.choice([-1, 1], size=n).astype(np.int8)
        polished_signs_r, polished_score_r = _local_search(rand_seq, ap_indices, index_to_aps, deadline)
        
        if polished_score_r < best_score:
            best_score = polished_score_r
            best_signs = polished_signs_r

    # 4. Final Verification and Return
    # The local search minimizes the max discrepancy over APs with len >= 4.
    # The true score is the max over ALL APs.
    # Since APs with len < 4 have max discrepancy <= 3, and we expect best_score > 3,
    # best_score is likely the true score.
    # However, if best_score <= 3, the true score might be 3 (from a singleton or pair).
    # Let's just return the signs. The evaluator will compute the true score.
    
    if best_signs is None:
        # Ultimate fallback
        rng = np.random.default_rng(123)
        best_signs = list(rng.choice([-1, 1], size=n))
        
    return list(best_signs)

# EVOLVE-BLOCK-END
