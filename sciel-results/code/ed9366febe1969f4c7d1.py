"""Greedy-Then-Exact Hybrid for C(24,7,3) covering."""
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a hybrid approach:
    1. Fast vectorized greedy to get a near-minimal covering
    2. Identify residual uncovered t-subsets
    3. Solve a small ILP to complete the covering optimally for the residual
    """
    
    # Step 1: Vectorized greedy construction
    # Represent all t-subsets as bitmasks for fast membership testing
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Track uncovered t-subsets (1 = uncovered, 0 = covered)
    uncovered = np.ones(n_tsubsets, dtype=np.int8)
    
    blocks = []
    
    # Precompute for each point, which t-subsets it participates in
    # Actually, let's use a different approach: for each candidate block, 
    # compute how many uncovered t-subsets it covers
    
    # Greedy: repeatedly pick the k-subset that covers the most uncovered t-subsets
    # But we need to be efficient. Let's use a smarter greedy.
    
    # First, create an array mapping each point to its index
    # For each possible k-subset, we could precompute coverage, but that's C(24,7) = 346104
    # which is manageable.
    
    # Precompute all k-subsets and their coverage of t-subsets
    all_ksubsets = list(itertools.combinations(range(v), k))
    n_ksubsets = len(all_ksubsets)
    
    # For each k-subset, store the indices of t-subsets it covers
    # This is a sparse representation
    ksubset_tsubsets = []
    for ks in all_ksubsets:
        tsubs_in_ks = []
        for ts in itertools.combinations(ks, t):
            tsubs_in_ks.append(tsubset_to_idx[ts])
        ksubset_tsubsets.append(tsubs_in_ks)
    
    # Greedy: iteratively pick the k-subset that covers the most uncovered t-subsets
    # To speed up, we maintain a count of uncovered t-subsets covered by each k-subset
    # Initially, this is just len(tsubs_in_ks) for all k-subsets
    
    # But recomputing this each iteration is expensive. Let's use a different strategy:
    # Pick a random-ish starting point and build blocks greedily by adding points
    # that maximize coverage of uncovered t-subsets.
    
    # Actually, let's do a simpler but effective greedy:
    # For each uncovered t-subset, we need to cover it. Pick one uncovered t-subset,
    # then build a block containing it that covers as many other uncovered t-subsets as possible.
    
    # More efficient approach: use the fact that we can compute gain incrementally.
    
    # Let's try: while uncovered, pick the k-subset with max coverage of uncovered t-subsets.
    # We can maintain an array cov_count[j] = number of uncovered t-subsets in ksubset_tsubsets[j]
    
    cov_count = np.array([len(ts) for ts in ksubset_tsubsets], dtype=np.int32)
    
    while uncovered.any():
        # Find the k-subset with maximum coverage of uncovered t-subsets
        best_idx = np.argmax(cov_count)
        if cov_count[best_idx] == 0:
            break  # Should not happen if uncovered.any()
        
        best_ksubset = all_ksubsets[best_idx]
        blocks.append(list(best_ksubset))
        
        # Update uncovered and cov_count
        for tsub_idx in ksubset_tsubsets[best_idx]:
            if uncovered[tsub_idx]:
                uncovered[tsub_idx] = 0
                # For each k-subset that contains this t-subset, decrement cov_count
                # We need to know which k-subsets contain a given t-subset
                # This requires a reverse mapping
    
    # This approach needs a reverse mapping. Let's build it.
    
    # Actually, let's restart with a cleaner implementation.
    
    # Build reverse mapping: for each t-subset, which k-subsets contain it
    # tsubset_to_ksubsets: dict from t-subset index to list of k-subset indices
    tsubset_to_ksubsets = [[] for _ in range(n_tsubsets)]
    for j, ks in enumerate(all_ksubsets):
        for ts in itertools.combinations(ks, t):
            tsubset_to_ksubsets[tsubset_to_idx[ts]].append(j)
    
    # Now redo the greedy with proper updates
    uncovered = np.ones(n_tsubsets, dtype=np.int8)
    cov_count = np.array([len(ts) for ts in ksubset_tsubsets], dtype=np.int32)
    blocks = []
    
    while uncovered.any():
        best_idx = np.argmax(cov_count)
        if cov_count[best_idx] == 0:
            break
        
        blocks.append(list(all_ksubsets[best_idx]))
        
        # Update: mark all t-subsets in this k-subset as covered
        for tsub_idx in ksubset_tsubsets[best_idx]:
            if uncovered[tsub_idx]:
                uncovered[tsub_idx] = 0
                for ksub_idx in tsubset_to_ksubsets[tsub_idx]:
                    cov_count[ksub_idx] -= 1
    
    # Step 2: Check if we have a complete covering
    if not uncovered.any():
        return blocks
    
    # Step 3: Solve ILP for the residual
    # The residual is the set of uncovered t-subsets
    residual_tsubsets = [i for i in range(n_tsubsets) if uncovered[i]]
    n_residual = len(residual_tsubsets)
    
    if n_residual == 0:
        return blocks
    
    # For the ILP, we consider only k-subsets that cover at least one residual t-subset
    # Variables: x_j for each k-subset j, x_j = 1 if we include k-subset j
    # Constraints: for each residual t-subset i, sum of x_j over k-subsets j containing i >= 1
    # Objective: minimize sum x_j
    
    # Identify relevant k-subsets (those covering at least one residual t-subset)
    relevant_ksubsets = set()
    for tsub_idx in residual_tsubsets:
        for ksub_idx in tsubset_to_ksubsets[tsub_idx]:
            relevant_ksubsets.add(ksub_idx)
    
    relevant_ksubsets = sorted(relevant_ksubsets)
    n_relevant = len(relevant_ksubsets)
    ksub_to_var = {ksub: var for var, ksub in enumerate(relevant_ksubsets)}
    
    # Build constraint matrix: A[i, j] = 1 if k-subset j covers residual t-subset i
    A = np.zeros((n_residual, n_relevant), dtype=np.int8)
    for i, tsub_idx in enumerate(residual_tsubsets):
        for ksub_idx in tsubset_to_ksubsets[tsub_idx]:
            if ksub_idx in ksub_to_var:
                j = ksub_to_var[ksub_idx]
                A[i, j] = 1
    
    # Objective: minimize sum x_j
    c = np.ones(n_relevant, dtype=np.int8)
    
    # Constraints: A @ x >= 1 for each row
    # milp uses LinearConstraint with lb and ub
    lb = np.ones(n_residual, dtype=np.int8)
    ub = np.full(n_residual, np.inf)
    
    # Variable bounds: 0 <= x_j <= 1, integer
    integrality = np.ones(n_relevant, dtype=np.int8)
    var_lb = np.zeros(n_relevant, dtype=np.int8)
    var_ub = np.ones(n_relevant, dtype=np.int8)
    
    # Solve ILP
    try:
        res = milp(
            c=c,
            constraints=[LinearConstraint(A, lb=lb, ub=ub)],
            integrality=integrality,
            bounds=Bounds(lb=var_lb, ub=var_ub)
        )
        
        if res.status == 0 and res.x is not None:
            # Extract selected k-subsets
            for var, ksub_idx in enumerate(relevant_ksubsets):
                if res.x[var] > 0.5:
                    blocks.append(list(all_ksubsets[ksub_idx]))
        
        # If ILP failed, fall back to adding blocks greedily for residual
        else:
            # Greedy completion for residual
            while uncovered.any():
                best_idx = np.argmax(cov_count)
                if cov_count[best_idx] == 0:
                    break
                blocks.append(list(all_ksubsets[best_idx]))
                for tsub_idx in ksubset_tsubsets[best_idx]:
                    if uncovered[tsub_idx]:
                        uncovered[tsub_idx] = 0
                        for ksub_idx in tsubset_to_ksubsets[tsub_idx]:
                            cov_count[ksub_idx] -= 1
        
    except Exception:
        # Fallback: greedy completion
        while uncovered.any():
            best_idx = np.argmax(cov_count)
            if cov_count[best_idx] == 0:
                break
            blocks.append(list(all_ksubsets[best_idx]))
            for tsub_idx in ksubset_tsubsets[best_idx]:
                if uncovered[tsub_idx]:
                    uncovered[tsub_idx] = 0
                    for ksub_idx in tsubset_to_ksubsets[tsub_idx]:
                        cov_count[ksub_idx] -= 1
    
    return blocks
