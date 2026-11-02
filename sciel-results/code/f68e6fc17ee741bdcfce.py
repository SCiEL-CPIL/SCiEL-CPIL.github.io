"""Greedy-Then-Exact Hybrid for covering design C(24,7,3)."""
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    """
    Construct a covering design: find as few k-element blocks of v points as possible
    such that every t-element subset is contained in at least one block.
    
    Strategy:
    1. Fast vectorized greedy heuristic to build a near-minimal covering.
    2. Identify residual uncovered t-subsets.
    3. Formulate completion as an ILP and solve with scipy.optimize.milp.
    """
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute indices for fast access
    # For each (t-1) subset S and point x not in S, store index of S U {x}
    # This allows O(1) lookup in the greedy step
    # Key: (tuple(sorted_S), x) -> index
    # But S is always size t-1.
    # We can precompute a dict: key = (s1, s2, ..., s_{t-1}, x) where s_i < s_{i+1} and x not in S.
    # Actually, simpler: just use the tsub_to_idx with sorted tuple.
    # To speed up, we precompute the list of (t-1) subsets for each possible block size?
    # No, we just optimize the inner loop.
    
    uncovered = np.ones(n_tsubs, dtype=bool)
    blocks = []
    
    # Helper to get indices of t-subsets in a block
    def get_block_tsub_indices(block):
        indices = []
        for ts in itertools.combinations(block, t):
            indices.append(tsub_to_idx[ts])
        return indices

    # Greedy construction
    max_greedy_iterations = 100  # Safety cap
    
    for _ in range(max_greedy_iterations):
        if not uncovered.any():
            break
        
        # Pick the first uncovered t-subset as a seed
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        seed_ts = all_tsubs[uncovered_indices[0]]
        block = list(seed_ts)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            best_x = None
            best_gain = -1
            # Optimization: Only consider points that might add new coverage
            for x in range(v):
                if x in block:
                    continue
                # Calculate gain
                # Gain is number of uncovered t-subsets formed by adding x to block
                # These are (S U {x}) where S is a (t-1)-subset of block
                gain = 0
                # Precomputed combinations of block of size t-1
                # Since block size is small (<7), itertools is fine
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(ts)
                    if idx is not None:
                        if uncovered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for idx in get_block_tsub_indices(block):
            uncovered[idx] = False
    
    # Check if we've covered everything
    if not uncovered.any():
        return blocks
    
    # ILP completion phase
    # We need to cover the remaining uncovered t-subsets with additional blocks.
    
    uncovered_indices = np.where(uncovered)[0]
    uncovered_set = set(all_tsubs[i] for i in uncovered_indices)
    
    if len(uncovered_set) == 0:
        return blocks

    # Generate candidate blocks
    # To avoid explosion, we limit the number of candidates.
    # We generate candidates by extending each uncovered t-subset with random extensions.
    # We also include some "greedy" extensions.
    
    candidate_blocks = set()
    existing_blocks = set(tuple(b) for b in blocks)
    
    # Limit total candidates to ~5000 to keep ILP fast
    max_candidates = 5000
    
    # 1. Add blocks from a greedy pass on the residual
    # This ensures we have good candidates
    residual_uncovered = list(uncovered_set)
    # Shuffle to avoid bias? Or just pick first few
    # Let's just do a quick greedy on the residual to find some blocks
    temp_uncovered = np.zeros(n_tsubs, dtype=bool)
    for ts in uncovered_set:
        temp_uncovered[tsub_to_idx[ts]] = True
        
    # Quick greedy on residual
    for _ in range(50): # Max 50 blocks from this phase
        if not temp_uncovered.any():
            break
        # Pick first uncovered
        idx = np.where(temp_uncovered)[0][0]
        seed_ts = all_tsubs[idx]
        block = list(seed_ts)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    i = tsub_to_idx.get(ts)
                    if i is not None and temp_uncovered[i]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None: break
            block.append(best_x)
        
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in existing_blocks:
            candidate_blocks.add(b_tuple)
            # Update temp_uncovered
            for ts in itertools.combinations(block, t):
                temp_uncovered[tsub_to_idx[ts]] = False
        
        if len(candidate_blocks) > max_candidates:
            break

    # 2. Add random extensions for remaining uncovered t-subsets
    # If we still have uncovered, try to cover them with random blocks
    # But we already added some. Let's add a few random ones to ensure coverage in ILP
    # Actually, the greedy above should cover most.
    # If not, we might need more candidates.
    
    # Filter candidates
    useful_candidates = []
    for block in candidate_blocks:
        if block in existing_blocks:
            continue
        # Check if covers any *original* uncovered
        covers_any = False
        for ts in itertools.combinations(block, t):
            if ts in uncovered_set:
                covers_any = True
                break
        if covers_any:
            useful_candidates.append(block)
            
    # If we have too many, sample
    if len(useful_candidates) > max_candidates:
        import random
        random.shuffle(useful_candidates)
        useful_candidates = useful_candidates[:max_candidates]
    
    n_candidates = len(useful_candidates)
    
    if n_candidates == 0:
        # Fallback: just add greedy blocks directly
        return blocks
    
    # Build the constraint matrix
    # For each uncovered t-subset, we need at least one block to cover it
    
    uncovered_list = list(uncovered_set)
    n_uncovered = len(uncovered_list)
    ts_to_uncovered_idx = {ts: i for i, ts in enumerate(uncovered_list)}
    
    # Build A matrix (n_uncovered x n_candidates)
    A = np.zeros((n_uncovered, n_candidates), dtype=np.int8)
    for j, block in enumerate(useful_candidates):
        for ts in itertools.combinations(block, t):
            if ts in ts_to_uncovered_idx:
                A[ts_to_uncovered_idx[ts], j] = 1
    
    # ILP: minimize sum(x_j) subject to A @ x >= 1, 0 <= x_j <= 1, x_j integer
    c = np.ones(n_candidates, dtype=float)
    
    # Constraints: A @ x >= 1
    constraint = LinearConstraint(A, lb=np.ones(n_uncovered), ub=np.inf)
    
    # Bounds: 0 <= x_j <= 1
    bounds = Bounds(lb=np.zeros(n_candidates), ub=np.ones(n_candidates))
    
    # Integrality: all variables are integer
    integrality = np.ones(n_candidates, dtype=np.int8)
    
    # Solve
    solved = False
    try:
        # Set a time limit for the solver to avoid hanging
        # scipy milp doesn't have a direct time limit in older versions, 
        # but we can rely on the problem size being small.
        result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if result.status == 0 and result.x is not None:
            # Extract selected blocks
            selected = np.where(result.x > 0.5)[0]
            for j in selected:
                blocks.append(list(useful_candidates[j]))
            
            # Verify coverage
            all_covered = np.zeros(n_tsubs, dtype=bool)
            for block in blocks:
                for ts in itertools.combinations(block, t):
                    all_covered[tsub_to_idx[ts]] = True
            
            if all_covered.all():
                return blocks
            else:
                solved = True # ILP ran, but maybe incomplete due to candidate limitation
        else:
            solved = False
    except Exception:
        solved = False
    
    # Fallback: greedy completion for any remaining uncovered t-subsets
    # Rebuild uncovered status
    covered_set = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered_set.add(ts)
    
    # Find remaining uncovered
    still_uncovered = [ts for ts in all_tsubs if ts not in covered_set]
    
    # Greedily cover them
    # Use a simple greedy: pick block covering most remaining
    remaining = set(still_uncovered)
    while remaining:
        # Pick first one
        seed_ts = next(iter(remaining))
        block = list(seed_ts)
        
        # Extend greedily
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in remaining:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None: break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update remaining
        for ts in itertools.combinations(block, t):
            remaining.discard(ts)
            
    return blocks
