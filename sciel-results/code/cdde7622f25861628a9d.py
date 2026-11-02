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
    
    # Precompute for each point x and each (t-1)-subset, which t-subsets contain both
    # This is used in the greedy step
    # For speed, we'll track uncovered as a boolean array
    
    uncovered = np.ones(n_tsubs, dtype=bool)
    blocks = []
    
    # Greedy phase: iteratively pick a block that covers the most uncovered t-subsets
    # We use a vectorized approach where possible
    
    def count_covered(block_set, x):
        """Count how many uncovered t-subsets would be covered by adding x to block_set."""
        block_list = list(block_set)
        if len(block_list) < t - 1:
            return 0
        # For each (t-1)-subset of block_list, the t-subset formed by adding x
        count = 0
        for rest in itertools.combinations(block_list, t - 1):
            ts = tuple(sorted(rest + (x,)))
            idx = tsub_to_idx.get(ts)
            if idx is not None and uncovered[idx]:
                count += 1
        return count
    
    # Greedy construction
    max_greedy_iterations = 200  # Safety cap; we expect ~76 blocks for C(24,7,3)
    
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
            for x in range(v):
                if x in block:
                    continue
                gain = count_covered(block, x)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            uncovered[idx] = False
    
    # Check if we've covered everything
    if not uncovered.any():
        return blocks
    
    # ILP completion phase
    # We need to cover the remaining uncovered t-subsets with additional blocks.
    # Let's find all possible blocks (k-subsets) that cover at least one uncovered t-subset.
    
    uncovered_indices = np.where(uncovered)[0]
    uncovered_set = set(all_tsubs[i] for i in uncovered_indices)
    
    # Generate candidate blocks: k-subsets that contain at least one uncovered t-subset
    # To keep the ILP small, we only consider blocks that are "useful"
    # A block is useful if it covers at least one uncovered t-subset
    
    # For efficiency, let's generate candidates from the uncovered t-subsets
    # Each uncovered t-subset can be extended to a k-subset in C(v-t, k-t) ways
    # But many will be duplicates. Let's collect them in a set.
    
    candidate_blocks = set()
    for ts in uncovered_set:
        remaining_points = [x for x in range(v) if x not in ts]
        # Need to choose k-t more points from remaining_points
        for ext in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(ts + ext))
            # Check if this block actually covers any uncovered t-subset
            # (It should, since it contains ts, but let's be safe)
            candidate_blocks.add(block)
    
    # Filter: keep only blocks that cover at least one uncovered t-subset
    # And also check they don't duplicate existing blocks
    existing_blocks = set(tuple(b) for b in blocks)
    
    useful_candidates = []
    for block in candidate_blocks:
        if block in existing_blocks:
            continue
        # Check if this block covers any uncovered t-subset
        covers_any = False
        for ts in itertools.combinations(block, t):
            if ts in uncovered_set:
                covers_any = True
                break
        if covers_any:
            useful_candidates.append(block)
    
    n_candidates = len(useful_candidates)
    
    if n_candidates == 0:
        # Should not happen, but just in case
        return blocks
    
    # Build the constraint matrix
    # For each uncovered t-subset, we need at least one block to cover it
    # x_j = 1 if we use candidate block j
    # For each uncovered t-subset i: sum_{j: block j covers ts i} x_j >= 1
    
    # Map uncovered t-subsets to indices
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
    # In scipy milp, constraints are of the form lb <= A @ x <= ub
    # So we set lb = 1, ub = inf
    constraint = LinearConstraint(A, lb=np.ones(n_uncovered), ub=np.inf)
    
    # Bounds: 0 <= x_j <= 1
    bounds = Bounds(lb=np.zeros(n_candidates), ub=np.ones(n_candidates))
    
    # Integrality: all variables are integer
    integrality = np.ones(n_candidates, dtype=np.int8)
    
    # Solve
    try:
        result = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if result.status == 0 and result.x is not None:
            # Extract selected blocks
            selected = np.where(result.x > 0.5)[0]
            for j in selected:
                blocks.append(list(useful_candidates[j]))
            
            # Verify coverage
            # Re-check that all t-subsets are covered
            all_covered = np.zeros(n_tsubs, dtype=bool)
            for block in blocks:
                for ts in itertools.combinations(block, t):
                    all_covered[tsub_to_idx[ts]] = True
            
            if all_covered.all():
                return blocks
            else:
                # Fallback: add more blocks greedily for any remaining
                # This shouldn't happen but let's be safe
                still_uncovered = set(all_tsubs[i] for i in range(n_tsubs) if not all_covered[i])
                for ts in still_uncovered:
                    # Find a block containing ts
                    found = False
                    for block in blocks:
                        if all(x in block for x in ts):
                            found = True
                            break
                    if not found:
                        # Create a new block containing ts
                        new_block = list(ts)
                        for x in range(v):
                            if x not in new_block:
                                new_block.append(x)
                            if len(new_block) == k:
                                break
                        new_block.sort()
                        blocks.append(new_block)
                return blocks
        else:
            # ILP failed, fall back to greedy for the rest
            pass
    except Exception:
        # ILP failed, fall back to greedy
        pass
    
    # Fallback: greedy completion
    uncovered_final = set(all_tsubs[i] for i in range(n_tsubs) if not uncovered[i] is False)
    # Rebuild uncovered from scratch
    covered_set = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered_set.add(ts)
    
    for ts in all_tsubs:
        if ts not in covered_set:
            # Add a block containing ts
            new_block = list(ts)
            for x in range(v):
                if x not in new_block:
                    new_block.append(x)
                if len(new_block) == k:
                    break
            new_block.sort()
            blocks.append(new_block)
    
    return blocks
