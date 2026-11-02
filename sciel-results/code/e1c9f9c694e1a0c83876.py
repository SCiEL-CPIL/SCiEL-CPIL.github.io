import itertools
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute all t-subsets and map to indices
    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    n_t_subs = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Track coverage count for each t-subset
    coverage = np.zeros(n_t_subs, dtype=np.int32)
    
    blocks = []
    
    def block_t_indices(block):
        """Get indices of all t-subsets contained in a block."""
        return [t_sub_to_idx[tri] for tri in itertools.combinations(block, t)]
    
    def get_block_t_indices_array(block):
        """Get numpy array of t-subset indices for a block."""
        return np.array([t_sub_to_idx[tri] for tri in itertools.combinations(block, t)], dtype=np.int64)
    
    # Vectorized greedy construction
    # Precompute all possible k-subsets? No, too many. Use greedy with random targets.
    
    # Phase 1: Fast greedy to get a good initial covering
    while time.time() < deadline - 30.0:
        uncovered_mask = coverage == 0
        n_uncovered = np.count_nonzero(uncovered_mask)
        if n_uncovered == 0:
            break
        
        # Pick a random uncovered t-subset as target
        uncovered_indices = np.where(uncovered_mask)[0]
        target_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
        target = all_t_subs[target_idx]
        target_set = set(target)
        
        # Generate candidate blocks containing the target
        remaining_points = [p for p in all_points if p not in target_set]
        candidates = list(itertools.combinations(remaining_points, k - t))
        
        if len(candidates) > 3000:
            random.shuffle(candidates)
            candidates = candidates[:3000]
        
        best_block = None
        max_new_cover = -1
        
        for combo in candidates:
            block = tuple(sorted(target + combo))
            
            # Count how many currently uncovered t-subsets this block covers
            new_cover = 0
            for tri in itertools.combinations(block, t):
                if coverage[t_sub_to_idx[tri]] == 0:
                    new_cover += 1
            
            if new_cover > max_new_cover:
                max_new_cover = new_cover
                best_block = block
        
        if best_block is None:
            break
        
        # Add the block
        blocks.append(list(best_block))
        
        # Update coverage
        for tri in itertools.combinations(best_block, t):
            coverage[t_sub_to_idx[tri]] += 1
    
    # Check if fully covered
    if np.all(coverage > 0):
        # Try redundancy removal
        blocks = _remove_redundant_blocks(blocks, v, k, t, deadline)
        return blocks
    
    # Phase 2: ILP to cover remaining uncovered t-subsets
    if time.time() < deadline - 5.0:
        blocks = _solve_residual_ilp(blocks, v, k, t, coverage, t_sub_to_idx, all_t_subs, deadline)
    
    # Final redundancy removal
    blocks = _remove_redundant_blocks(blocks, v, k, t, deadline)
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t, deadline):
    """Remove redundant blocks using vectorized operations."""
    if not blocks:
        return blocks
    
    n_blocks = len(blocks)
    
    # Build coverage matrix: n_blocks x n_t_subs (sparse)
    # For each block, which t-subsets does it cover?
    all_t_subs_count = len(list(itertools.combinations(range(v), t)))
    
    # Precompute t-subset indices for each block
    block_t_idx_list = []
    for b in blocks:
        idxs = [sum(1 for _ in ()) for _ in range(0)]  # placeholder
        idxs = []
        for tri in itertools.combinations(b, t):
            # Need global index
            pass
        block_t_idx_list.append(idxs)
    
    # Simpler approach: iterative removal
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        new_blocks = []
        for i in range(len(blocks)):
            # Check if block i is redundant
            redundant = True
            for tri in itertools.combinations(blocks[i], t):
                found = False
                for j in range(len(blocks)):
                    if i == j:
                        continue
                    if set(tri).issubset(set(blocks[j])):
                        found = True
                        break
                if not found:
                    redundant = False
                    break
            
            if not redundant:
                new_blocks.append(blocks[i])
        
        if len(new_blocks) < len(blocks):
            blocks = new_blocks
            changed = True
    
    return blocks


def _solve_residual_ilp(blocks, v, k, t, coverage, t_sub_to_idx, all_t_subs, deadline):
    """Use ILP to find minimal set of additional blocks to cover remaining uncovered t-subsets."""
    if time.time() > deadline - 5.0:
        return blocks
    
    # Find uncovered t-subsets
    uncovered_indices = np.where(coverage == 0)[0]
    if len(uncovered_indices) == 0:
        return blocks
    
    n_uncovered = len(uncovered_indices)
    
    # Generate candidate blocks that cover at least one uncovered t-subset
    # For efficiency, limit candidates
    candidate_blocks = []
    seen_blocks = set()
    
    # For each uncovered t-subset, generate some candidate blocks
    max_candidates = 500
    for idx in uncovered_indices[:min(50, n_uncovered)]:
        target = all_t_subs[idx]
        target_set = set(target)
        remaining_points = [p for p in range(v) if p not in target_set]
        
        # Generate a few random candidate blocks containing this target
        n_candidates = max(1, max_candidates // min(50, n_uncovered))
        for _ in range(n_candidates):
            if len(candidate_blocks) >= max_candidates:
                break
            combo = random.sample(remaining_points, k - t)
            block = tuple(sorted(target + combo))
            if block not in seen_blocks:
                seen_blocks.add(block)
                candidate_blocks.append(block)
    
    if not candidate_blocks:
        return blocks
    
    n_candidates = len(candidate_blocks)
    
    # Build constraint matrix: for each uncovered t-subset, which candidate blocks cover it
    # A[i, j] = 1 if candidate block j covers uncovered t-subset i
    A = np.zeros((n_uncovered, n_candidates), dtype=np.float64)
    
    for j, block in enumerate(candidate_blocks):
        block_set = set(block)
        for i, idx in enumerate(uncovered_indices):
            tri = all_t_subs[idx]
            if all(p in block_set for p in tri):
                A[i, j] = 1.0
    
    # ILP: minimize sum(x_j) subject to A @ x >= 1 for all i, x_j in {0, 1}
    c = np.ones(n_candidates, dtype=np.float64)
    
    # Constraints: A @ x >= 1
    constraint = LinearConstraint(A, 1.0, np.inf)
    integrality = np.ones(n_candidates, dtype=np.int8)
    bounds = Bounds(0, 1)
    
    try:
        result = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds)
        
        if result.status == 0 and result.x is not None:
            # Add selected blocks
            for j in range(n_candidates):
                if result.x[j] > 0.5:
                    blocks.append(list(candidate_blocks[j]))
    except Exception:
        pass
    
    return blocks
