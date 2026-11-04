import itertools
import time
import random
from collections import Counter

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    import numpy as np
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_triples = len(all_tsubsets)
    
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Run multiple randomized greedy trials
    trial = 0
    while time.time() < deadline - 60.0:  # Reserve 60s for ILP phase
        trial += 1
        blocks, valid = _greedy_cover(v, k, t, all_tsubsets, deadline, trial)
        if valid:
            # Local improvement: try to remove redundant blocks
            improved_blocks = _local_improve(blocks, v, k, t, deadline)
            if len(improved_blocks) < len(blocks):
                blocks = improved_blocks
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
        if best_blocks and len(best_blocks) <= 76:
            break
    
    if best_blocks is None:
        # Fallback: run a deterministic greedy to ensure we return *something* valid
        blocks, valid = _greedy_cover(v, k, t, all_tsubsets, deadline, -1)
        if valid:
            best_blocks = blocks
            improved_blocks = _local_improve(best_blocks, v, k, t, deadline)
            if len(improved_blocks) < len(best_blocks):
                best_blocks = improved_blocks
        else:
            best_blocks = []
    
    # Phase 2: Try ILP optimization if scipy is available
    if HAS_SCIPY and best_blocks is not None and time.time() < deadline - 5.0:
        ilp_result = _solve_ilp(best_blocks, v, k, t, all_tsubsets, deadline)
        if ilp_result is not None and len(ilp_result) < len(best_blocks):
            best_blocks = ilp_result
    
    return best_blocks

def _greedy_cover(v, k, t, all_tsubsets, deadline, seed_val):
    """
    Greedy covering with randomized tie-breaking and ratio-based scoring.
    Optimized for speed using numpy where possible.
    """
    rng = random.Random(seed_val if seed_val >= 0 else time.time())
    
    uncovered = set(all_tsubsets)
    blocks = []
    
    points = list(range(v))
    
    while uncovered:
        if time.time() > deadline:
            return None, False
        
        # Pick a random uncovered subset to seed the block
        seed = rng.choice(list(uncovered))
        block = list(seed)
        
        while len(block) < k:
            if time.time() > deadline:
                return None, False
            
            block_set = set(block)
            best_x = -1
            best_score = -1
            best_tiebreakers = []
            
            # Precompute the (t-1)-subsets of the current block
            t_minus_1_subsets = list(itertools.combinations(block, t-1))
            
            for x in points:
                if x in block_set:
                    continue
                
                # Count how many new t-subsets this point would cover
                gain = 0
                for p in t_minus_1_subsets:
                    s = tuple(sorted(p + (x,)))
                    if s in uncovered:
                        gain += 1
                
                # Use a ratio-based score: gain / (number of points in block)
                # This penalizes adding points that don't contribute much
                if gain > 0:
                    score = gain / (len(block) + 1)
                else:
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_tiebreakers = [x]
                elif score == best_score and score > 0:
                    best_tiebreakers.append(x)
            
            if best_x == -1:
                # No point gives positive gain; pick a random one
                remaining = [x for x in points if x not in block_set]
                if not remaining:
                    return None, False
                chosen_x = rng.choice(remaining)
            else:
                chosen_x = rng.choice(best_tiebreakers)
            
            block.append(chosen_x)
        
        block.sort()
        blocks.append(tuple(block))
        
        for s in itertools.combinations(block, t):
            uncovered.discard(s)
            
    return blocks, True

def _local_improve(blocks, v, k, t, deadline):
    """
    Try to remove redundant blocks. A block is redundant if removing it
    doesn't leave any t-subset uncovered.
    Uses multiple passes to catch cascading redundancies.
    """
    if not blocks:
        return blocks
    
    improved = list(blocks)
    
    # Multiple passes to catch cascading redundancies
    for _pass in range(5):
        if time.time() > deadline:
            break
        
        # Build a count for each t-subset: how many blocks cover it
        cover_count = Counter()
        for block in improved:
            for s in itertools.combinations(block, t):
                cover_count[s] += 1
        
        # Try removing each block if it's redundant
        # We iterate backwards so that removal doesn't affect indices
        removed_any = False
        for i in range(len(improved) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Check if block i is redundant: all its t-subsets are covered by other blocks
            block = improved[i]
            redundant = True
            for s in itertools.combinations(block, t):
                if cover_count[s] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                for s in itertools.combinations(block, t):
                    cover_count[s] -= 1
                improved.pop(i)
                removed_any = True
        
        if not removed_any:
            break
    
    return improved

def _solve_ilp(initial_blocks, v, k, t, all_tsubsets, deadline):
    """
    Solve a reduced ILP to find a minimum subset of the initial_blocks
    that still covers all t-subsets.
    
    This is a set cover problem:
    - Variables: x_j in {0,1} for each candidate block j
    - Objective: minimize sum(x_j)
    - Constraints: for each triple s, sum_{j: s in block_j} x_j >= 1
    """
    if not initial_blocks:
        return None
    
    n_blocks = len(initial_blocks)
    n_triples = len(all_tsubsets)
    
    # Create a mapping from triple to index
    triple_to_idx = {triple: i for i, triple in enumerate(all_tsubsets)}
    
    # Build the constraint matrix A (n_triples x n_blocks)
    # A[i, j] = 1 if triple i is in block j, else 0
    A_rows = []
    for j, block in enumerate(initial_blocks):
        block_set = set(block)
        for s in itertools.combinations(block, t):
            idx = triple_to_idx.get(s)
            if idx is not None:
                A_rows.append((idx, j))
    
    if not A_rows:
        return None
    
    # Use sparse matrix format
    from scipy.sparse import csr_matrix
    
    row_indices = [r for r, c in A_rows]
    col_indices = [c for r, c in A_rows]
    data = [1.0] * len(A_rows)
    
    A = csr_matrix((data, (row_indices, col_indices)), shape=(n_triples, n_blocks))
    
    # Objective: minimize sum of x_j
    c = np.ones(n_blocks)
    
    # Constraints: A @ x >= 1
    lb = np.ones(n_triples)
    ub = np.full(n_triples, np.inf)
    
    constraints = LinearConstraint(A, lb, ub)
    
    # Bounds: 0 <= x_j <= 1
    integrality = np.ones(n_blocks)  # All variables are integer
    bounds = Bounds(0, 1)
    
    try:
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={'time_limit': max(1.0, deadline - time.time() - 3.0)}
        )
        
        if result.status == 0 and result.x is not None:
            selected = [initial_blocks[j] for j in range(n_blocks) if result.x[j] > 0.5]
            if selected:
                # Verify validity
                if _is_valid_cover(selected, v, k, t, all_tsubsets):
                    return selected
        elif result.status == 1:  # Optimal
            pass  # Already handled above
        
    except Exception as e:
        pass
    
    return None

def _is_valid_cover(blocks, v, k, t, all_tsubsets):
    """Check if the given blocks form a valid covering."""
    covered = set()
    for block in blocks:
        for s in itertools.combinations(block, t):
            covered.add(s)
    return len(covered) == len(all_tsubsets)
