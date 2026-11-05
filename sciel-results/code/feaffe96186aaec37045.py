import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit
    
    # Pre-compute all t-subsets (triples for t=3)
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build the constraint matrix A (num_triples x num_blocks)
    # A[i, j] = 1 if triple i is contained in block j
    # Using sparse matrix for efficiency
    A = lil_matrix((num_triples, num_blocks), dtype=np.int8)
    
    # Map block index to its list of covered triple indices
    # This is useful for post-processing and potentially for a fallback greedy if MILP fails
    block_covered = [set() for _ in range(num_blocks)]
    
    for j, block in enumerate(all_blocks):
        # For each block, find all t-subsets within it
        for tri in itertools.combinations(block, t):
            i = triple_to_idx[tri]
            A[i, j] = 1
            block_covered[j].add(i)
        
        # Time check during construction (should be fast, but just in case)
        if time.time() > deadline - 10:
            # Fallback to a quick greedy if construction takes too long
            return greedy_fallback(v, k, t, deadline)

    # Convert to CSR for scipy.optimize.milp
    A_csr = A.tocsr()
    
    # Objective: minimize sum of x_j
    c = np.ones(num_blocks, dtype=np.float64)
    
    # Constraints: A @ x >= 1 for all triples
    # scipy.optimize.milp expects: lb <= A @ x <= ub
    # So we set lb = 1, ub = inf
    A_ub = A_csr
    A_lb = np.ones(num_triples, dtype=np.float64)
    A_ub_val = np.inf * np.ones(num_triples, dtype=np.float64)
    
    constraints = LinearConstraint(A_ub, A_lb, A_ub_val)
    
    # Bounds: 0 <= x_j <= 1 (binary variables)
    # In scipy.optimize.milp, integrality is specified separately
    integrality = np.ones(num_blocks, dtype=np.int8)
    bounds = Bounds(lb=0, ub=1)
    
    # Solve MILP
    try:
        # time_limit is in seconds
        # We reserve 5 seconds for post-processing
        remaining_time = deadline - time.time() - 5
        if remaining_time < 1:
            return greedy_fallback(v, k, t, deadline)
            
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={'time_limit': remaining_time, 'mip_rel_gap': 0.0}
        )
        
        if res.status == 0:  # Optimal
            x = res.x
            # Extract blocks where x_j > 0.5
            blocks = []
            for j in range(num_blocks):
                if x[j] > 0.5:
                    blocks.append(list(all_blocks[j]))
            return blocks
        elif res.status == 2:  # Feasible (not optimal)
            # res.x might contain a feasible solution
            if hasattr(res, 'x') and res.x is not None:
                x = res.x
                blocks = []
                for j in range(num_blocks):
                    if x[j] > 0.5:
                        blocks.append(list(all_blocks[j]))
                # Verify validity
                if is_valid_cover(blocks, v, k, t):
                    return blocks
            # If no valid solution from res.x, fallback
            return greedy_fallback(v, k, t, deadline)
        else:
            # Infeasible or error
            return greedy_fallback(v, k, t, deadline)
            
    except Exception as e:
        # Fallback to greedy on any error
        return greedy_fallback(v, k, t, deadline)

def greedy_fallback(v, k, t, deadline):
    """
    A fast greedy set cover fallback if MILP fails or times out.
    """
    start_time = time.time()
    
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    uncovered = [True] * num_triples
    num_uncovered = num_triples
    blocks = []
    
    # Pre-compute block covers for greedy
    all_blocks = list(itertools.combinations(range(v), k))
    block_covered = []
    for b in all_blocks:
        idxs = set()
        for tri in itertools.combinations(b, t):
            idxs.add(triple_to_idx[tri])
        block_covered.append(idxs)
    
    # Map triple to blocks containing it
    triple_to_blocks = [[] for _ in range(num_triples)]
    for bi, b in enumerate(all_blocks):
        for ti in block_covered[bi]:
            triple_to_blocks[ti].append(bi)
            
    while num_uncovered > 0:
        if time.time() > deadline - 2:
            break
            
        # Pick random uncovered triple
        target_tri = -1
        for _ in range(100): # Try a few times to find one
            idx = random.randint(0, num_triples - 1)
            if uncovered[idx]:
                target_tri = idx
                break
        if target_tri == -1:
            # Fallback scan
            for i in range(num_triples):
                if uncovered[i]:
                    target_tri = i
                    break
            if target_tri == -1:
                break
                
        candidates = triple_to_blocks[target_tri]
        best_block = -1
        best_gain = -1
        
        for bi in candidates:
            gain = 0
            for ti in block_covered[bi]:
                if uncovered[ti]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = bi
                
        if best_block != -1:
            blocks.append(all_blocks[best_block])
            for ti in block_covered[best_block]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    num_uncovered -= 1
        else:
            break
            
    # Remove redundancies
    if not blocks:
        return []
        
    counts = [0] * num_triples
    for b in blocks:
        bi = all_blocks.index(b) # This is slow, better to use index from generation
        # Actually, let's just rebuild counts properly
    # Rebuild counts efficiently
    counts = [0] * num_triples
    block_indices = []
    for b in blocks:
        # Find index
        # This is O(N) per block, which is bad.
        # Let's just use the indices we generated.
        pass
        
    # Simpler redundancy removal:
    # Re-calculate coverage for the current list
    current_counts = [0] * num_triples
    for b in blocks:
        for tri in itertools.combinations(b, t):
            current_counts[triple_to_idx[tri]] += 1
            
    final_blocks = []
    for b in blocks:
        redundant = True
        for tri in itertools.combinations(b, t):
            if current_counts[triple_to_idx[tri]] == 1:
                redundant = False
                break
        if not redundant:
            final_blocks.append(b)
            for tri in itertools.combinations(b, t):
                current_counts[triple_to_idx[tri]] -= 1
                
    return final_blocks

def is_valid_cover(blocks, v, k, t):
    """Verify that every t-subset is covered."""
    covered = set()
    for b in blocks:
        for tri in itertools.combinations(b, t):
            covered.add(tri)
    # Check if all possible t-subsets are covered
    all_triples = list(itertools.combinations(range(v), t))
    for tri in all_triples:
        if tri not in covered:
            return False
    return True
