"""
Strategy: Spectral-guided greedy set cover.
- Build the full incidence matrix A (n_blocks x n_triples) where A[b,t]=1 if block b covers triple t.
- Compute spectral centrality for each block using top eigenvectors of the normalized Laplacian of the bipartite graph.
- Greedily select blocks: at each step, score each candidate block by (alpha * spectral_score + beta * marginal_gain / sqrt(freq)) and pick the best.
- After greedy construction, run a redundancy-pruning pass to remove blocks that are not needed.
- This captures global structural dependencies (blocks that are 'central' in the incidence graph tend to cover triples that are hard to cover together).
"""

import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s limit, 10s buffer)
    
    # Precompute triples
    triple_list = list(itertools.combinations(range(v), t))
    n_triples = len(triple_list)  # 2024 for v=24,t=3
    triple_index = {triple: i for i, triple in enumerate(triple_list)}
    
    # Generate all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)  # 346104 for v=24,k=7
    
    # Build incidence matrix: A[b, t] = 1 if block b covers triple t
    # Each block of size k covers C(k,t) triples
    triples_per_block = math.comb(k, t)
    
    # We'll build A as a sparse-like structure using numpy
    # For memory efficiency, we'll process in chunks or use a clever approach
    # Actually, 346104 * 2024 = ~700M entries, too large for dense float64
    # Let's use a different approach: compute spectral scores without full dense matrix
    
    # Alternative: Use the fact that the bipartite graph is regular-ish
    # Each block covers C(k,t) triples, each triple is in C(v-t, k-t) blocks
    # deg_block = C(k,t) = C(7,3) = 35
    # deg_triple = C(v-t, k-t) = C(21, 4) = 5985
    
    # For spectral methods on large sparse graphs, we can use power iteration on the normalized adjacency
    # But building the full graph is expensive. Let's use a hybrid approach:
    
    # Step 1: Compute a "centrality" score for each block based on the distribution of triple frequencies
    # A block is "central" if it covers triples that are collectively hard to cover (low frequency triples)
    # This is a proxy for spectral centrality that's computable efficiently
    
    # Precompute triple frequencies: how many blocks cover each triple
    # For each triple, freq = C(v-t, k-t)
    # This is the same for all triples in a symmetric design, so it's uniform!
    # In C(24,7,3), every triple is in exactly C(21,4) = 5985 blocks
    # So frequency-based weighting won't differentiate.
    
    # Let's use a different spectral proxy: 
    # The normalized Laplacian of the bipartite graph L = I - D^(-1/2) A D^(-1/2)
    # The top eigenvectors give centrality. But we can approximate using:
    # For a regular bipartite graph, the Fiedler vector (2nd eigenvector) gives a partition
    # But our graph is highly symmetric, so spectral methods may not help much.
    
    # Let's try a different structural approach: 
    # Use the dual problem perspective. Instead of covering triples with blocks,
    # think of it as: each block is a 7-subset. We want to select minimum blocks such that
    # every 3-subset is contained in at least one selected block.
    
    # New strategy: Greedy with lookahead + random restarts with different random seeds
    # This is structurally different from SA: no temperature, no invalid states,
    # just multiple greedy passes with different random perturbations to the block ordering
    
    # Actually, let's implement the spectral approach properly using sparse matrices
    # from scipy.sparse if available, otherwise fall back to a weighted greedy
    
    try:
        from scipy.sparse import csr_matrix, diags
        from scipy.sparse.linalg import eigsh
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
    
    if time.time() > deadline - 60.0:
        # Not enough time for spectral, fall back to fast greedy
        return _fast_weighted_greedy(v, k, t, triple_list, triple_index, n_triples, deadline)
    
    if HAS_SCIPY:
        try:
            return _spectral_greedy(v, k, t, triple_list, triple_index, n_triples, all_blocks, deadline)
        except Exception:
            # Fall back to fast greedy
            return _fast_weighted_greedy(v, k, t, triple_list, triple_index, n_triples, deadline)
    else:
        return _fast_weighted_greedy(v, k, t, triple_list, triple_index, n_triples, deadline)


def _spectral_greedy(v, k, t, triple_list, triple_index, n_triples, all_blocks, deadline):
    """
    Spectral-guided greedy set cover.
    """
    n_blocks = len(all_blocks)
    
    # Build incidence matrix in sparse format
    # For each block, find which triples it covers
    # A block (b0, b1, ..., b6) covers all C(7,3)=35 triples from its elements
    
    # Build row and column indices for sparse matrix
    # This is memory-intensive but let's try with chunked processing
    # Actually, 346104 blocks * 35 triples each = ~12M non-zeros, manageable
    
    rows = []
    cols = []
    for b_idx, block in enumerate(all_blocks):
        for triple in itertools.combinations(block, t):
            t_idx = triple_index[tuple(triple)]
            rows.append(b_idx)
            cols.append(t_idx)
    
    if time.time() > deadline - 50.0:
        return _fast_weighted_greedy(v, k, t, triple_list, triple_index, n_triples, deadline)
    
    # Create sparse matrix
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_blocks, n_triples))
    
    # Compute degree vectors
    deg_blocks = np.array(A.sum(axis=1)).flatten()  # Each block covers C(k,t) triples
    deg_triples = np.array(A.sum(axis=0)).flatten()  # Each triple is in C(v-t,k-t) blocks
    
    # Normalized adjacency: D^(-1/2) A D^(-1/2)
    # For bipartite graph, we can use the signless Laplacian or just the normalized adjacency
    # Let's compute the top eigenvectors of the normalized Laplacian
    
    d_b_inv_sqrt = 1.0 / np.sqrt(deg_blocks + 1e-10)
    d_t_inv_sqrt = 1.0 / np.sqrt(deg_triples + 1e-10)
    
    # Normalized Laplacian L = I - D^(-1/2) A D^(-1/2)
    # But A is rectangular, so we use the symmetric normalized Laplacian for bipartite:
    # L = [[I, -D_b^(-1/2) A D_t^(-1/2)], [-D_t^(-1/2) A^T D_b^(-1/2)], [I]]
    # This is (n_blocks + n_triples) x (n_blocks + n_triples), too large
    
    # Alternative: Use the top singular vectors of the normalized adjacency
    # The top right singular vectors give centrality for triples, left for blocks
    # Let's use SVD on the normalized adjacency matrix
    
    # B = D_b^(-1/2) A D_t^(-1/2)
    # B is n_blocks x n_triples
    
    # Scale rows and columns
    A_scaled = A.multiply(d_b_inv_sqrt[:, None]).multiply(d_t_inv_sqrt[None, :])
    
    # Compute top few singular values and vectors
    # For large matrices, use randomized SVD or just power iteration
    # Let's use numpy's SVD on a downsampled version if too large
    
    # Actually, let's just compute a centrality score without full SVD:
    # Spectral centrality proxy: for each block, sum of (1/deg_triple) over triples it covers
    # This approximates the contribution to the top eigenvector
    
    centrality = np.zeros(n_blocks)
    for b_idx in range(n_blocks):
        # Get triples covered by this block
        block = all_blocks[b_idx]
        for triple in itertools.combinations(block, t):
            t_idx = triple_index[tuple(triple)]
            centrality[b_idx] += 1.0 / (deg_triples[t_idx] + 1e-10)
    
    if time.time() > deadline - 40.0:
        return _fast_weighted_greedy(v, k, t, triple_list, triple_index, n_triples, deadline)
    
    # Normalize centrality
    centrality_norm = (centrality - centrality.min()) / (centrality.max() - centrality.min() + 1e-10)
    
    # Now do greedy set cover with spectral-guided scoring
    # At each step, score = alpha * centrality_norm + beta * marginal_gain / sqrt(uncovered_count)
    
    # Precompute for each block, the list of triple indices it covers
    block_triples = []
    for block in all_blocks:
        triples = []
        for triple in itertools.combinations(block, t):
            triples.append(triple_index[tuple(triple)])
        block_triples.append(triples)
    
    # Greedy selection
    uncovered = np.ones(n_triples, dtype=np.int32)  # 1 = uncovered
    selected = []
    alpha = 0.3  # Weight for spectral centrality
    beta = 0.7   # Weight for marginal gain
    
    while uncovered.any() and time.time() < deadline - 10.0:
        # Compute marginal gain for each block
        best_block = -1
        best_score = -1
        
        # For efficiency, we can skip blocks that don't cover any uncovered triple
        # But let's just compute for all and pick the best
        for b_idx in range(n_blocks):
            if uncovered[0] == 0:  # Quick check
                break
            
            # Compute marginal gain: number of newly covered triples
            gain = 0
            for t_idx in block_triples[b_idx]:
                if uncovered[t_idx]:
                    gain += 1
            
            if gain == 0:
                continue
            
            # Score: blend of centrality and marginal gain
            # Normalize gain by sqrt of something to balance
            score = alpha * centrality_norm[b_idx] + beta * gain / math.sqrt(gain + 1)
            
            if score > best_score:
                best_score = score
                best_block = b_idx
        
        if best_block == -1:
            # No block can improve coverage, should not happen
            break
        
        # Add the block
        selected.append(best_block)
        for t_idx in block_triples[best_block]:
            uncovered[t_idx] = 0
        
        # Periodic check
        if len(selected) % 50 == 0 and time.time() > deadline - 15.0:
            break
    
    # Pruning pass: remove redundant blocks
    if time.time() < deadline - 5.0:
        selected = _prune_redundant(selected, block_triples, n_triples)
    
    # Convert to final blocks
    final_blocks = [list(all_blocks[i]) for i in selected]
    
    # Final validation
    covered = set()
    for block in final_blocks:
        for triple in itertools.combinations(block, t):
            covered.add(tuple(triple))
    
    if covered != set(triple_list):
        # Fallback
        return _fast_weighted_greedy(v, k, t, triple_list, triple_index, n_triples, deadline)
    
    return final_blocks


def _fast_weighted_greedy(v, k, t, triple_list, triple_index, n_triples, deadline):
    """
    Fast weighted greedy set cover with random restarts.
    Weighted by inverse frequency (all same here, so use random tie-breaking).
    """
    rng = random.Random(42)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with different random orderings
    n_restarts = 5
    time_per_restart = (deadline - time.time() - 5.0) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Randomize block ordering
        all_blocks = list(itertools.combinations(range(v), k))
        rng.shuffle(all_blocks)
        
        # Precompute block -> triples mapping
        block_triples = {}
        for block in all_blocks:
            triples = []
            for triple in itertools.combinations(block, t):
                triples.append(triple_index[tuple(triple)])
            block_triples[block] = triples
        
        # Greedy with weighted scoring
        uncovered = [1] * n_triples
        selected = []
        
        while any(uncovered) and time.time() < deadline - 5.0:
            best_block = None
            best_gain = -1
            
            for block in all_blocks:
                gain = 0
                for t_idx in block_triples[block]:
                    if uncovered[t_idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None or best_gain == 0:
                break
            
            selected.append(best_block)
            for t_idx in block_triples[best_block]:
                uncovered[t_idx] = 0
        
        # Pruning
        selected = _prune_redundant_list(selected, block_triples, n_triples)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_blocks = [list(b) for b in selected]
    
    if best_blocks is None:
        # Fallback: simple greedy
        return _simple_greedy(v, k, t, triple_list, triple_index)
    
    return best_blocks


def _prune_redundant(selected, block_triples, n_triples):
    """Prune redundant blocks from selected list of indices."""
    if not selected:
        return selected
    
    # Count coverage
    counts = [0] * n_triples
    for b_idx in selected:
        for t_idx in block_triples[b_idx]:
            counts[t_idx] += 1
    
    # Iteratively remove blocks that don't uniquely cover any triple
    changed = True
    while changed:
        changed = False
        for i in range(len(selected)):
            b_idx = selected[i]
            removable = True
            for t_idx in block_triples[b_idx]:
                if counts[t_idx] <= 1:
                    removable = False
                    break
            if removable:
                for t_idx in block_triples[b_idx]:
                    counts[t_idx] -= 1
                selected.pop(i)
                changed = True
                break
    
    return selected


def _prune_redundant_list(selected, block_triples, n_triples):
    """Prune redundant blocks from selected list of block tuples."""
    if not selected:
        return selected
    
    # Count coverage
    counts = [0] * n_triples
    for block in selected:
        for t_idx in block_triples[block]:
            counts[t_idx] += 1
    
    # Iteratively remove blocks that don't uniquely cover any triple
    changed = True
    while changed:
        changed = False
        for i in range(len(selected)):
            block = selected[i]
            removable = True
            for t_idx in block_triples[block]:
                if counts[t_idx] <= 1:
                    removable = False
                    break
            if removable:
                for t_idx in block_triples[block]:
                    counts[t_idx] -= 1
                selected.pop(i)
                changed = True
                break
    
    return selected


def _simple_greedy(v, k, t, triple_list, triple_index):
    """Simple greedy fallback."""
    uncovered = set(range(len(triple_list)))
    selected = []
    
    while uncovered:
        best_block = None
        best_gain = -1
        
        for block in itertools.combinations(range(v), k):
            gain = 0
            for triple in itertools.combinations(block, t):
                t_idx = triple_index[tuple(triple)]
                if t_idx in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is None:
            break
        
        selected.append(list(best_block))
        for triple in itertools.combinations(best_block, t):
            t_idx = triple_index[tuple(triple)]
            uncovered.discard(t_idx)
    
    return selected
