import itertools
import numpy as np
import scipy.sparse as sp
import time
import random

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a global greedy heuristic
    (accelerated by sparse matrix multiplication) followed by local search.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all blocks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Build sparse matrix M: M[i, j] = 1 if block i covers triple j
    rows = []
    cols = []
    for i, block in enumerate(all_blocks):
        for tr in itertools.combinations(block, t):
            j = triple_to_idx[tr]
            rows.append(i)
            cols.append(j)
            
    M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_blocks, num_triples))
    
    # Helper to get block indices for a list of triple indices
    # We will use this for finding candidates in local search
    # triple_to_blocks: dict mapping triple_idx -> list of block indices
    triple_to_blocks = {i: [] for i in range(num_triples)}
    for i, block in enumerate(all_blocks):
        for tr in itertools.combinations(block, t):
            j = triple_to_idx[tr]
            triple_to_blocks[j].append(i)

    # 1. Global Greedy Construction
    def global_greedy():
        uncovered = np.ones(num_triples, dtype=np.int32)
        selected_indices = []
        
        while uncovered.sum() > 0:
            if time.time() > deadline - 10:
                break
                
            # Calculate gain for each block: number of currently uncovered triples it covers
            # M is (num_blocks, num_triples), uncovered is (num_triples,)
            # Result is (num_blocks,)
            gains = M.dot(uncovered)
            
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break # Should not happen if problem is solvable
                
            selected_indices.append(best_idx)
            
            # Update uncovered
            # Get the row of M for best_idx
            row = M.getrow(best_idx)
            # Set uncovered to 0 for all triples in this row
            uncovered[row.indices] = 0
            
        return [all_blocks[i] for i in selected_indices]

    blocks = global_greedy()
    
    # 2. Local Search Optimization
    # We maintain a list of block indices to allow fast coverage calculation via M
    current_indices = [all_blocks.index(b) for b in blocks]
    
    def get_coverage(indices):
        if not indices:
            return np.zeros(num_triples, dtype=np.int32)
        # M[indices] returns a sparse matrix
        # .sum(axis=0) returns a matrix, we need an array
        cov = M[indices].sum(axis=0)
        return np.asarray(cov).flatten().astype(np.int32)

    def prune_indices(indices):
        """
        Removes redundant blocks from the list of indices.
        A block is redundant if it does not cover any triple that is covered by exactly one block.
        Returns new list of indices.
        """
        if not indices:
            return indices
            
        changed = True
        while changed:
            if time.time() > deadline - 5:
                break
                
            coverage = get_coverage(indices)
            
            new_indices = []
            for idx in indices:
                # Check if this block has any unique triple
                row = M.getrow(idx)
                # For each triple in this block, check if coverage == 1
                # If any is 1, the block is essential
                is_essential = False
                for tr_idx in row.indices:
                    if coverage[tr_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_indices.append(idx)
            
            if len(new_indices) == len(indices):
                changed = False
            else:
                indices = new_indices
                changed = True
                
        return indices

    # Initial Prune
    current_indices = prune_indices(current_indices)
    best_len = len(current_indices)
    
    # Local Search Loop
    # Strategy:
    # 1. Try to remove a block if it's redundant (handled by prune)
    # 2. Try to swap a block with another that covers its unique triples, hoping to enable further pruning.
    
    stuck_count = 0
    rng = random.Random(42)
    
    while time.time() < deadline - 2:
        improved = False
        
        # Prune first
        prev_len = len(current_indices)
        current_indices = prune_indices(current_indices)
        if len(current_indices) < prev_len:
            improved = True
            best_len = len(current_indices)
            stuck_count = 0
            continue
            
        n_blocks = len(current_indices)
        if n_blocks == 0:
            break
            
        # Pick a random block to try to replace
        # We prioritize blocks with few unique triples? Or just random.
        idx_pos = rng.randint(0, n_blocks - 1)
        idx = current_indices[idx_pos]
        
        # Find unique triples for this block
        row = M.getrow(idx)
        coverage = get_coverage(current_indices)
        
        unique_triples = []
        for tr_idx in row.indices:
            if coverage[tr_idx] == 1:
                unique_triples.append(tr_idx)
                
        if not unique_triples:
            # Should have been pruned, but just in case
            current_indices.pop(idx_pos)
            improved = True
            best_len = len(current_indices)
            stuck_count = 0
            continue
            
        # Find candidate blocks that cover all unique_triples
        # Candidate must be in the intersection of triple_to_blocks[tr] for tr in unique_triples
        # And candidate != idx
        
        if len(unique_triples) > 3:
            # Too many constraints, likely no single block covers all (since block size is 7, max unique triples in a block is 35, but usually small)
            # Actually, if unique_triples are disjoint, they can't be in one block.
            # If they share points, they might be.
            # We just check intersection.
            pass
            
        candidates = set(triple_to_blocks[unique_triples[0]])
        candidates.discard(idx)
        
        for tr_idx in unique_triples[1:]:
            candidates.intersection_update(triple_to_blocks[tr_idx])
            
        if not candidates:
            stuck_count += 1
            if stuck_count > 50:
                # Perturb: Randomly swap with a random block?
                # Or just break and restart?
                # Let's try a random swap to escape local opt
                if rng.random() < 0.1:
                    new_idx = rng.randint(0, num_blocks - 1)
                    if new_idx not in current_indices:
                        current_indices[idx_pos] = new_idx
                        current_indices = prune_indices(current_indices)
                        if len(current_indices) < best_len:
                            best_len = len(current_indices)
                            stuck_count = 0
                            improved = True
                        else:
                            # Revert? We don't have a history.
                            # Just continue.
                            pass
                continue
        else:
            # Try each candidate
            # To save time, limit candidates if there are too many
            cand_list = list(candidates)
            if len(cand_list) > 10:
                cand_list = rng.sample(cand_list, 10)
                
            for cand_idx in cand_list:
                # Swap
                new_indices = current_indices.copy()
                new_indices[idx_pos] = cand_idx
                
                # Prune
                pruned_indices = prune_indices(new_indices)
                
                if len(pruned_indices) < len(current_indices):
                    current_indices = pruned_indices
                    best_len = len(current_indices)
                    improved = True
                    stuck_count = 0
                    break # Restart loop with new state
                # Else, revert (do nothing)
            
        if not improved:
            stuck_count += 1
            if stuck_count > 100:
                # Reset stuck counter and maybe try a different strategy
                stuck_count = 0

    # Final prune
    current_indices = prune_indices(current_indices)
    
    # Convert back to list of lists
    final_blocks = [list(all_blocks[i]) for i in current_indices]
    
    return final_blocks
