import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    # Compute triple coverage frequency: how many blocks cover each triple
    # triple_freq[j] = number of blocks that cover triple j
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1
        
    # Weight for each triple: 1/freq (rarer triples get higher weight)
    triple_weights = 1.0 / (triple_freq.astype(np.float64) + 1e-10)
    
    # Precompute block weights: sum of weights of triples in each block
    # block_weight[i] = sum of triple_weights for triples in block i
    block_weights = np.sum(triple_weights[block_triples], axis=1)
    
    def greedy_cover(uncovered_init, use_weighted=True, random_tie=False):
        """Run greedy set cover, optionally weighted."""
        uncovered = uncovered_init.copy()
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            if use_weighted:
                # Weighted gain: sum of weights of uncovered triples in block
                gains = np.sum(uncovered[block_triples] * triple_weights[block_triples], axis=1)
            else:
                # Unweighted gain
                gains = np.sum(uncovered[block_triples], axis=1)
                
            # Find the block with maximum gain
            if random_tie:
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
                    
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
            
        return selected
    
    def prune_blocks(selected_indices):
        """Remove redundant blocks from a covering using efficient coverage counting."""
        if not selected_indices:
            return selected_indices
            
        # Build coverage count for each triple among selected blocks
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for block_idx in selected_indices:
            coverage_count[block_triples[block_idx]] += 1
            
        # Try removing each block
        remaining = list(selected_indices)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_remaining = []
            
            # For each block in remaining, check if it can be removed
            for block_idx in remaining:
                if time.time() > deadline:
                    break
                    
                # Get triples covered by this block
                block_triple_indices = block_triples[block_idx]
                
                # Check if all triples in this block have coverage > 1
                # (i.e., covered by at least one other block)
                can_remove = True
                for triple_idx in block_triple_indices:
                    if coverage_count[triple_idx] <= 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove this block and update coverage counts
                    for triple_idx in block_triple_indices:
                        coverage_count[triple_idx] -= 1
                    changed = True
                else:
                    new_remaining.append(block_idx)
                    
            remaining = new_remaining
            
        return remaining
    
    def local_search_improve(solution_blocks, max_iterations=50):
        """Try to improve solution by swapping blocks."""
        if not solution_blocks:
            return solution_blocks
            
        current_size = len(solution_blocks)
        current_blocks = list(solution_blocks)
        
        # Build coverage count
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for block_idx in current_blocks:
            coverage_count[block_triples[block_idx]] += 1
            
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Try removing each block
            for i in range(len(current_blocks)):
                if time.time() > deadline:
                    return current_blocks
                    
                block_idx = current_blocks[i]
                block_triple_indices = block_triples[block_idx]
                
                # Check if all triples have coverage > 1
                can_remove = True
                for triple_idx in block_triple_indices:
                    if coverage_count[triple_idx] <= 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove block
                    for triple_idx in block_triple_indices:
                        coverage_count[triple_idx] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break  # Restart search after removal
            
            # If no removal possible, try swapping: remove one block and add another
            if not improved:
                for i in range(len(current_blocks)):
                    if time.time() > deadline:
                        return current_blocks
                        
                    block_idx = current_blocks[i]
                    block_triple_indices = block_triples[block_idx]
                    
                    # Temporarily remove
                    for triple_idx in block_triple_indices:
                        coverage_count[triple_idx] -= 1
                    
                    # Find uncovered triples
                    uncovered_mask = coverage_count == 0
                    if not np.any(uncovered_mask):
                        # No uncovered, we can keep the removal
                        current_blocks.pop(i)
                        improved = True
                        break
                    
                    # Try to find a single block that covers all uncovered triples
                    # and doesn't add too many new blocks
                    uncovered_indices = np.where(uncovered_mask)[0]
                    
                    # Find blocks that cover all uncovered triples
                    best_new_block = -1
                    best_extra = float('inf')
                    
                    for new_block_idx in range(n_blocks):
                        if new_block_idx == block_idx:
                            continue
                        if new_block_idx in current_blocks:
                            continue
                        
                        new_block_triples = block_triples[new_block_idx]
                        
                        # Check if all uncovered triples are covered
                        all_covered = True
                        for unc_idx in uncovered_indices:
                            if unc_idx not in new_block_triples:
                                all_covered = False
                                break
                                
                        if not all_covered:
                            continue
                        
                        # Count how many new triples this would cover
                        extra = 0
                        for triple_idx in new_block_triples:
                            if coverage_count[triple_idx] == 0:
                                extra += 1
                                
                        if extra < best_extra:
                            best_extra = extra
                            best_new_block = new_block_idx
                            
                    if best_new_block != -1:
                        # Perform the swap
                        for triple_idx in block_triples[best_new_block]:
                            coverage_count[triple_idx] += 1
                        current_blocks.pop(i)
                        current_blocks.append(best_new_block)
                        improved = True
                        break
                    else:
                        # Restore coverage
                        for triple_idx in block_triple_indices:
                            coverage_count[triple_idx] += 1
                        continue
                else:
                    # If we broke out of the for loop, improved is True
                    # If we completed the loop, improved is False
                    pass
        
        return current_blocks
    
    # Run multiple greedy passes with different strategies
    best_solution = None
    best_size = float('inf')
    
    # Strategy 1: Standard greedy (unweighted)
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=False, random_tie=False)
    if selected:
        pruned = prune_blocks(selected)
        pruned = local_search_improve(pruned)
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_solution = [list(blocks[i]) for i in pruned]
    
    # Strategy 2: Weighted greedy
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=False)
    if selected:
        pruned = prune_blocks(selected)
        pruned = local_search_improve(pruned)
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_solution = [list(blocks[i]) for i in pruned]
    
    # Strategy 3-15: Random tie-breaking restarts (mix of weighted and unweighted)
    for restart in range(15):
        if time.time() > deadline - 20.0:
            break
            
        use_weighted = (restart % 2 == 0)
        selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=use_weighted, random_tie=True)
        if selected:
            pruned = prune_blocks(selected)
            pruned = local_search_improve(pruned)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_solution = [list(blocks[i]) for i in pruned]
    
    # Final validation and return
    if best_solution is None:
        # Fallback: return a valid covering (shouldn't happen)
        return [list(b) for b in blocks[:100]]
        
    return best_solution
