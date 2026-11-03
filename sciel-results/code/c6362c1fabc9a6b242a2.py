import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Number of t-subsets in a k-set
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    # Compute triple coverage frequency: how many blocks cover each triple
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1
        
    # Weight for each triple: 1/freq (rarer triples get higher weight)
    triple_weights = 1.0 / (triple_freq.astype(np.float64) + 1e-10)
    
    def greedy_cover(uncovered_init, use_weighted=True, random_tie=False, temperature=0.0):
        """Run greedy set cover, optionally weighted with optional randomization."""
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
            if random_tie and temperature > 0:
                # Soft-max sampling with temperature
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                # Scale gains by temperature for soft-max
                scaled = gains / temperature
                # Numerical stability
                scaled -= np.max(scaled)
                probs = np.exp(scaled)
                probs /= np.sum(probs)
                best_idx = np.random.choice(n_blocks, p=probs)
            elif random_tie:
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
        """Remove redundant blocks from a covering using efficient coverage tracking."""
        if not selected_indices:
            return selected_indices
            
        remaining = list(selected_indices)
        
        # Build coverage count for each triple: how many remaining blocks cover it
        # Use a dictionary or array for efficiency
        triple_coverage = np.zeros(n_triples, dtype=np.int32)
        for block_idx in remaining:
            triple_coverage[block_triples[block_idx]] += 1
            
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_remaining = []
            
            # Try removing each block
            for i, block_idx in enumerate(remaining):
                if time.time() > deadline:
                    break
                    
                # Check if this block can be removed
                # It can be removed if every triple it covers has coverage > 1
                block_triples_set = block_triples[block_idx]
                
                can_remove = True
                for triple_idx in block_triples_set:
                    if triple_coverage[triple_idx] <= 1:
                        can_remove = False
                        break
                        
                if can_remove:
                    # Remove this block and update coverage
                    triple_coverage[block_triples_set] -= 1
                    changed = True
                else:
                    new_remaining.append(block_idx)
                    
            remaining = new_remaining
            
        return remaining
    
    def local_search(selected_indices, max_iters=500):
        """Local search: try to improve by swapping blocks or removing extras."""
        if not selected_indices:
            return selected_indices
            
        remaining = list(selected_indices)
        
        # Build coverage count for each triple
        triple_coverage = np.zeros(n_triples, dtype=np.int32)
        for block_idx in remaining:
            triple_coverage[block_triples[block_idx]] += 1
            
        # Also track which blocks cover each triple (for swap operations)
        # Use a dictionary: triple_idx -> set of block indices in remaining
        triple_to_blocks = {}
        for block_idx in remaining:
            for triple_idx in block_triples[block_idx]:
                if triple_idx not in triple_to_blocks:
                    triple_to_blocks[triple_idx] = set()
                triple_to_blocks[triple_idx].add(block_idx)
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # First, try to remove redundant blocks
            changed = True
            while changed and time.time() < deadline:
                changed = False
                new_remaining = []
                for i, block_idx in enumerate(remaining):
                    if time.time() > deadline:
                        break
                    block_triples_set = block_triples[block_idx]
                    can_remove = True
                    for triple_idx in block_triples_set:
                        if triple_coverage[triple_idx] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        triple_coverage[block_triples_set] -= 1
                        for triple_idx in block_triples_set:
                            triple_to_blocks[triple_idx].discard(block_idx)
                        changed = True
                    else:
                        new_remaining.append(block_idx)
                remaining = new_remaining
                if changed:
                    improved = True
                    
            # Try swaps: replace one block with another to potentially reduce size
            # or maintain same size but improve coverage for future reductions
            if time.time() > deadline:
                break
                
            # Try random swaps
            for _ in range(min(max_iters, 100)):
                if time.time() > deadline:
                    break
                    
                if len(remaining) < 2:
                    break
                    
                # Pick a random block to potentially replace
                idx = random.randint(0, len(remaining) - 1)
                old_block = remaining[idx]
                
                # Find a candidate replacement block
                # Prefer blocks that cover many currently singly-covered triples
                # and don't introduce new uncovered triples
                
                # Get triples that would become uncovered if we remove old_block
                critical_triples = []
                for triple_idx in block_triples[old_block]:
                    if triple_coverage[triple_idx] == 1:
                        critical_triples.append(triple_idx)
                        
                if not critical_triples:
                    # This block is redundant, should have been removed
                    continue
                    
                # Find candidate blocks that cover all critical_triples
                # and are not already in remaining
                candidates = []
                for candidate_block in range(n_blocks):
                    if candidate_block in remaining:
                        continue
                    # Check if candidate covers all critical triples
                    candidate_triples = set(block_triples[candidate_block].tolist())
                    if all(t in candidate_triples for t in critical_triples):
                        # Calculate benefit: how many additional triples it covers
                        # that are currently covered by multiple blocks
                        additional = 0
                        for triple_idx in block_triples[candidate_block]:
                            if triple_coverage[triple_idx] >= 2:
                                additional += 1
                        candidates.append((additional, candidate_block))
                        
                if not candidates:
                    continue
                    
                # Pick the best candidate (most additional coverage)
                candidates.sort(reverse=True)
                _, new_block = candidates[0]
                
                # Perform the swap
                # Remove old_block
                triple_coverage[block_triples[old_block]] -= 1
                for triple_idx in block_triples[old_block]:
                    triple_to_blocks[triple_idx].discard(old_block)
                    
                # Add new_block
                triple_coverage[block_triples[new_block]] += 1
                for triple_idx in block_triples[new_block]:
                    if triple_idx not in triple_to_blocks:
                        triple_to_blocks[triple_idx] = set()
                    triple_to_blocks[triple_idx].add(new_block)
                    
                remaining[idx] = new_block
                improved = True
                
        return remaining
    
    # Run multiple greedy passes with different strategies
    best_solution = None
    best_size = float('inf')
    
    # Strategy 1: Standard greedy (unweighted)
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=False, random_tie=False)
    if selected:
        pruned = prune_blocks(selected)
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_solution = [list(all_blocks[i]) for i in pruned]
    
    # Strategy 2: Weighted greedy
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=False)
    if selected:
        pruned = prune_blocks(selected)
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_solution = [list(all_blocks[i]) for i in pruned]
    
    # Strategy 3: Weighted greedy with local search
    selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=True, random_tie=False)
    if selected:
        pruned = prune_blocks(selected)
        local_search_result = local_search(pruned)
        if len(local_search_result) < best_size:
            best_size = len(local_search_result)
            best_solution = [list(all_blocks[i]) for i in local_search_result]
    
    # Strategy 4-15: Random tie-breaking restarts with various temperatures
    for restart in range(12):
        if time.time() > deadline - 20.0:
            break
            
        use_weighted = (restart % 2 == 0)
        temperature = 0.5 + 0.5 * restart  # Increasing temperature
        
        selected = greedy_cover(np.ones(n_triples, dtype=bool), use_weighted=use_weighted, 
                               random_tie=True, temperature=temperature)
        if selected:
            pruned = prune_blocks(selected)
            # Apply local search to some restarts
            if restart % 3 == 0 and time.time() < deadline - 10.0:
                local_search_result = local_search(pruned)
                if len(local_search_result) < best_size:
                    best_size = len(local_search_result)
                    best_solution = [list(all_blocks[i]) for i in local_search_result]
            else:
                if len(pruned) < best_size:
                    best_size = len(pruned)
                    best_solution = [list(all_blocks[i]) for i in pruned]
    
    # Final polish: try local search on the best solution found
    if best_solution is not None and time.time() < deadline - 5.0:
        # Reconstruct the list of block indices from best_solution
        # We need to find which block indices correspond to best_solution
        block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        best_indices = [block_to_idx[tuple(b)] for b in best_solution]
        polished = local_search(best_indices, max_iters=200)
        if len(polished) < best_size:
            best_size = len(polished)
            best_solution = [list(all_blocks[i]) for i in polished]
    
    # Final validation and return
    if best_solution is None:
        # Fallback: return a valid covering (shouldn't happen)
        return [list(b) for b in all_blocks[:100]]
        
    return best_solution
