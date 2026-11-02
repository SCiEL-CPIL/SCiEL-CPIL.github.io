import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    # Shape: (n_blocks, C(k, t))
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    # Compute how many blocks cover each triple (for weighted greedy)
    triple_counts = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_counts[block_triples[i]] += 1
    # Weight: rare triples get higher weight
    triple_weights = 1.0 / (triple_counts + 1).astype(np.float64)
    
    # Precompute triple_to_blocks for efficient local search
    # triple_to_blocks[triple_idx] = np.array of block indices covering that triple
    triple_to_blocks = [np.empty(0, dtype=np.int32) for _ in range(n_triples)]
    # 1. Allocate based on counts
    for i in range(n_triples):
        triple_to_blocks[i] = np.empty(triple_counts[i], dtype=np.int32)
    # 2. Fill
    cursors = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        for triple_idx in block_triples[i]:
            triple_to_blocks[triple_idx][cursors[triple_idx]] = i
            cursors[triple_idx] += 1
    
    def greedy_cover(random_tiebreak=False, alpha=1.0):
        """
        Weighted greedy with alpha parameter controlling exploration vs exploitation.
        alpha=1.0: pure weighted greedy (prioritize rare triples)
        alpha=0.5: balance between raw count and weighted gain
        alpha=0.0: pure raw count greedy
        """
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Raw count of uncovered triples in each block
            raw_counts = np.sum(uncovered[block_triples].astype(np.int32), axis=1)
            
            # Weighted gain: sum of weights of uncovered triples in block
            weighted_gains = np.sum(uncovered[block_triples].astype(np.float64) * triple_weights[block_triples], axis=1)
            
            # Normalize weighted gains to comparable scale
            max_weighted = np.max(weighted_gains)
            if max_weighted > 0:
                normalized_weighted = weighted_gains / max_weighted
            else:
                normalized_weighted = weighted_gains
            
            # Combined score: blend of raw count and weighted gain
            # Higher alpha emphasizes rare triples, lower alpha emphasizes coverage count
            max_raw = np.max(raw_counts)
            if max_raw > 0:
                normalized_raw = raw_counts.astype(np.float64) / max_raw
            else:
                normalized_raw = raw_counts.astype(np.float64)
            
            combined_scores = alpha * normalized_weighted + (1 - alpha) * normalized_raw
            
            if random_tiebreak:
                max_score = np.max(combined_scores)
                if max_score == 0:
                    break
                # Allow some randomness among near-best candidates
                threshold = max_score * 0.95
                candidates = np.where(combined_scores >= threshold)[0]
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(combined_scores)
                if combined_scores[best_idx] == 0:
                    break
            
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
        
        return selected
    
    def remove_redundant(block_indices):
        """Remove blocks that are not needed to cover all triples."""
        if len(block_indices) == 0:
            return block_indices
        
        # For each block, check if removing it leaves all triples covered
        # Use a coverage count approach
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        result = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(result)):
                if time.time() > deadline:
                    return result
                bi = result[i]
                # Check if any triple in this block would become uncovered
                triples_in_block = block_triples[bi]
                if np.any(coverage[triples_in_block] <= 1):
                    continue
                # Safe to remove
                coverage[triples_in_block] -= 1
                result.pop(i)
                changed = True
                break
        
        return result
    
    def local_search(block_indices, max_iterations=50):
        """
        Improved local search with:
        1. Redundancy removal
        2. Block replacement with better candidates (minimize overlap with existing)
        3. Try multiple candidate swaps, not just first
        """
        if len(block_indices) == 0:
            return block_indices
        
        # Build coverage count
        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1
        
        current = list(block_indices)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Pass 1: Remove redundant blocks
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                bi = current[i]
                block_triples_bi = block_triples[bi]
                
                # Check for redundancy: if all triples in this block have coverage > 1
                if np.all(coverage[block_triples_bi] > 1):
                    coverage[block_triples_bi] -= 1
                    current.pop(i)
                    improved = True
                    break # Restart to handle index shift
            
            if improved:
                continue
            
            # Pass 2: Try to replace blocks with better alternatives
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                bi = current[i]
                block_triples_bi = block_triples[bi]
                
                # Find critical triples (coverage == 1)
                critical_mask = coverage[block_triples_bi] == 1
                if not np.any(critical_mask):
                    # All triples covered multiple times - this is redundant
                    coverage[block_triples_bi] -= 1
                    current.pop(i)
                    improved = True
                    break
                
                critical_triples = block_triples_bi[critical_mask]
                
                # Find candidate blocks that cover ALL critical_triples
                min_ct = critical_triples[np.argmin(triple_counts[critical_triples])]
                candidates = triple_to_blocks[min_ct]
                
                other_critical = critical_triples[critical_triples != min_ct]
                
                valid_candidates = []
                for cand in candidates:
                    if cand == bi:
                        continue
                    if len(other_critical) == 0:
                        valid_candidates.append(cand)
                    else:
                        if np.all(np.isin(other_critical, block_triples[cand])):
                            valid_candidates.append(cand)
                
                if not valid_candidates:
                    continue
                
                # Score candidates: prefer those that cover more non-critical uncovered triples
                # and minimize overlap with existing blocks
                best_cand = None
                best_score = -1
                
                for cand in valid_candidates[:20]:  # Limit candidates for speed
                    # Count how many new triples this block covers (not already covered by others)
                    new_triples = block_triples[cand]
                    new_count = np.sum(coverage[new_triples] == 0)
                    
                    # Overlap penalty: how many triples are already well-covered
                    overlap_penalty = np.sum(coverage[new_triples] >= 2)
                    
                    score = new_count - 0.1 * overlap_penalty
                    
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                
                if best_cand is not None:
                    coverage[block_triples_bi] -= 1
                    coverage[block_triples[best_cand]] += 1
                    current[i] = best_cand
                    improved = True
                    break # Restart loop
            
            if not improved:
                break
                
        return current
    
    # First greedy pass with different alpha values
    best_solution = None
    best_size = float('inf')
    
    # Try multiple alpha values for the first pass
    alphas = [1.0, 0.8, 0.6, 0.5]
    for alpha in alphas:
        if time.time() > deadline - 30.0:
            break
        selected = greedy_cover(random_tiebreak=False, alpha=alpha)
        selected = remove_redundant(selected)
        selected = local_search(selected, max_iterations=30)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(blocks[i]) for i in selected]
    
    if best_solution is None:
        # Fallback: simple greedy
        selected = greedy_cover(random_tiebreak=False, alpha=1.0)
        selected = remove_redundant(selected)
        best_solution = [list(blocks[i]) for i in selected]
        best_size = len(best_solution)
    
    # Multiple random restarts with varying alpha
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Vary alpha and randomness
        alpha = 0.5 + 0.5 * (r % 3) / 2  # Cycle through 0.5, 0.75, 1.0
        selected = greedy_cover(random_tiebreak=True, alpha=alpha)
        selected = remove_redundant(selected)
        selected = local_search(selected, max_iterations=15)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(blocks[i]) for i in selected]
    
    # Final polish: try more aggressive local search on best solution
    if time.time() < deadline - 5.0 and best_solution is not None:
        best_indices = []
        for block in best_solution:
            best_indices.append(blocks.index(tuple(block)))
        polished = local_search(best_indices, max_iterations=50)
        if len(polished) < best_size:
            best_size = len(polished)
            best_solution = [list(blocks[i]) for i in polished]
    
    return best_solution
