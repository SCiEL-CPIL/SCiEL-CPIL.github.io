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
            
    # Precompute triple frequencies for weighted greedy
    triple_freq = np.zeros(n_triples, dtype=np.float64)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1
    # Weight: inverse frequency (rarity)
    triple_weight = 1.0 / np.maximum(triple_freq, 1.0)
    
    def greedy_cover(use_weighted=False, random_tiebreak=False):
        """Run greedy set cover with optional weighting and random tie-breaking."""
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            # Calculate gains
            if use_weighted:
                # Weighted gain: sum of weights of uncovered triples covered
                gains = np.sum(uncovered[block_triples] * triple_weight[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)
            
            # Find best block
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            if random_tiebreak:
                candidates = np.where(gains == max_gain)[0]
                best_idx = np.random.choice(candidates)
            else:
                best_idx = np.argmax(gains)
            
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
            
        return selected
    
    def peel_redundant(block_indices):
        """Remove redundant blocks from a solution."""
        if not block_indices:
            return block_indices
        
        # Build coverage matrix: which triples are covered by which selected blocks
        n_sel = len(block_indices)
        covered_by = np.zeros((n_triples, n_sel), dtype=bool)
        for i, bidx in enumerate(block_indices):
            covered_by[block_triples[bidx], i] = True
        
        # For each block, check if all its triples are covered by other blocks
        keep = []
        for i in range(n_sel):
            # Check if removing block i still covers all its triples
            # A triple is covered by block i if covered_by[triple, i] is True
            # After removal, triple is still covered if any other block covers it
            my_triples = block_triples[block_indices[i]]
            # For each triple in this block, check if it's covered by another block
            still_covered = True
            for triple_idx in my_triples:
                # Count how many blocks cover this triple
                if np.sum(covered_by[triple_idx]) <= 1:
                    # Only covered by this block (or no other block)
                    still_covered = False
                    break
            if still_covered:
                # Can remove this block
                continue
            else:
                keep.append(block_indices[i])
        
        return keep
    
    # Run multiple strategies
    best_solution = None
    best_size = float('inf')
    
    # Strategy 1: Deterministic greedy
    sel = greedy_cover(use_weighted=False, random_tiebreak=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = [list(blocks[i]) for i in peeled]
    
    # Strategy 2: Weighted greedy
    sel = greedy_cover(use_weighted=True, random_tiebreak=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = [list(blocks[i]) for i in peeled]
    
    # Strategy 3+: Random restarts with weighted greedy
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        sel = greedy_cover(use_weighted=True, random_tiebreak=True)
        if sel:
            peeled = peel_redundant(sel)
            size = len(peeled)
            if size < best_size:
                best_size = size
                best_solution = [list(blocks[i]) for i in peeled]
    
    # Final polish: try to remove more blocks with multiple passes
    if best_solution is None:
        # Fallback: return empty or a valid covering
        best_solution = []
    else:
        # Multiple peeling passes
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            current_indices = []
            for b in best_solution:
                # Find the index of this block
                for i, blk in enumerate(blocks):
                    if list(blk) == b:
                        current_indices.append(i)
                        break
            peeled = peel_redundant(current_indices)
            if len(peeled) < len(best_solution):
                best_solution = [list(blocks[i]) for i in peeled]
            else:
                break
    
    return best_solution
