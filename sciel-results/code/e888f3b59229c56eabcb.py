"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # Safety margin
    
    # 1. Generate all t-subsets and map to indices
    tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(tsubs)
    mask_to_idx = {}
    for i, c in enumerate(tsubs):
        m = 0
        for x in c:
            m |= (1 << x)
        mask_to_idx[m] = i

    # 2. Generate all k-subsets (candidate blocks)
    ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(ksubs)

    # 3. Build coverage matrix
    # coverage[b, i] = 1 if k-subset b covers t-subset i
    coverage_list = []
    for block in ksubs:
        covered_indices = [0] * n_tsubs
        for t_cand in itertools.combinations(block, t):
            m = 0
            for x in t_cand:
                m |= (1 << x)
            idx = mask_to_idx.get(m)
            if idx is not None:
                covered_indices[idx] = 1
        coverage_list.append(covered_indices)
        
    # Convert to numpy array
    coverage = np.array(coverage_list, dtype=np.uint8)
    
    # 4. Greedy Set Cover
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    n_uncovered = n_tsubs
    selected_blocks = []
    
    while n_uncovered > 0:
        if time.time() > deadline:
            break
            
        # Calculate gain for all blocks
        gains = coverage.dot(uncovered)
        
        best_idx = int(np.argmax(gains))
        best_gain = int(gains[best_idx])
        
        if best_gain == 0:
            break
            
        selected_blocks.append(best_idx)
        
        # Update uncovered
        uncovered[coverage[best_idx] == 1] = 0
        n_uncovered -= best_gain
        
    # 5. Handle remaining uncovered t-subsets if deadline hit
    if n_uncovered > 0:
        # Recalculate uncovered count to be safe
        n_uncovered = int(np.sum(uncovered))
        if n_uncovered > 0:
            remaining_indices = np.where(uncovered == 1)[0]
            for idx in remaining_indices:
                if time.time() > deadline - 1.0:
                    break
                # Find any block covering this t-subset
                candidates = np.where(coverage[:, idx] == 1)[0]
                if len(candidates) > 0:
                    # Pick the first one
                    selected_blocks.append(int(candidates[0]))
                    # Update uncovered
                    uncovered[coverage[candidates[0]] == 1] = 0

    # Convert indices back to blocks
    blocks = [list(ksubs[i]) for i in selected_blocks]
    
    return blocks
# EVOLVE-BLOCK-END
