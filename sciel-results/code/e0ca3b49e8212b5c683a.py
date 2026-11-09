import itertools
import random
import time
import sys
import numpy as np

# Global cache
_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    start_time = time.time()
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {tri: i for i, tri in enumerate(triples)}
    n_triples = len(triples)
    
    # Incidence matrix: M[i, j] is True if block i contains triple j
    # Shape: (n_blocks, n_triples)
    # Using bool to save memory and allow fast bitwise ops via numpy
    M = np.zeros((n_blocks, n_triples), dtype=bool)
    
    # Precompute triple indices for each block to speed up local search
    block_triples = [[] for _ in range(n_blocks)]
    
    for i, block in enumerate(all_blocks):
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            M[i, idx] = True
            block_triples[i].append(idx)
            
    # For faster lookup in local search: which blocks contain a given triple?
    # blocks_containing[j] is a list of block indices
    blocks_containing = [[] for _ in range(n_triples)]
    for j in range(n_triples):
        # Find indices where M[:, j] is True
        # np.nonzero is fast enough for this size
        indices = np.nonzero(M[:, j])[0]
        blocks_containing[j] = indices.tolist()

    _CACHE[key] = (all_blocks, M, block_triples, blocks_containing, n_blocks, n_triples)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, M, block_triples, blocks_containing, n_blocks, n_triples, seed):
    rng = random.Random(seed)
    # uncovered_mask is a boolean array of size n_triples
    uncovered = np.ones(n_triples, dtype=bool)
    blocks_used = []
    
    # To speed up greedy, we can precompute the number of triples each block covers
    # But gains change. We just need to find max gain each step.
    
    while np.any(uncovered):
        # gain[i] = number of uncovered triples in block i
        # M[:, uncovered] gives a (n_blocks, num_uncovered) boolean array
        # Summing along axis 1 gives the gain for each block
        # This is a large operation, but vectorized.
        # Optimization: only consider blocks that have non-zero gain?
        # Actually, M[:, uncovered].sum(axis=1) is efficient in NumPy.
        
        gains = M[:, uncovered].sum(axis=1)
        
        # Find max gain
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        # Get all blocks with max gain
        candidates = np.nonzero(gains == max_gain)[0]
        
        # Random tie-breaking
        best_idx = candidates[rng.randint(0, len(candidates) - 1)]
        
        blocks_used.append(int(best_idx))
        # Update uncovered: remove triples covered by best_idx
        uncovered[M[best_idx, :]] = False
        
    return blocks_used

def _local_search(v, k, t, initial_blocks, all_blocks, M, block_triples, blocks_containing, n_blocks, n_triples, seed, deadline):
    rng = random.Random(seed)
    
    selected = set(initial_blocks)
    # coverage_count[j] = number of selected blocks covering triple j
    coverage_count = np.zeros(n_triples, dtype=np.int32)
    
    for b in selected:
        for tri_idx in block_triples[b]:
            coverage_count[tri_idx] += 1
            
    best_size = len(selected)
    best_selected = set(selected)
    best_coverage = coverage_count.copy()
    
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # 1. Remove redundant blocks
        if selected:
            items = list(selected)
            rng.shuffle(items)
            for b in items:
                redundant = True
                # Check if any triple in b has coverage count == 1
                # vectorized check
                tris = block_triples[b]
                if any(coverage_count[ti] == 1 for ti in tris):
                    redundant = False
                if redundant:
                    for ti in tris:
                        coverage_count[ti] -= 1
                    selected.remove(b)
                    improved = True
                    no_improve_count = 0
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
                        best_coverage = coverage_count.copy()
                    break
        
        if not improved:
            # 2. Swap: Remove one block, add one block
            # Try to find a block b in selected that has many "unique" triples (coverage == 1)
            # and replace it with a block b' that covers those unique triples + more uncovered/covered triples
            # Actually, standard swap: remove b, add b'. Must maintain coverage.
            # So b' must cover all unique triples of b.
            
            sel_list = list(selected)
            rng.shuffle(sel_list)
            
            for b in sel_list:
                tris_b = block_triples[b]
                # Find unique triples in b
                unique_tris = [ti for ti in tris_b if coverage_count[ti] == 1]
                
                if not unique_tris:
                    continue # b is redundant, should have been caught above, but safe
                
                # We need a block b' that covers all unique_tris
                # Intersect the lists of blocks containing each unique triple
                # Start with the rarest triple to minimize intersection size
                
                # Sort unique_tris by frequency (length of blocks_containing)
                unique_tris_sorted = sorted(unique_tris, key=lambda x: len(blocks_containing[x]))
                
                candidates = set(blocks_containing[unique_tris_sorted[0]])
                
                for ti in unique_tris_sorted[1:]:
                    candidates.intersection_update(blocks_containing[ti])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                    
                # Filter candidates: must not be in selected (unless we allow swap with self? No)
                # Also, we want to minimize "waste" or maximize "new coverage"
                # Since we remove b, we lose coverage of non-unique triples in b.
                # We need b' to cover those too? No, other blocks might cover them.
                # Condition for valid swap:
                # 1. b' covers all unique_tris of b.
                # 2. For any triple in b that is NOT unique, it must be covered by some other block in selected - {b} OR by b'.
                
                # Let's check condition 2 efficiently.
                # Let S = selected - {b}
                # For each tri in b:
                #   if coverage_count[tri] == 1 (unique): b' MUST cover it. (Already ensured)
                #   if coverage_count[tri] > 1:
                #      if b' covers it: ok
                #      else: it must be covered by S. Since coverage_count > 1, and b covers it,
                #      at least one block in S covers it. So it's ok.
                
                # So the only constraint is that b' covers all unique_tris.
                # We just need to pick a valid b' from candidates.
                # To improve solution, we might want b' to cover MORE triples than b?
                # Or just any valid swap to escape local optimum?
                # Let's pick a random candidate to explore the space.
                
                cand_list = list(candidates)
                # Exclude b itself? If b' == b, no change.
                cand_list = [c for c in cand_list if c != b]
                
                if not cand_list:
                    continue
                
                b_prime = cand_list[rng.randint(0, len(cand_list) - 1)]
                
                # Perform swap
                # Remove b
                for ti in tris_b:
                    coverage_count[ti] -= 1
                selected.remove(b)
                
                # Add b_prime
                tris_bp = block_triples[b_prime]
                for ti in tris_bp:
                    coverage_count[ti] += 1
                selected.add(b_prime)
                
                improved = True
                no_improve_count = 0
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = set(selected)
                    best_coverage = coverage_count.copy()
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 50:
                break
    
    # Final check: ensure validity
    # The local search logic preserves validity if initialized validly.
    # Greedy produces valid solution.
    # Swap logic:
    #   If we remove b and add b', and b' covers all unique triples of b,
    #   and all non-unique triples of b are covered by S (since count > 1),
    #   then all triples in b are covered by S U {b'}.
    #   Triples not in b are covered by S (since b didn't cover them) or by b'.
    #   So validity is preserved.
    
    return list(best_selected)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115 # 5s safety margin
    
    # Precompute
    all_blocks, M, block_triples, blocks_containing, n_blocks, n_triples = _precompute(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    
    # Run multiple restarts
    while time.time() < deadline - 5:
        # 1. Greedy
        initial_blocks = _greedy_construct(v, k, t, all_blocks, M, block_triples, blocks_containing, n_blocks, n_triples, seed)
        
        if not initial_blocks:
            seed += 1
            continue
        
        # 2. Local Search
        remaining = deadline - time.time()
        # Allocate time to local search
        ls_time = min(10.0, remaining - 2.0)
        if ls_time < 1.0:
            break
        ls_deadline = time.time() + ls_time
            
        blocks = _local_search(v, k, t, initial_blocks, all_blocks, M, block_triples, blocks_containing, n_blocks, n_triples, seed, ls_deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            # print(f"New best: {best_len} blocks at t={time.time()-start_time:.1f}s", file=sys.stderr)
        
        seed += 1
    
    if best_blocks is None:
        # Fallback: just greedy with a new seed
        blocks = _greedy_construct(v, k, t, all_blocks, M, block_triples, blocks_containing, n_blocks, n_triples, 999)
        best_blocks = blocks
    
    result = [list(all_blocks[idx]) for idx in best_blocks]
    
    return result
