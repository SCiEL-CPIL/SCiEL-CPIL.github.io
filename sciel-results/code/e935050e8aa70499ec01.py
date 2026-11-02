import itertools
import time
import random
import sys

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    # Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # Precompute: for each pair (a,b), the bitmask of triples containing that pair
    # triple_mask_for_pair[(a,b)] = bitmask of all triples {a,b,x} for x in 0..v-1, x!=a,b
    triple_mask_for_pair = {}
    for a in range(v):
        for b in range(a+1, v):
            mask = 0
            for x in range(v):
                if x == a or x == b:
                    continue
                tri = tuple(sorted((a, b, x)))
                idx = subset_to_idx[tri]
                mask |= (1 << idx)
            triple_mask_for_pair[(a, b)] = mask
    
    # Precompute: for each point x, the list of pairs (y,z) with y,z != x
    # Actually, we need: for a block B and candidate x, gain = number of uncovered triples {x,y,z} where {y,z} subset of B
    # = popcount(uncovered & OR of triple_mask_for_pair[(y,z)] for {y,z} subset of B, y,z != x)
    
    # For efficiency, precompute for each block size m, the pairs in the block
    # But blocks change, so we compute on the fly but use the precomputed masks
    
    best_blocks = None
    best_count = float('inf')
    
    def greedy_construct(seed_rng):
        blocks = []
        local_uncovered = (1 << n_subsets) - 1
        
        def local_is_uncovered(sub):
            idx = subset_to_idx[sub]
            return (local_uncovered >> idx) & 1
            
        def local_mark_covered(sub):
            nonlocal local_uncovered
            idx = subset_to_idx[sub]
            local_uncovered &= ~(1 << idx)
            
        def local_mark_covered_mask(mask):
            nonlocal local_uncovered
            local_uncovered &= ~mask

        while local_uncovered != 0:
            if time.time() > deadline:
                return None
            
            # Select a seed subset from uncovered - pick a random uncovered triple
            seed_idx = -1
            while seed_idx == -1:
                candidate = seed_rng.randint(0, n_subsets - 1)
                if (local_uncovered >> candidate) & 1:
                    seed_idx = candidate
            
            seed_sub = all_subsets[seed_idx]
            block = list(seed_sub)
            block_set = set(block)
            
            # Greedily add points to maximize new coverage
            while len(block) < k:
                block_set = set(block)
                best_point = None
                best_gain = -1
                
                # Candidates are all points not in block
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Gain = number of uncovered triples {x, y, z} where {y,z} subset of block
                    # For each pair (y,z) in block, the triples {x,y,z} are covered by triple_mask_for_pair[(y,z)] & (1<<idx_x)
                    # But we need to OR all masks for pairs in block, then count bits in uncovered
                    # Actually, for fixed x, the triples containing x and two elements from block are:
                    # For each pair (y,z) in block, triple (x,y,z) - but x must not equal y or z
                    # So we need pairs (y,z) from block where y,z != x (which is guaranteed since x not in block)
                    
                    # Compute the union of triple masks for all pairs in block
                    # This is expensive: C(m,2) OR operations
                    # Alternative: precompute for each point x, which triples contain x
                    # triples_containing_x[x] = bitmask of all triples {x, a, b}
                    
                    # Let me use a different approach: for each pair (y,z) in block, 
                    # the triple (x,y,z) is at index subset_to_idx[tuple(sorted((x,y,z)))]
                    # We can check if it's uncovered directly
                    
                    gain = 0
                    for y, z in itertools.combinations(block, t - 1):
                        # y, z are distinct, and x is distinct from both (x not in block)
                        tri = tuple(sorted((x, y, z)))
                        idx = subset_to_idx[tri]
                        if (local_uncovered >> idx) & 1:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        
                if best_point is None or best_gain <= 0:
                    available = [x for x in range(v) if x not in block_set]
                    if available:
                        best_point = seed_rng.choice(available)
                    else:
                        break
                block.append(best_point)
                
            if len(block) < k:
                block_set = set(block)
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block_set:
                        block.append(x)
                        
            block.sort()
            blocks.append(block)
            
            # Update uncovered: mark all triples in this block as covered
            for sub in itertools.combinations(block, t):
                local_mark_covered(sub)
                
        return blocks

    def remove_redundant(blocks):
        """Remove blocks that are redundant (all their triples covered by other blocks)"""
        if not blocks:
            return blocks
        
        n = len(blocks)
        # For each block, compute which triples it covers
        block_triples = []
        for b in blocks:
            triples = list(itertools.combinations(b, t))
            block_triples.append(triples)
        
        # For each triple, find which blocks cover it
        triple_coverers = {}  # triple -> set of block indices
        for i, triples in enumerate(block_triples):
            for tri in triples:
                if tri not in triple_coverers:
                    triple_coverers[tri] = set()
                triple_coverers[tri].add(i)
        
        # A block is redundant if for every triple it covers, there's another block covering it
        keep = [True] * n
        for i in range(n):
            redundant = True
            for tri in block_triples[i]:
                coverers = triple_coverers.get(tri, set())
                if len(coverers) == 1 and i in coverers:
                    # This triple is only covered by block i
                    redundant = False
                    break
            if redundant:
                keep[i] = False
        
        return [blocks[i] for i in range(n) if keep[i]]
    
    def local_search(blocks, seed_rng, max_iters=50):
        """Try to improve by removing redundant blocks or finding better alternatives"""
        blocks = list(blocks)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try to remove redundant blocks
            new_blocks = remove_redundant(blocks)
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                continue
            
            # Try to find a block that can be replaced by a better one
            # For each block, try removing it and see if the remaining blocks still cover everything
            # If not, try to find a single block that covers the missing triples
            # This is complex; skip for now and just do redundancy removal
            
            break  # No more easy improvements
        
        return blocks

    # Run multiple greedy restarts with different seeds
    num_restarts = 0
    while time.time() < deadline - 30:
        seed_rng = random.Random(num_restarts)
        blocks = greedy_construct(seed_rng)
        if blocks is None:
            continue
        
        # Post-processing: Remove redundant blocks and try local search
        pruned_blocks = remove_redundant(blocks)
        pruned_blocks = local_search(pruned_blocks, seed_rng, max_iters=20)
        
        if len(pruned_blocks) < best_count:
            best_count = len(pruned_blocks)
            best_blocks = pruned_blocks
        
        num_restarts += 1
    
    if best_blocks is None:
        if time.time() < deadline - 5:
            seed_rng = random.Random(999)
            blocks = greedy_construct(seed_rng)
            if blocks:
                return blocks
        return []
        
    return best_blocks
