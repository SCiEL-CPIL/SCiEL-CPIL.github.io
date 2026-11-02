import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    # Precomputation
    # Map triples to indices
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # LUT for fast index lookup: LUT[x, y, z] -> index of triple (x, y, z)
    # Since order doesn't matter, we sort x,y,z first.
    # We can just use a 3D array of size v x v x v
    LUT = np.full((v, v, v), -1, dtype=np.int32)
    for i, (a, b, c) in enumerate(triples):
        LUT[a, b, c] = i
        LUT[a, c, b] = i
        LUT[b, a, c] = i
        LUT[b, c, a] = i
        LUT[c, a, b] = i
        LUT[c, b, a] = i

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_blocks = None
    best_size = float('inf')
    
    # Run multiple greedy attempts with randomization
    # The number of attempts can be scaled by remaining time
    # But first, let's run one deterministic attempt to get a baseline
    
    def greedy_cover(randomize=False):
        uncovered = np.ones(num_triples, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            # Pick an uncovered triple to start the block
            # If randomize, pick a random uncovered triple, else the first
            uncovered_indices = np.where(uncovered)[0]
            if randomize:
                start_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            else:
                start_idx = uncovered_indices[0]
            
            start_triple = triples[start_idx]
            block = list(start_triple)
            
            # Greedily add points
            while len(block) < k:
                best_gain = -1
                best_x = -1
                # Precompute pairs in current block for gain calculation
                # Pairs are (y, z) with y < z
                pairs = list(itertools.combinations(block, 2))
                
                for x in range(v):
                    if x in block:
                        continue
                    # Calculate gain
                    # Indices of triples (x, y, z) for y, z in pairs
                    # We can vectorize this lookup
                    # Create arrays for y and z
                    ys = np.array([p[0] for p in pairs])
                    zs = np.array([p[1] for p in pairs])
                    # Lookup indices
                    idxs = LUT[x, ys, zs]
                    # Filter out invalid indices (shouldn't happen if x not in block)
                    # Sum uncovered bits
                    gain = np.sum(uncovered[idxs])
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and randomize:
                        # Random tie-breaking
                        if random.random() < 0.5:
                            best_x = x
                
                if best_x == -1:
                    break # Should not happen
                block.append(best_x)
            
            block.sort()
            blocks.append(tuple(block))
            
            # Mark triples covered
            for a, b, c in itertools.combinations(block, t):
                idx = triple_to_idx[(a, b, c)]
                uncovered[idx] = False
        
        return blocks

    # 1. Deterministic run
    blocks = greedy_cover(randomize=False)
    best_blocks = blocks
    best_size = len(blocks)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return [list(b) for b in best_blocks]
    
    # 2. Try to improve with randomization and local search (removing redundant blocks)
    
    def local_search(blocks):
        # Try to remove redundant blocks
        # A block is redundant if removing it doesn't uncover any triple
        # We can check this by counting how many blocks cover each triple
        # If count == 1, the block is essential.
        
        # This is O(N * C(v, t)) which is fast enough for N ~ 80
        triple_counts = np.zeros(num_triples, dtype=np.int32)
        block_triples = []
        for b in blocks:
            trs = []
            for comb in itertools.combinations(b, t):
                idx = triple_to_idx[comb]
                triple_counts[idx] += 1
                trs.append(idx)
            block_triples.append(trs)
        
        essential = [False] * len(blocks)
        for i, trs in enumerate(block_triples):
            for idx in trs:
                if triple_counts[idx] == 1:
                    essential[i] = True
                    break
        
        new_blocks = []
        for i, b in enumerate(blocks):
            if not essential[i]:
                # Can we remove this block?
                # If we remove it, we must ensure no triple count drops to 0
                # Since essential[i] is False, all its triples have count >= 2
                # So we can safely remove it?
                # Wait, removing it reduces counts.
                # If a triple had count 2, it becomes 1. That's fine.
                # So yes, if no triple in this block has count 1, we can remove it.
                continue
            else:
                new_blocks.append(b)
        
        return new_blocks

    # Run a few random restarts if time permits
    attempts = 0
    while time.time() < deadline:
        blocks = greedy_cover(randomize=True)
        # Apply local search to reduce size
        blocks = local_search(blocks)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
        attempts += 1
        # Limit attempts to avoid infinite loop if too fast? 
        # No, the time check handles it.
        # But if it's too fast, we might do thousands of attempts.
        # That's fine, it's cheap.
        
        # Occasionally, try a more complex local search?
        # For now, just restarts.

    return [list(b) for b in best_blocks]
