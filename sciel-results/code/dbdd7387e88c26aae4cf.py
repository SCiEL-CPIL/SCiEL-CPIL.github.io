import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    # Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Precompute for each pair, the indices of triples containing that pair
    # pair_id can be the tuple itself or a flattened index
    # Since v is small, using tuple keys in a dict is fine, or we can map pairs to indices too
    pairs = list(itertools.combinations(range(v), 2))
    pair_to_idx = {p: i for i, p in enumerate(pairs)}
    
    # pair_idxs[pair_id] is an array of length v, where entry x is the index of triple (pair + x)
    # or -1 if x is in pair
    pair_idxs = np.full((len(pairs), v), -1, dtype=int)
    for p_id, p in enumerate(pairs):
        for x in range(v):
            if x in p:
                continue
            triple = tuple(sorted(p + (x,)))
            pair_idxs[p_id, x] = triple_to_idx[triple]
            
    # Data structure for fast access
    data = {
        'num_triples': num_triples,
        'triple_to_idx': triple_to_idx,
        'pair_to_idx': pair_to_idx,
        'pair_idxs': pair_idxs,
        'all_triples': all_triples
    }
    
    def run_greedy(seed):
        rng = random.Random(seed)
        uncovered = np.ones(num_triples, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Pick a random uncovered triple to seed the block
            uncovered_idxs = np.where(uncovered)[0]
            start_idx = rng.choice(uncovered_idxs)
            B = list(all_triples[start_idx])
            
            # Extend B to size k
            while len(B) < k:
                gains = np.zeros(v, dtype=int)
                # For each pair in B, add contributions
                for i in range(len(B)):
                    for j in range(i+1, len(B)):
                        p = (B[i], B[j])
                        p_id = data['pair_to_idx'][p]
                        # Add 1 to gains[x] for each x such that triple(p, x) is uncovered
                        # pair_idxs[p_id] contains indices, -1 for invalid
                        # We need to map -1 to a safe index or mask it
                        # Let's use a temporary array
                        idxs = data['pair_idxs'][p_id]
                        # Create a mask for valid indices
                        valid = idxs >= 0
                        if np.any(valid):
                            # Get the uncovered status for these indices
                            # We need to add to gains[x] if uncovered[idxs[x]] is True
                            # This is a scatter-add
                            # gains[valid_indices] += uncovered[idxs[valid_indices]]
                            # But we need to do this for all x
                            # We can do:
                            # temp = np.zeros(v)
                            # temp[valid] = uncovered[idxs[valid]]
                            # gains += temp
                            # This is O(v) per pair.
                            # Since v=24, this is fine.
                            temp = np.zeros(v, dtype=int)
                            temp[valid] = uncovered[idxs[valid]].astype(int)
                            gains += temp
                
                # Find best x
                max_gain = np.max(gains)
                if max_gain == 0:
                    # No gain, pick random point not in B
                    candidates = [x for x in range(v) if x not in B]
                    if not candidates:
                        break
                    x = rng.choice(candidates)
                else:
                    # Pick random among max gain candidates
                    max_candidates = np.where(gains == max_gain)[0]
                    max_candidates = [x for x in max_candidates if x not in B]
                    if not max_candidates:
                         # Fallback
                         candidates = [x for x in range(v) if x not in B]
                         x = rng.choice(candidates)
                    else:
                        x = rng.choice(max_candidates)
                
                B.append(x)
            
            # Add block
            B.sort()
            blocks.append(B)
            
            # Update uncovered
            for triple in itertools.combinations(B, t):
                idx = data['triple_to_idx'][triple]
                uncovered[idx] = False
        
        return blocks

    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Map each block to the set of triple indices it covers
        block_triple_idxs = []
        for B in blocks:
            idxs = set()
            for triple in itertools.combinations(B, t):
                idxs.add(data['triple_to_idx'][triple])
            block_triple_idxs.append(idxs)
        
        # Count coverage
        coverage_count = np.zeros(num_triples, dtype=int)
        for idxs in block_triple_idxs:
            for idx in idxs:
                coverage_count[idx] += 1
        
        redundant_blocks = []
        changed = True
        while changed:
            changed = False
            for i, idxs in enumerate(block_triple_idxs):
                if i in redundant_blocks:
                    continue
                # Check if all triples in this block are covered by others
                # i.e., coverage_count[idx] > 1 for all idx in idxs
                redundant = True
                for idx in idxs:
                    if coverage_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    redundant_blocks.append(i)
                    changed = True
                    # Decrement coverage count
                    for idx in idxs:
                        coverage_count[idx] -= 1
                    block_triple_idxs[i] = None # Mark as removed
        
        # Filter out redundant blocks
        new_blocks = [B for i, B in enumerate(blocks) if i not in redundant_blocks]
        return new_blocks

    best_blocks = None
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    seed = 0
    
    while time.time() < deadline:
        blocks = run_greedy(seed)
        blocks = remove_redundant(blocks)
        
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
        
        seed += 1
        
        # If we found a very good solution (e.g., close to lower bound), we might stop early?
        # Lower bound is 69. Best known is 76.
        # If we find 76, we can keep going to see if we find 75.
        # But let's just run until deadline.
        
    return best_blocks
