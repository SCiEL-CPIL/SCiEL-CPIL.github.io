import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precomputation
    # Map 7-subset to index
    # We will use tuples of points as keys for a dictionary, or map to integer indices.
    # Given v=24, k=7, there are 346,104 blocks.
    
    # 1. Generate all blocks and their triple coverage
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map triple to index
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # block_to_triples: list of lists of triple indices
    block_to_triples = [None] * n_blocks
    # block_index: map from block tuple to index
    block_index = {block: i for i, block in enumerate(all_blocks)}
    
    # triple_to_blocks: list of lists of block indices
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(all_blocks):
        triples_in_block = []
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            triples_in_block.append(idx)
            triple_to_blocks[idx].append(i)
        block_to_triples[i] = triples_in_block

    # Greedy Construction
    def greedy_cover():
        uncovered = set(range(n_triples))
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            # Pick a random uncovered triple
            seed_triple = random.choice(list(uncovered))
            
            # Get all blocks covering this triple
            candidates = triple_to_blocks[seed_triple]
            
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                gain = 0
                for t_idx in block_to_triples[b_idx]:
                    if t_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                return None # Should not happen if valid cover exists
            
            blocks.append(best_block_idx)
            
            # Update uncovered
            for t_idx in block_to_triples[best_block_idx]:
                uncovered.discard(t_idx)
                
        return blocks

    # Local Search
    def local_search(blocks):
        if not blocks:
            return blocks
            
        # cover_count: array of size n_triples
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b_idx in blocks:
            for t_idx in block_to_triples[b_idx]:
                cover_count[t_idx] += 1
                
        improved = list(blocks)
        changed = True
        no_improve_count = 0
        max_no_improve = 50
        
        while changed and time.time() < deadline:
            changed = False
            
            # Phase 1: Remove redundant blocks
            i = 0
            while i < len(improved):
                b_idx = improved[i]
                redundant = True
                for t_idx in block_to_triples[b_idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for t_idx in block_to_triples[b_idx]:
                        cover_count[t_idx] -= 1
                    improved.pop(i)
                    changed = True
                else:
                    i += 1
                    
            if not improved:
                break
                
            # Phase 2: Swap to remove
            # Identify blocks with few unique triples
            # For each block, check if we can swap it with another to make some block redundant
            
            # To speed up, we can pick a random block and try to swap it
            # Or iterate through all blocks
            
            # Let's try iterating through blocks
            swap_done = False
            for i in range(len(improved)):
                if time.time() > deadline:
                    break
                
                b_idx = improved[i]
                
                # Find unique triples for this block
                unique_triples = [t_idx for t_idx in block_to_triples[b_idx] if cover_count[t_idx] == 1]
                
                if not unique_triples:
                    # This block is redundant? Should have been removed in Phase 1
                    # But maybe cover_count changed? No, Phase 1 ran.
                    # Actually, if unique_triples is empty, it means all its triples are covered by others.
                    # So it is redundant.
                    # Why didn't Phase 1 remove it?
                    # Maybe it was added later? No, we are in a loop.
                    # Ah, Phase 1 removes *all* redundant blocks.
                    # So if we are here, unique_triples should not be empty?
                    # Wait, if a block has NO unique triples, it is redundant.
                    # So Phase 1 should have removed it.
                    # So unique_triples must be non-empty.
                    pass

                # Get union of points in unique_triples
                # We need to map triple idx back to points
                # We have all_triples list.
                points_in_unique = set()
                for t_idx in unique_triples:
                    points_in_unique.update(all_triples[t_idx])
                
                if len(points_in_unique) > k:
                    continue # Cannot cover with single block
                
                # Generate candidate blocks containing points_in_unique
                # We need to find blocks that are supersets of points_in_unique
                # We can iterate through all_blocks? Too slow.
                # We can use the triple_to_blocks structure?
                # If points_in_unique has size m, we need blocks containing all m points.
                # This is an intersection of lists.
                
                # Optimization: If points_in_unique is large, few candidates.
                # If small, many candidates.
                
                # Let's just sample a few candidates from all_blocks?
                # No, we must contain points_in_unique.
                
                # Find one triple in unique_triples.
                # Get all blocks containing that triple.
                # Filter those that contain all of points_in_unique.
                
                seed_t = unique_triples[0]
                candidates = triple_to_blocks[seed_t]
                
                valid_candidates = []
                for c_idx in candidates:
                    # Check if c_idx contains all points in points_in_unique
                    # We don't have a fast "contains set" check for block idx.
                    # We can precompute a bitmask for each block?
                    # 24 bits fits in int.
                    pass
