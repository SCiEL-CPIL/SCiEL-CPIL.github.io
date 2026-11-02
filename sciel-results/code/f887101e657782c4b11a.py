import itertools
import numpy as np
import time


def construct(v, k, t):
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Precompute all t-subsets
    all_triples = np.array(list(itertools.combinations(range(v), t)))
    n_triples = len(all_triples)
    
    # Map each triple to an index
    triple_to_idx = {tuple(triple): i for i, triple in enumerate(all_triples)}
    
    # Track which triples are uncovered (1 = uncovered, 0 = covered)
    uncovered = np.ones(n_triples, dtype=np.uint8)
    uncovered_count = n_triples
    
    blocks = []
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        # Find the first uncovered triple to seed a block
        first_uncovered_idx = np.argmax(uncovered)
        seed_triple = all_triples[first_uncovered_idx]
        block = list(seed_triple)
        
        # Greedily add points to maximize coverage of uncovered triples
        while len(block) < k:
            if time.time() > deadline:
                break
            
            block_arr = np.array(block)
            # For each candidate point, compute how many new triples it would cover
            best_gain = -1
            best_point = -1
            
            for x in range(v):
                if x in block_arr:
                    continue
                
                # New triples = combinations of (t-1) elements from block + x
                # that are currently uncovered
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    triple = tuple(sorted(rest + (x,)))
                    idx = triple_to_idx.get(triple)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1 or best_gain == 0:
                # Pick any point not in block
                for x in range(v):
                    if x not in block_arr:
                        best_point = x
                        break
            
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Mark all triples in this block as covered
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[tuple(triple)]
            if uncovered[idx]:
                uncovered[idx] = 0
                uncovered_count -= 1
    
    # Post-processing: remove redundant blocks
    if time.time() < deadline - 5:
        blocks = _remove_redundant_blocks(blocks, v, k, t, deadline)
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t, deadline):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks
    
    # For each triple, track which blocks cover it
    triple_to_blocks = {}
    for i, block in enumerate(blocks):
        for triple in itertools.combinations(block, t):
            triple_to_blocks.setdefault(tuple(triple), []).append(i)
    
    # A block is redundant if no triple is covered only by it
    redundant = [False] * len(blocks)
    
    for triple, block_indices in triple_to_blocks.items():
        if len(block_indices) == 1:
            redundant[block_indices[0]] = False  # This block is essential
        # else: all blocks covering this triple are potentially redundant
    
    # Keep only non-redundant blocks
    kept = [blocks[i] for i in range(len(blocks)) if not redundant[i]]
    
    # Verify coverage is still complete
    if kept:
        covered = set()
        for block in kept:
            for triple in itertools.combinations(block, t):
                covered.add(tuple(triple))
        
        all_triples_needed = set(itertools.combinations(range(v), t))
        if covered == all_triples_needed:
            return kept
    
    # If removal broke coverage, return original
    return blocks
