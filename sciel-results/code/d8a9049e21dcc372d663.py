import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Precompute all triples
    all_triples = list(itertools.combinations(range(v), t))
    triple_set = set(all_triples)
    
    best_blocks = None
    best_len = float('inf')
    
    # Helper: Greedy Construction
    def greedy_construct(seed_val):
        random.seed(seed_val)
        uncovered = set(triple_set)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to anchor the block
            # Converting to list/tuple for random.choice is O(N)
            # Since N is small (max 2024), this is fine.
            anchor_triple = random.choice(tuple(uncovered))
            
            # Generate all k-subsets containing anchor_triple
            remaining_points = [p for p in range(v) if p not in anchor_triple]
            
            best_block = None
            best_gain = -1
            best_gain_blocks = []
            
            # Choose k-t points from the remaining_points
            for combo in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(anchor_triple + combo))
                
                # Calculate gain
                gain = 0
                for tri in itertools.combinations(candidate_block, t):
                    if tri in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_gain_blocks = [candidate_block]
                elif gain == best_gain:
                    best_gain_blocks.append(candidate_block)
            
            # Pick randomly from best gain blocks to diversify
            selected_block = random.choice(best_gain_blocks)
            blocks.append(selected_block)
            
            # Update uncovered
            for tri in itertools.combinations(selected_block, t):
                if tri in uncovered:
                    uncovered.discard(tri)
                    
        return blocks

    # Helper: Reduce Blocks
    def reduce_blocks(blocks):
        # Map each triple to the number of blocks covering it
        coverage = defaultdict(int)
        block_triples = [] # List of list of triples for each block
        
        for b in blocks:
            tris = list(itertools.combinations(b, t))
            block_triples.append(tris)
            for tri in tris:
                coverage[tri] += 1
        
        improved = True
        # We might need multiple passes to remove blocks that become redundant
        # only after other blocks are removed.
        while improved:
            improved = False
            
            new_blocks = []
            new_block_triples = []
            
            for i, b in enumerate(blocks):
                tris = block_triples[i]
                # Check if all triples in this block are covered by other blocks
                # i.e., coverage[tri] > 1 for all tri in tris
                removable = True
                for tri in tris:
                    if coverage[tri] <= 1:
                        removable = False
                        break
                
                if removable:
                    # Remove this block
                    for tri in tris:
                        coverage[tri] -= 1
                    improved = True
                else:
                    new_blocks.append(b)
                    new_block_triples.append(tris)
            
            blocks = new_blocks
            block_triples = new_block_triples
            
        return blocks

    # Main Loop
    attempt = 0
    while time.time() < deadline:
        blocks = greedy_construct(attempt)
        reduced = reduce_blocks(blocks)
        
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_blocks = reduced
            
        attempt += 1
        
    if best_blocks is None:
        # Fallback
        blocks = greedy_construct(0)
        best_blocks = reduce_blocks(blocks)
        
    return [list(b) for b in best_blocks]
