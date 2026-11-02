import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    # Set a deadline slightly before the 150s hard limit to allow for cleanup/return
    deadline = start_time + 145.0 

    points = range(v)
    
    # Precompute all possible blocks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute triples for each block
    # block_triples[i] is a set of tuples (a, b, c)
    block_triples = []
    # Precompute which blocks contain each point
    # point_to_blocks[p] is a list of indices into all_blocks
    point_to_blocks = {p: [] for p in points}
    
    for i, block in enumerate(all_blocks):
        triples = set(itertools.combinations(block, t))
        block_triples.append(triples)
        for p in block:
            point_to_blocks[p].append(i)
            
    # Initialize the set of uncovered triples
    all_triples = set(itertools.combinations(points, t))
    uncovered = set(all_triples)
    
    selected_blocks = []
    used_block_indices = set()
    
    while uncovered:
        # Check time limit
        if time.time() > deadline:
            break
            
        # 1. Calculate the degree of each point in the residual hypergraph
        # degree[p] = number of uncovered triples containing p
        degrees = {p: 0 for p in points}
        for triple in uncovered:
            for p in triple:
                degrees[p] += 1
        
        # 2. Find the point with the minimum non-zero degree (the "bottleneck")
        min_degree = float('inf')
        bottleneck_point = None
        for p in points:
            deg = degrees[p]
            if 0 < deg < min_degree:
                min_degree = deg
                bottleneck_point = p
                
        if bottleneck_point is None:
            # If no point has a positive degree, uncovered must be empty (handled by while loop)
            # Or something went wrong. Break to avoid infinite loop.
            break
            
        # 3. Among all blocks containing the bottleneck point, find the one
        #    that covers the most uncovered triples.
        best_block_idx = -1
        best_gain = 0
        
        candidate_indices = point_to_blocks[bottleneck_point]
        
        for idx in candidate_indices:
            if idx in used_block_indices:
                continue
                
            # Calculate gain: number of uncovered triples covered by this block
            gain = 0
            for triple in block_triples[idx]:
                if triple in uncovered:
                    gain += 1
                    
            if gain > best_gain:
                best_gain = gain
                best_block_idx = idx
                
        if best_block_idx == -1 or best_gain == 0:
            # If no block can cover any new triple (should not happen if logic is correct),
            # fallback to global greedy to ensure progress
            global_best_idx = -1
            global_best_gain = 0
            for idx in range(num_blocks):
                if idx in used_block_indices:
                    continue
                gain = 0
                for triple in block_triples[idx]:
                    if triple in uncovered:
                        gain += 1
                if gain > global_best_gain:
                    global_best_gain = gain
                    global_best_idx = idx
                    
            if global_best_idx == -1:
                break
            
            best_block_idx = global_best_idx
            best_gain = global_best_gain
            
        # 4. Select the block and update state
        selected_blocks.append(list(all_blocks[best_block_idx]))
        used_block_indices.add(best_block_idx)
        uncovered.difference_update(block_triples[best_block_idx])
        
    return selected_blocks
