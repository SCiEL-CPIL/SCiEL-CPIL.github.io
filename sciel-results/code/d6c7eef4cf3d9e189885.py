import time
import itertools
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    # Map triple to index for faster lookups if needed, but set of tuples is fine for 2024 items
    
    def greedy_cover_with_best_extension():
        uncovered = set(all_t_subsets)
        blocks = []
        
        # Precompute for each triple, the list of points that can complete it to a block?
        # No, we pick a triple, then find best block containing it.
        
        while uncovered:
            # Pick a random uncovered triple
            seed_triple = random.choice(list(uncovered))
            seed_set = set(seed_triple)
            
            # Find the best block containing seed_triple
            # Candidates: choose k-t points from v-t remaining points
            remaining_points = [p for p in all_points if p not in seed_set]
            
            best_block = None
            best_gain = -1
            # To speed up, we can sample candidates if there are too many, but C(21,4)=5985 is small enough to iterate?
            # 5985 * 35 checks * 70 iterations = ~14 million checks. In Python, this is ~1-2 seconds per run.
            # We have ~110 seconds. So we can do ~50-100 runs. This is good.
            
            # Optimization: We don't need to check all 5985 if we are just looking for "good" coverage.
            # But for now, let's try full search over candidates for the seed triple.
            
            # We can speed up gain calculation by precomputing which triples are uncovered.
            # gain = number of triples in the block that are in `uncovered`.
            
            # Iterate through all combinations of remaining_points of size k-t
            for combo in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(seed_triple + list(combo)))
                
                # Calculate gain
                gain = 0
                for triple in itertools.combinations(candidate_block, t):
                    if triple in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = list(candidate_block)
                    
                    # Early exit if we covered all uncovered triples in this block? 
                    # Max possible gain is C(k,t). If we reach it, it's optimal for this seed.
                    if gain == len(list(itertools.combinations(range(k), t))): 
                        # Note: This check is slightly wrong because we don't know if ALL are uncovered.
                        # But if gain is high, it's good.
                        pass
            
            if best_block is None:
                # Fallback
                best_block = list(seed_triple) + remaining_points[:k-t]
            
            blocks.append(best_block)
            
            # Update uncovered
            for triple in itertools.combinations(best_block, t):
                uncovered.discard(tuple(triple))
                
        return blocks

    def local_search_remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Map each triple to the count of blocks covering it
        # And a list of block indices for each triple (to check if count==1)
        triple_counts = {triple: 0 for triple in all_t_subsets}
        triple_blocks = {triple: [] for triple in all_t_subsets}
        
        for i, block in enumerate(blocks):
            for triple in itertools.combinations(block, t):
                triple_counts[tuple(triple)] += 1
                triple_blocks[tuple(triple)].append(i)
        
        changed = True
        while changed:
            changed = False
            to_remove = []
            
            for i, block in enumerate(blocks):
                is_redundant = True
                for triple in itertools.combinations(block, t):
                    if triple_counts[tuple(triple)] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    to_remove.append(i)
            
            if to_remove:
                remove_set = set(to_remove)
                new_blocks = [b for i, b in enumerate(blocks) if i not in remove_set]
                
                # Update counts and blocks
                # Rebuilding is safer and fast enough for N~100
                triple_counts = {triple: 0 for triple in all_t_subsets}
                triple_blocks = {triple: [] for triple in all_t_subsets}
                
                for i, block in enumerate(new_blocks):
                    for triple in itertools.combinations(block, t):
                        triple_counts[tuple(triple)] += 1
                        triple_blocks[tuple(triple)].append(i)
                        
                blocks = new_blocks
                changed = True
                
        return blocks

    def local_search_swap(blocks):
        # Try to swap points to increase redundancy? 
        # Or simply try to find a block that can be removed by swapping points in another block?
        # This is complex. Let's stick to removal for now, but make the initial cover denser.
        # A simple swap: pick a random block, pick a random point in it, replace with a random point outside.
        # If coverage is maintained, accept. This might help "unstick" redundancy.
        # However, if we are in a state where no block is redundant, swapping might create one.
        
        # Let's try a simple hill climbing on the number of blocks? No, we want to minimize blocks.
        # If we can't remove, we are stuck.
        # Maybe we can try to "merge" or "replace" blocks?
        
        # Given time constraints, the main gain is from the better greedy initialization.
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline:
        blocks = greedy_cover_with_best_extension()
        blocks = local_search_remove_redundant(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks if best_blocks is not None else []
