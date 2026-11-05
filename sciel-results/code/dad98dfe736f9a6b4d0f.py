import itertools
import random
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a greedy heuristic followed by local search.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets (triples)
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Helper to get triples in a block
    def get_block_triples(block):
        return [triple_to_idx[tr] for tr in itertools.combinations(block, t)]

    # 1. Greedy Construction
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = set(range(num_triples))
        blocks = []
        
        # Precompute for each triple, the list of points it contains? No, we need to find blocks containing it.
        # Instead, for each uncovered triple, we find the best block.
        
        # To speed up, we can maintain a count of uncovered triples for each potential block?
        # That's C(24,7) ~ 346k blocks. Too big to precompute fully for each step.
        
        # Greedy step:
        while uncovered:
            # Pick a random uncovered triple to break symmetry
            target_triple_idx = random.choice(list(uncovered))
            target_triple = all_triples[target_triple_idx]
            
            best_block = None
            max_gain = -1
            
            # Candidates must contain target_triple
            # target_triple has t points. We need k-t more points.
            remaining_points = [p for p in points if p not in target_triple]
            
            # If k-t is small, iterate combinations.
            # Here k=7, t=3, so we need 4 more points from 21.
            # C(21, 4) = 5985 candidates.
            
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_triple + combo))
                
                # Calculate gain: number of uncovered triples covered by this block
                gain = 0
                for tr_idx in get_block_triples(block):
                    if tr_idx in uncovered:
                        gain += 1
                
                if gain > max_gain:
                    max_gain = gain
                    best_block = block
            
            if best_block is not None:
                blocks.append(list(best_block))
                for tr_idx in get_block_triples(best_block):
                    uncovered.discard(tr_idx)
                
        return blocks

    # Initial greedy solution
    best_blocks = greedy_construct(42)
    
    # 2. Local Search
    # State: list of blocks, cover_count array
    def evaluate_and_prune(blocks):
        """
        Removes redundant blocks. Returns new list of blocks.
        """
        if not blocks:
            return blocks
            
        # Count coverage
        cover_count = [0] * num_triples
        block_triples_list = []
        
        for b in blocks:
            trs = get_block_triples(b)
            block_triples_list.append(trs)
            for tr_idx in trs:
                cover_count[tr_idx] += 1
                
        # Check for redundancy
        new_blocks = []
        new_block_triples = []
        
        for i, b in enumerate(blocks):
            # Check if any triple in this block has count == 1
            is_redundant = True
            for tr_idx in block_triples_list[i]:
                if cover_count[tr_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove it
                for tr_idx in block_triples_list[i]:
                    cover_count[tr_idx] -= 1
            else:
                new_blocks.append(b)
                new_block_triples.append(block_triples_list[i])
                
        return new_blocks, cover_count, new_block_triples

    # Local search loop
    current_blocks = best_blocks
    current_cover_count, current_block_triples = evaluate_and_prune(current_blocks)[1:]
    
    # We need to handle the case where evaluate_and_prune returns 3 things
    # Let's refactor evaluate_and_prune to return (blocks, counts, triples_list)
    
    def prune_blocks(blocks):
        if not blocks:
            return blocks, [0]*num_triples, []
            
        cover_count = [0] * num_triples
        block_triples_list = []
        
        for b in blocks:
            trs = get_block_triples(b)
            block_triples_list.append(trs)
            for tr_idx in trs:
                cover_count[tr_idx] += 1
                
        new_blocks = []
        new_block_triples = []
        
        for i, b in enumerate(blocks):
            is_redundant = True
            for tr_idx in block_triples_list[i]:
                if cover_count[tr_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                for tr_idx in block_triples_list[i]:
                    cover_count[tr_idx] -= 1
            else:
                new_blocks.append(b)
                new_block_triples.append(block_triples_list[i])
                
        return new_blocks, cover_count, new_block_triples

    current_blocks, current_cover_count, current_block_triples = prune_blocks(current_blocks)
    best_len = len(current_blocks)
    
    # Precompute point indices for faster combination generation?
    # We will generate candidates on the fly.
    
    stuck_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to prune again (in case previous swaps made blocks redundant)
        prev_len = len(current_blocks)
        current_blocks, current_cover_count, current_block_triples = prune_blocks(current_blocks)
        if len(current_blocks) < prev_len:
            improved = True
            best_len = len(current_blocks)
            stuck_count = 0
            continue
            
        # 2. Try to swap a block to enable pruning
        # For each block, find its exclusive triples
        # If exclusive triples can be covered by another block that "helps" more, swap
        
        n_blocks = len(current_blocks)
        if n_blocks == 0: break
        
        # Pick a random block to try to replace
        idx = random.randint(0, n_blocks - 1)
        block_to_replace = current_blocks[idx]
        exclusive_triples = []
        
        for tr_idx in current_block_triples[idx]:
            if current_cover_count[tr_idx] == 1:
                exclusive_triples.append(tr_idx)
                
        if not exclusive_triples:
            # This block is redundant, should have been pruned.
            # Force prune
            current_blocks.pop(idx)
            # Recalculate
            current_blocks, current_cover_count, current_block_triples = prune_blocks(current_blocks)
            improved = True
            best_len = len(current_blocks)
            stuck_count = 0
            continue
            
        # Find candidate blocks that cover all exclusive_triples
        # Union of points in exclusive_triples
        union_points = set()
        for tr_idx in exclusive_triples:
            union_points.update(all_triples[tr_idx])
            
        if len(union_points) > k:
            # Cannot cover with a single block
            stuck_count += 1
            if stuck_count > 50:
                # Perturb: replace a random block with a random valid block?
                # Or just restart greedy with new seed?
                # Let's try a random swap to escape local opt
                if random.random() < 0.1:
                    # Randomly replace a block
                    r_idx = random.randint(0, len(current_blocks)-1)
                    r_block = current_blocks[r_idx]
                    # Generate a random block containing the exclusive triples of r_block?
                    # No, just a random block.
                    # This might break coverage.
                    # Safer: just break and restart greedy
                    break
                continue
                
        # Generate candidates
        # Points in union_points are fixed.
        # Need k - len(union_points) more points from remaining.
        remaining_points = [p for p in points if p not in union_points]
        num_to_add = k - len(union_points)
        
        best_candidate = None
        best_score = -1
        
        # Limit candidates to avoid timeout if num_to_add is large
        # Here max num_to_add is 7 (if union is empty, but exclusive is not empty so union>=3)
        # Min union size is 3 (one triple). Max add is 4. C(21,4)=5985.
        # This is fast.
        
        for combo in itertools.combinations(remaining_points, num_to_add):
            cand_block = tuple(sorted(union_points + combo))
            
            # Check if this candidate is already in the list?
            # If it is, swapping is pointless.
            if cand_block in current_blocks:
                continue
                
            # Calculate score:
            # We want to maximize the number of triples in cand_block that are covered by OTHER blocks.
            # This increases redundancy elsewhere, potentially allowing deletion.
            
            score = 0
            valid = True
            
            for tr_idx in get_block_triples(cand_block):
                if tr_idx in [t for t in current_block_triples[idx]]:
                    # This is an exclusive triple of the block we are replacing.
                    # It will be covered by cand_block.
                    # We don't count this as "help".
                    continue
                else:
                    # This triple is not exclusive to the block we are replacing.
                    # If it is covered by at least one other block, it helps.
                    # Note: current_cover_count includes the block we are replacing.
                    # So if count > 1, it is covered by others.
                    if current_cover_count[tr_idx] > 1:
                        score += 1
                        # Also, if this triple was exclusive to ANOTHER block, that's even better?
                        # No, just counting redundancy is a good proxy.
                        
            # We also prefer candidates that are "diverse" or cover many triples?
            # The score above is good.
            
            if score > best_score:
                best_score = score
                best_candidate = cand_block
                
        if best_candidate is not None:
            # Perform swap
            current_blocks[idx] = list(best_candidate)
            
            # Prune
            prev_len = len(current_blocks)
            current_blocks, current_cover_count, current_block_triples = prune_blocks(current_blocks)
            
            if len(current_blocks) < prev_len:
                improved = True
                best_len = len(current_blocks)
                stuck_count = 0
            else:
                # No improvement, but we swapped.
                # We might be stuck in a cycle.
                # To avoid cycles, we could keep a history, but for simplicity,
                # we just let it run. The random selection of idx helps.
                pass
        else:
            stuck_count += 1
            if stuck_count > 100:
                # Try a different block or restart
                stuck_count = 0
                # Randomly pick a new block to try
                continue
                
    return current_blocks
