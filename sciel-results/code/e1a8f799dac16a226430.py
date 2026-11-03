import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute all t-subsets (triples for this task)
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Helper to get the set of triple indices covered by a block
    def get_block_triple_indices(block):
        indices = set()
        for triple in itertools.combinations(block, t):
            indices.add(triple_to_idx[triple])
        return indices

    def greedy_initial_cover(rng):
        """
        Construct a valid covering using a greedy algorithm.
        """
        uncovered = set(range(n_triples))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to target
            target_idx = rng.choice(list(uncovered))
            target_triple = all_triples[target_idx]
            
            # We need to find a block of size k containing target_triple
            # that maximizes the number of newly covered triples.
            
            remaining_points = [p for p in range(v) if p not in target_triple]
            best_block = None
            best_gain = -1
            
            # To speed up, we can sample combinations if v is large, but v=24 is small.
            # C(21, 4) = 5985 combinations. This is fast.
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_triple + combo))
                
                # Calculate gain
                gain = 0
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[triple]
                    if idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    
            if best_block is None:
                # Fallback: just add a random block containing the target
                combo = tuple(rng.sample(remaining_points, k - t))
                best_block = tuple(sorted(target_triple + combo))
            
            blocks.append(best_block)
            
            # Update uncovered
            for triple in itertools.combinations(best_block, t):
                idx = triple_to_idx[triple]
                uncovered.discard(idx)
                
        return blocks

    def remove_redundant(blocks, counts):
        """
        Remove redundant blocks iteratively.
        Returns new blocks and updated counts.
        """
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for i, b in enumerate(blocks):
                # Check if block i is redundant
                redundant = True
                for triple in itertools.combinations(b, t):
                    idx = triple_to_idx[triple]
                    if counts[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Update counts
                    for triple in itertools.combinations(b, t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
        return blocks, counts

    # Initialize
    rng = random.Random(42)
    
    # Generate initial cover
    blocks = greedy_initial_cover(rng)
    
    # Calculate initial counts
    counts = [0] * n_triples
    for b in blocks:
        for triple in itertools.combinations(b, t):
            idx = triple_to_idx[triple]
            counts[idx] += 1
            
    # Initial polish
    blocks, counts = remove_redundant(blocks, counts)
    
    best_blocks = [list(b) for b in blocks]
    best_len = len(blocks)
    
    # Simulated Annealing / Local Search
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    # Precompute list of triples for each block for faster access in SA loop?
    # Or just compute on the fly. C(7,3)=35 is small.
    
    while time.time() < deadline:
        # Try to remove a redundant block
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            b = blocks[i]
            redundant = True
            for triple in itertools.combinations(b, t):
                idx = triple_to_idx[triple]
                if counts[idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                for triple in itertools.combinations(b, t):
                    idx = triple_to_idx[triple]
                    counts[idx] -= 1
                blocks.pop(i)
                
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                continue # Restart loop immediately after a successful removal
        
        # Try to swap a block
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            
            # Find critical triples (count == 1)
            critical_points = set()
            for triple in itertools.combinations(old_block, t):
                idx = triple_to_idx[triple]
                if counts[idx] == 1:
                    critical_points.update(triple)
            
            # If critical points exceed block size, we can't replace this block with a single block
            # that maintains validity. Skip.
            if len(critical_points) > k:
                continue
            
            # Construct new block
            new_block_set = set(critical_points)
            # We need to add k - len(critical_points) more points to maximize coverage
            remaining = [p for p in range(v) if p not in new_block_set]
            rng.shuffle(remaining)
            
            # Greedily add points to maximize gain
            # Gain: number of triples with count == 0 (impossible in valid cover) or count == 1 (risky) or just general coverage?
            # Actually, we just want to maximize the total weight of covered triples.
            # Let's prioritize covering triples with count == 1 (to make them non-critical) and count > 1.
            
            best_gain = -1
            best_new_block = None
            
            # If we have few critical points, we have many choices for the rest.
            # C(24 - |critical|, k - |critical|)
            # If |critical| is small, this is large.
            # Let's just sample a few or use a greedy fill.
            
            # Greedy fill
            current_block = list(new_block_set)
            rng.shuffle(current_block)
            
            # Calculate current gain (coverage of new triples)
            # We want to add points that cover triples with high "need".
            # Since we are in a valid state, all triples have count >= 1.
            # We want to increase counts of triples that are currently 1, to make them 2, so they become non-critical.
            
            # Let's just try to maximize the number of triples covered by the new block that are currently count 1.
            
            # Simplified: Just pick random remaining points and check validity + potential for future removal.
            # A better heuristic: Pick points that form many triples with existing points in the block.
            
            # For speed, let's just try a few random completions
            found_better = False
            for _ in range(5): # Try 5 random completions
                if len(new_block_set) < k:
                    to_add = k - len(new_block_set)
                    candidates = [p for p in range(v) if p not in new_block_set]
                    rng.shuffle(candidates)
                    new_block_set = set(critical_points)
                    new_block_set.update(candidates[:to_add])
                
                new_block = tuple(sorted(new_block_set))
                
                # Check validity: New block must cover all critical triples.
                # By construction, it contains critical_points, so it covers all critical triples.
                
                # Calculate "score" for this new block:
                # Number of triples in new_block that have count == 1 (in the current state, excluding the old block's contribution? No, counts include old block).
                # If we replace old_block with new_block:
                # For triple in old_block \ new_block: count decreases by 1. Must not go to 0.
                # For triple in new_block \ old_block: count increases by 1.
                # For triple in intersection: count unchanged.
                
                # We already ensured critical triples (count 1) are in new_block.
                # So any triple in old_block \ new_block has count >= 2 (since it's not critical).
                # So validity is guaranteed.
                
                # To decide if this is a "good" swap, we can look at how many count-1 triples we "fix" (move from count 1 to 2).
                # Actually, the critical triples are the ones with count 1. They are in new_block.
                # Their count will remain 1?
                # Wait.
                # Old state: triple T has count 1 (covered by old_block).
                # New state: T is covered by new_block.
                # Is T covered by any other block? No, count was 1.
                # So in new state, T is covered by new_block. Count is still 1.
                # This doesn't help with redundancy removal directly.
                
                # However, if we pick a new block that covers *more* triples that are currently count 1,
                # we might create opportunities? No, count 1 means it's critical.
                
                # The real benefit of swapping is if the new block allows *other* blocks to be removed.
                # This is hard to predict locally.
                
                # Let's just accept any valid swap with some probability, or accept if it doesn't increase the number of critical triples?
                # Let's count the number of critical triples in the new configuration.
                
                new_critical_count = 0
                for triple in itertools.combinations(new_block, t):
                    idx = triple_to_idx[triple]
                    # Current count
                    c = counts[idx]
                    # If this triple is in old_block, its count will be c - 1 + 1 = c (if in new) or c - 1 (if not).
                    # This is getting complicated to simulate without applying.
                    
                # Let's just apply the swap if it's valid.
                # Since validity is guaranteed by containing critical points, we can just do it.
                # To avoid cycling, we can add a small bias towards blocks that cover more "low count" triples.
                
                found_better = True
                break # Just take the first random valid completion for speed? 
                # Or try to optimize?
                # Let's try to maximize the number of triples with count == 1 that are covered by the new block.
                # This might help in the long run.
                
                # Actually, let's just do a simple random swap.
                pass
            
            if found_better:
                # Apply swap
                # Remove old block coverage
                for triple in itertools.combinations(old_block, t):
                    idx = triple_to_idx[triple]
                    counts[idx] -= 1
                # Add new block coverage
                for triple in itertools.combinations(new_block, t):
                    idx = triple_to_idx[triple]
                    counts[idx] += 1
                
                blocks[i] = new_block
                
                # Check if we improved
                # (No direct improvement in block count, but might enable future removals)
                
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min

    # Final Polish
    blocks, counts = remove_redundant(blocks, counts)
    
    if len(blocks) < best_len:
        best_blocks = [list(b) for b in blocks]

    return best_blocks
