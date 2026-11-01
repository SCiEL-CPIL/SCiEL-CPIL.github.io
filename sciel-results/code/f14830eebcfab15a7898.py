import itertools
import random
import time

def construct(v, k, t):
    all_points = list(range(v))
    all_tsubsets = list(itertools.combinations(all_points, t))
    n_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(all_tsubsets)}
    
    blocks = []
    cover_count = [0] * n_tsubsets
    block_subs = []  # list of lists of subset indices for each block
    
    def get_block_subsets(block):
        """Return list of indices of t-subsets contained in block"""
        subs = []
        block_set = set(block)
        for combo in itertools.combinations(block, t):
            idx = subset_to_idx.get(tuple(combo))
            if idx is not None:
                subs.append(idx)
        return subs
    
    def add_block(block):
        block = tuple(sorted(block))
        # Check if block already exists
        for b in blocks:
            if b == block:
                return False
        subs = get_block_subsets(block)
        blocks.append(block)
        block_subs.append(subs)
        for idx in subs:
            cover_count[idx] += 1
        return True
    
    def remove_block(i):
        subs = block_subs[i]
        for idx in subs:
            cover_count[idx] -= 1
        blocks.pop(i)
        block_subs.pop(i)
    
    def is_redundant(i):
        for idx in block_subs[i]:
            if cover_count[idx] == 1:
                return False
        return True
    
    def prune():
        # Remove redundant blocks
        i = 0
        while i < len(blocks):
            if is_redundant(i):
                remove_block(i)
            else:
                i += 1
    
    def uncovered_count():
        return sum(1 for c in cover_count if c == 0)
    
    start_time = time.time()
    time_limit = 115
    
    rng = random.Random(12345)
    
    # Phase 1: Greedy initialization
    # Sort uncovered subsets by "difficulty" (how many blocks could cover them)
    # For a t-subset S, the number of k-subsets containing it is C(v-t, k-t)
    # But we want to prioritize subsets that are "harder" to cover
    # Actually, all t-subsets have the same number of containing k-subsets: C(v-t, k-t)
    # So we just pick randomly or greedily
    
    while time.time() - start_time < time_limit * 0.15:
        if uncovered_count() == 0:
            break
        
        # Find uncovered subsets
        uncovered = [i for i in range(n_tsubsets) if cover_count[i] == 0]
        if not uncovered:
            break
        
        # Pick a random uncovered subset
        seed_idx = rng.choice(uncovered)
        seed_subset = all_tsubsets[seed_idx]
        
        # Build a block around this subset
        seed_set = set(seed_subset)
        block = list(seed_subset)
        remaining = [p for p in all_points if p not in seed_set]
        rng.shuffle(remaining)
        block.extend(remaining[:k - t])
        rng.shuffle(block)
        block.sort()
        
        add_block(block)
    
    # If still not valid, force fill
    while uncovered_count() > 0:
        uncovered = [i for i in range(n_tsubsets) if cover_count[i] == 0]
        if not uncovered:
            break
        seed_idx = rng.choice(uncovered)
        seed_subset = all_tsubsets[seed_idx]
        seed_set = set(seed_subset)
        block = list(seed_subset)
        remaining = [p for p in all_points if p not in seed_set]
        rng.shuffle(remaining)
        block.extend(remaining[:k - t])
        rng.shuffle(block)
        block.sort()
        add_block(block)
    
    # Initial prune
    prune()
    
    best_len = len(blocks)
    best_blocks = [list(b) for b in blocks]
    
    # Phase 2: Local search with multiple strategies
    # Strategy A: Remove-replace (find blocks with unique coverage, try to replace)
    # Strategy B: Random restart within LS
    # Strategy C: Pairwise exchange (try swapping one block for another)
    
    # Precompute for each block the set of points it contains
    # For faster intersection calculations
    
    def block_points_set(block):
        return set(block)
    
    # Better approach for replacement:
    # For block i, find all t-subsets uniquely covered by block i
    # These must be covered by the replacement block
    # The replacement block must contain the union of these unique subsets
    # If union size > k, block i cannot be removed (it's "essential")
    
    # For speed, we'll use a different approach:
    # Try to remove block i and see if coverage breaks
    # If it breaks, find the uncovered subsets and try to build a new block covering them
    
    def try_remove_and_repair(i):
        """Try to remove block i, then repair if needed. Returns True if successful (size decreased or same)."""
        subs = block_subs[i]
        
        # Check which subsets would become uncovered
        would_be_uncovered = []
        for idx in subs:
            if cover_count[idx] == 1:
                would_be_uncovered.append(idx)
        
        if not would_be_uncovered:
            # Block is redundant
            remove_block(i)
            return True
        
        # Union of points in uniquely covered subsets
        union_points = set()
        for idx in would_be_uncovered:
            union_points.update(all_tsubsets[idx])
        
        if len(union_points) > k:
            return False
        
        # Try to build a new block covering union_points
        # Add random extra points to make size k
        remaining_points = [p for p in all_points if p not in union_points]
        rng.shuffle(remaining_points)
        extra_needed = k - len(union_points)
        new_block_set = set(union_points)
        new_block_set.update(remaining_points[:extra_needed])
        new_block = tuple(sorted(new_block_set))
        
        # Check if new block already exists
        for b in blocks:
            if b == new_block:
                # If it exists, we can just remove block i
                remove_block(i)
                return True
        
        # Remove old block, add new block
        remove_block(i)
        add_block(new_block)
        prune()
        return True
    
    # Strategy: Iterate through blocks, try to remove and repair
    no_improvement_streak = 0
    max_no_improvement = 500
    
    while time.time() - start_time < time_limit:
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1
        
        if no_improvement_streak > max_no_improvement and len(blocks) > 0:
            # Random perturbation: remove a random block and re-add random blocks
            # But this might break validity. Let's just do a random restart of LS phase.
            no_improvement_streak = 0
            max_no_improvement = int(max_no_improvement * 1.2)
        
        if not blocks:
            break
        
        # Try removing and repairing a random block
        i = rng.randint(0, len(blocks) - 1)
        
        # Quick check: if block has no unique coverage, remove it
        has_unique = False
        for idx in block_subs[i]:
            if cover_count[idx] == 1:
                has_unique = True
                break
        
        if not has_unique:
            remove_block(i)
            prune()
            continue
        
        # Try to replace
        old_len = len(blocks)
        success = try_remove_and_repair(i)
        
        # Prune after any change
        prune()
    
    return best_blocks
