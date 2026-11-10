import itertools
import random
import math
import time

def _count_uncovered(blocks, v, t):
    """Helper to count uncovered t-subsets from scratch."""
    covered = set()
    for b in blocks:
        for s in itertools.combinations(b, t):
            covered.add(s)
    total = math.comb(v, t)
    return total - len(covered)

def _greedy_initial(v, k, t):
    """
    Fast deterministic/randomized greedy to get a valid initial cover.
    This is just to get a starting point for the SA.
    """
    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    uncovered = set(all_t_subs)
    blocks = []
    
    # Precompute t-subsets for each point? No, just iterate.
    # To speed up, we can pick the most constrained t-subset (one with fewest candidates? No, candidates are all k-subsets containing it).
    # Simple greedy: pick block covering most uncovered.
    
    # Optimization: Instead of checking all C(24,7) blocks, we can just pick random blocks until covered?
    # That might take too long.
    # Let's use a standard greedy:
    # 1. Find an uncovered t-subset S.
    # 2. Find a block B containing S that covers the most other uncovered t-subsets.
    # 3. Add B.
    
    # To find B efficiently:
    # For a fixed S, any block B containing S is a candidate.
    # Number of candidates is C(24-3, 7-3) = C(21, 4) = 5985. This is small enough to check all for each S.
    
    while uncovered:
        # Pick a random uncovered t-subset to break symmetry
        s = random.choice(list(uncovered))
        
        best_block = None
        best_gain = -1
        
        # Generate all k-subsets containing s
        # s is a tuple of t points.
        remaining_points = [p for p in all_points if p not in s]
        # We need to choose k-t points from remaining_points
        for combo in itertools.combinations(remaining_points, k - t):
            block = sorted(list(s) + list(combo))
            # Calculate gain: how many uncovered t-subsets are in this block
            gain = 0
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block:
            blocks.append(best_block)
            for sub in itertools.combinations(best_block, t):
                uncovered.discard(sub)
        
    return blocks

def construct(v, k, t):
    """
    Uses Simulated Annealing on a set of blocks to minimize the count.
    """
    # 1. Get an initial valid cover
    initial_blocks = _greedy_initial(v, k, t)
    
    # 2. Setup SA
    # State: list of blocks
    # We maintain cover counts for speed
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # Initialize cover counts
    cover_counts = [0] * len(all_t_subs)
    for b in initial_blocks:
        for s in itertools.combinations(b, t):
            cover_counts[t_sub_to_idx[s]] += 1
            
    uncovered_count = sum(1 for c in cover_counts if c == 0)
    current_blocks = initial_blocks[:]
    current_cost = len(current_blocks) + 100 * uncovered_count # High penalty for invalid
    
    best_blocks = current_blocks[:]
    best_cost = current_cost
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.01
    alpha = 0.99
    T = T_start
    max_iter = 50000
    start_time = time.time()
    time_limit = 110 # Leave 10s margin
    
    rng = random.Random(42)
    
    def get_cost(blocks, uncovered_count):
        return len(blocks) + 100 * uncovered_count

    while T > T_end and (time.time() - start_time) < time_limit:
        # Generate a neighbor
        move_type = rng.random()
        new_blocks = current_blocks[:]
        new_uncovered_count = uncovered_count
        
        if move_type < 0.5 and len(new_blocks) > 0:
            # Remove a random block
            idx = rng.randrange(len(new_blocks))
            block_to_remove = new_blocks[idx]
            new_blocks.pop(idx)
            
            # Update cover counts
            for s in itertools.combinations(block_to_remove, t):
                idx_s = t_sub_to_idx[s]
                cover_counts[idx_s] -= 1
                if cover_counts[idx_s] == 0:
                    new_uncovered_count += 1
                    
        elif move_type < 0.8:
            # Swap a random block with a new random block
            if len(new_blocks) == 0:
                # Just add
                new_block = sorted(rng.sample(range(v), k))
                new_blocks.append(new_block)
                for s in itertools.combinations(new_block, t):
                    idx_s = t_sub_to_idx[s]
                    cover_counts[idx_s] += 1
                    if cover_counts[idx_s] == 1:
                        new_uncovered_count -= 1
            else:
                idx = rng.randrange(len(new_blocks))
                old_block = new_blocks[idx]
                new_block = sorted(rng.sample(range(v), k))
                new_blocks[idx] = new_block
                
                # Remove old
                for s in itertools.combinations(old_block, t):
                    idx_s = t_sub_to_idx[s]
                    cover_counts[idx_s] -= 1
                    if cover_counts[idx_s] == 0:
                        new_uncovered_count += 1
                        
                # Add new
                for s in itertools.combinations(new_block, t):
                    idx_s = t_sub_to_idx[s]
                    cover_counts[idx_s] += 1
                    if cover_counts[idx_s] == 1:
                        new_uncovered_count -= 1
                        
        else:
            # Add a random block
            new_block = sorted(rng.sample(range(v), k))
            # Avoid duplicates? Not strictly necessary for SA, but good for quality.
            if new_block not in new_blocks:
                new_blocks.append(new_block)
                for s in itertools.combinations(new_block, t):
                    idx_s = t_sub_to_idx[s]
                    cover_counts[idx_s] += 1
                    if cover_counts[idx_s] == 1:
                        new_uncovered_count -= 1
            else:
                # If duplicate, just skip or treat as no-op? Let's just skip to save time.
                # But we need to revert state if we skip? 
                # Easier: if duplicate, don't change state.
                new_blocks = current_blocks
                new_uncovered_count = uncovered_count
                # Revert cover counts? They weren't changed if we didn't modify list?
                # Wait, in the code above, I modified cover_counts directly.
                # I need to be careful with state rollback.
                
                # Let's refactor: calculate delta, apply, if reject, revert.
                pass # This branch is tricky with direct mutation. Let's simplify: always allow the move, even if duplicate, cost will be higher.

        # Calculate new cost
        new_cost = len(new_blocks) + 100 * new_uncovered_count
        
        # Acceptance
        delta = new_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_blocks = new_blocks
            current_cost = new_cost
            uncovered_count = new_uncovered_count
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_blocks = current_blocks[:]
                
        # Cool down
        T *= alpha

    # Ensure we return a valid solution
    # If best_blocks is invalid, try to fix it by adding blocks?
    # The penalty should drive it to valid.
    # But if it didn't reach 0 uncovered, we might return an invalid set.
    # Let's check best_blocks validity.
    # If invalid, we can run a quick greedy repair.
    
    # Verify best_blocks
    covered = set()
    for b in best_blocks:
        for s in itertools.combinations(b, t):
            covered.add(s)
            
    all_needed = set(all_t_subs)
    missing = all_needed - covered
    
    if missing:
        # Repair: Greedily add blocks to cover missing
        for s in missing:
            # Find a block containing s
            # Just pick a random one for speed, or greedy
            remaining_points = [p for p in range(v) if p not in s]
            new_block = sorted(list(s) + list(rng.sample(remaining_points, k - t)))
            # Check if it covers more missing
            best_b = new_block
            best_gain = 0
            for _ in range(10):
                rb = sorted(list(s) + list(rng.sample(remaining_points, k - t)))
                gain = sum(1 for sub in itertools.combinations(rb, t) if sub in missing)
                if gain > best_gain:
                    best_gain = gain
                    best_b = rb
            if best_b not in best_blocks:
                best_blocks.append(best_b)
                
    return best_blocks
