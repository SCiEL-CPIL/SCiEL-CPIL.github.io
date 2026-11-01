"""Simulated annealing approach for covering design C(24,7,3)."""
import itertools
import random
import time
import math

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using simulated annealing.
    
    Strategy:
    1. Start with a small set of random/greedy blocks.
    2. Use simulated annealing with point-replacement moves to minimize uncovered t-subsets.
    3. Periodically try to remove redundant blocks.
    4. Use multiple restarts with different initial configurations.
    
    Returns a list of blocks (each a list of k distinct integers in 0..v-1).
    """
    random.seed(42)
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each t-subset index, which points it contains (as a set for fast membership)
    t_subset_points = [set(ts) for ts in all_t_subsets]
    
    # For each point x, precompute the list of t-subset indices that contain x
    point_to_tsubsets = [[] for _ in range(v)]
    for i, ts in enumerate(all_t_subsets):
        for x in ts:
            point_to_tsubsets[x].append(i)
    
    def count_coverages(blocks):
        """Return an array where count[i] = number of blocks covering all_t_subsets[i]."""
        counts = [0] * n_t_subsets
        for block in blocks:
            block_set = set(block)
            for ts in itertools.combinations(block, t):
                idx = t_subset_to_idx[ts]
                counts[idx] += 1
        return counts
    
    def uncovered_count(counts):
        """Return the number of uncovered t-subsets."""
        return sum(1 for c in counts if c == 0)
    
    def greedy_covering(max_blocks=None):
        """Generate initial covering using greedy algorithm."""
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            if max_blocks is not None and len(blocks) >= max_blocks:
                break
            # Pick the first uncovered t-subset
            target = min(uncovered)
            block = list(target)
            while len(block) < k:
                # Find the point that covers the most uncovered t-subsets
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    # Pick a random point
                    available = [x for x in range(v) if x not in block]
                    best_x = random.choice(available)
                block.append(best_x)
            block.sort()
            blocks.append(block)
            # Remove all t-subsets covered by this block
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        return blocks
    
    def random_initial(n_blocks):
        """Generate random initial blocks."""
        blocks = []
        for _ in range(n_blocks):
            block = random.sample(range(v), k)
            block.sort()
            blocks.append(block)
        return blocks
    
    def simulated_annealing(initial_blocks, time_budget):
        """Run simulated annealing starting from initial_blocks."""
        start_time = time.time()
        
        blocks = [b[:] for b in initial_blocks]
        counts = count_coverages(blocks)
        current_uncovered = uncovered_count(counts)
        
        best_blocks = [b[:] for b in blocks]
        best_uncovered = current_uncovered
        
        n_blocks = len(blocks)
        
        # Temperature schedule
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        while T > T_min:
            if time.time() - start_time > time_budget:
                break
            
            # Randomly choose a move type
            move_type = random.random()
            
            if move_type < 0.7 and n_blocks >= 1:
                # Move 1: Replace a point in a random block
                bi = random.randint(0, n_blocks - 1)
                block = blocks[bi]
                pi = random.randint(0, k - 1)
                old_point = block[pi]
                
                # Choose a new point not already in the block
                available = [x for x in range(v) if x != old_point and x not in block]
                if not available:
                    continue
                new_point = random.choice(available)
                
                # Calculate delta: how many t-subsets change coverage status
                # t-subsets that contain old_point from this block's t-subsets
                # but not new_point will lose one coverage
                # t-subsets that contain new_point but not old_point will gain one coverage
                
                delta = 0
                
                # For each t-subset of the old block that contains old_point
                for ts in itertools.combinations(block, t):
                    if old_point in ts:
                        idx = t_subset_to_idx[ts]
                        # Check if new_point is in this t-subset
                        if new_point not in t_subset_points[idx]:
                            # This t-subset loses one coverage
                            if counts[idx] == 1:
                                delta += 1  # Becomes uncovered
                
                # For each t-subset of the new block that contains new_point
                new_block_temp = block[:]
                new_block_temp[pi] = new_point
                for ts in itertools.combinations(new_block_temp, t):
                    if new_point in ts:
                        idx = t_subset_to_idx[ts]
                        # Check if old_point is in this t-subset
                        if old_point not in t_subset_points[idx]:
                            # This t-subset gains one coverage
                            if counts[idx] == 0:
                                delta -= 1  # Becomes covered
                
                # Accept or reject
                accept = False
                if delta <= 0:
                    accept = True
                else:
                    if random.random() < math.exp(-delta / T):
                        accept = True
                
                if accept:
                    # Update counts
                    for ts in itertools.combinations(block, t):
                        if old_point in ts:
                            idx = t_subset_to_idx[ts]
                            if new_point not in t_subset_points[idx]:
                                counts[idx] -= 1
                    
                    new_block = block[:]
                    new_block[pi] = new_point
                    new_block.sort()
                    
                    for ts in itertools.combinations(new_block, t):
                        if new_point in ts:
                            idx = t_subset_to_idx[ts]
                            if old_point not in t_subset_points[idx]:
                                counts[idx] += 1
                    
                    blocks[bi] = new_block
                    current_uncovered += delta
                    
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_blocks = [b[:] for b in blocks]
            
            elif move_type < 0.9 and n_blocks > 1:
                # Move 2: Try to remove a redundant block
                bi = random.randint(0, n_blocks - 1)
                block = blocks[bi]
                
                # Check if all t-subsets in this block are covered by other blocks
                redundant = True
                for ts in itertools.combinations(block, t):
                    idx = t_subset_to_idx[ts]
                    if counts[idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove this block
                    for ts in itertools.combinations(block, t):
                        idx = t_subset_to_idx[ts]
                        counts[idx] -= 1
                    
                    blocks.pop(bi)
                    n_blocks -= 1
                    current_uncovered = uncovered_count(counts)
                    
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_blocks = [b[:] for b in blocks]
            
            else:
                # Move 3: Swap a point between two random blocks
                if n_blocks >= 2:
                    bi1 = random.randint(0, n_blocks - 1)
                    bi2 = random.randint(0, n_blocks - 1)
                    if bi1 == bi2:
                        continue
                    
                    block1 = blocks[bi1]
                    block2 = blocks[bi2]
                    
                    pi1 = random.randint(0, k - 1)
                    pi2 = random.randint(0, k - 1)
                    
                    p1 = block1[pi1]
                    p2 = block2[pi2]
                    
                    # Ensure p1 is not in block2 and p2 is not in block1
                    if p1 in block2 or p2 in block1:
                        continue
                    
                    new_block1 = block1[:]
                    new_block1[pi1] = p2
                    new_block1.sort()
                    
                    new_block2 = block2[:]
                    new_block2[pi2] = p1
                    new_block2.sort()
                    
                    delta = 0
                    
                    # Block1: remove p1, add p2
                    for ts in itertools.combinations(block1, t):
                        if p1 in ts:
                            idx = t_subset_to_idx[ts]
                            if p2 not in t_subset_points[idx]:
                                if counts[idx] == 1:
                                    delta += 1
                    
                    for ts in itertools.combinations(new_block1, t):
                        if p2 in ts:
                            idx = t_subset_to_idx[ts]
                            if p1 not in t_subset_points[idx]:
                                if counts[idx] == 0:
                                    delta -= 1
                    
                    # Block2: remove p2, add p1
                    for ts in itertools.combinations(block2, t):
                        if p2 in ts:
                            idx = t_subset_to_idx[ts]
                            if p1 not in t_subset_points[idx]:
                                if counts[idx] == 1:
                                    delta += 1
                    
                    for ts in itertools.combinations(new_block2, t):
                        if p1 in ts:
                            idx = t_subset_to_idx[ts]
                            if p2 not in t_subset_points[idx]:
                                if counts[idx] == 0:
                                    delta -= 1
                    
                    accept = False
                    if delta <= 0:
                        accept = True
                    else:
                        if random.random() < math.exp(-delta / T):
                            accept = True
                    
                    if accept:
                        # Update counts for block1
                        for ts in itertools.combinations(block1, t):
                            if p1 in ts:
                                idx = t_subset_to_idx[ts]
                                if p2 not in t_subset_points[idx]:
                                    counts[idx] -= 1
                        
                        for ts in itertools.combinations(new_block1, t):
                            if p2 in ts:
                                idx = t_subset_to_idx[ts]
                                if p1 not in t_subset_points[idx]:
                                    counts[idx] += 1
                        
                        # Update counts for block2
                        for ts in itertools.combinations(block2, t):
                            if p2 in ts:
                                idx = t_subset_to_idx[ts]
                                if p1 not in t_subset_points[idx]:
                                    counts[idx] -= 1
                        
                        for ts in itertools.combinations(new_block2, t):
                            if p1 in ts:
                                idx = t_subset_to_idx[ts]
                                if p2 not in t_subset_points[idx]:
                                    counts[idx] += 1
                        
                        blocks[bi1] = new_block1
                        blocks[bi2] = new_block2
                        current_uncovered += delta
                        
                        if current_uncovered < best_uncovered:
                            best_uncovered = current_uncovered
                            best_blocks = [b[:] for b in blocks]
            
            # Cool down
            T *= cooling_rate
        
        return best_blocks, best_uncovered
    
    def final_cleanup(blocks):
        """Remove redundant blocks from a valid covering."""
        if not blocks:
            return blocks
        
        changed = True
        while changed:
            changed = False
            counts = count_coverages(blocks)
            for i in range(len(blocks) - 1, -1, -1):
                block = blocks[i]
                redundant = True
                for ts in itertools.combinations(block, t):
                    idx = t_subset_to_idx[ts]
                    if counts[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    changed = True
                    break
        
        # Ensure all blocks are sorted
        for i in range(len(blocks)):
            blocks[i].sort()
        
        return blocks
    
    # Overall time budget
    total_start = time.time()
    max_total_time = 115.0
    
    # Try multiple strategies
    best_overall_blocks = None
    best_overall_uncovered = float('inf')
    
    # Strategy 1: Start with greedy covering, then SA
    greedy_blocks = greedy_covering()
    remaining_time = max_total_time - (time.time() - total_start)
    if remaining_time > 5:
        sa_blocks, sa_uncovered = simulated_annealing(greedy_blocks, remaining_time - 2)
        if sa_uncovered < best_overall_uncovered:
            best_overall_uncovered = sa_uncovered
            best_overall_blocks = sa_blocks
    
    # Strategy 2: Start with fewer random blocks, let SA fill in
    for n_init in [50, 60, 70]:
        remaining_time = max_total_time - (time.time() - total_start)
        if remaining_time < 10:
            break
        
        random_blocks = random_initial(n_init)
        sa_blocks, sa_uncovered = simulated_annealing(random_blocks, remaining_time - 2)
        if sa_uncovered < best_overall_uncovered:
            best_overall_uncovered = sa_uncovered
            best_overall_blocks = sa_blocks
    
    # Strategy 3: Start with greedy but fewer blocks
    for max_b in [60, 65, 70, 75]:
        remaining_time = max_total_time - (time.time() - total_start)
        if remaining_time < 10:
            break
        
        greedy_partial = greedy_covering(max_blocks=max_b)
        sa_blocks, sa_uncovered = simulated_annealing(greedy_partial, remaining_time - 2)
        if sa_uncovered < best_overall_uncovered:
            best_overall_uncovered = sa_uncovered
            best_overall_blocks = sa_blocks
    
    # Use the best solution found
    blocks = best_overall_blocks if best_overall_blocks is not None else greedy_covering()
    
    # Final cleanup: remove any redundant blocks
    blocks = final_cleanup(blocks)
    
    # Verify coverage
    counts = count_coverages(blocks)
    if uncovered_count(counts) > 0:
        # Fallback to greedy if we didn't find a valid covering
        blocks = greedy_covering()
    
    return blocks
