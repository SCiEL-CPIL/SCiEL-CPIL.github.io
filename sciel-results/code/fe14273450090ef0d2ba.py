"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a multi-strategy greedy approach
    with randomized restarts and local improvement.
    
    Strategy:
    1. Precompute all t-subsets and their bitmasks for fast coverage checking.
    2. Use a greedy algorithm with multiple variants:
       - Standard greedy: pick the block that covers the most uncovered t-subsets
       - Weighted greedy: use weights to break ties and explore different regions
       - Randomized restarts: try different random orderings
    3. Local improvement: try to swap points in blocks to reduce redundancy
    4. Return the best solution found within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 145  # 145s to leave 5s safety margin
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute for each possible block (k-subset), the list of t-subset indices it covers
    # This is expensive, so we'll compute on-the-fly but cache common patterns
    
    def get_tsubsets_in_block(block):
        """Get indices of t-subsets covered by a block."""
        return [tsubset_to_idx[ts] for ts in itertools.combinations(block, t)]
    
    def greedy_cover(uncovered_set, time_limit=None):
        """
        Greedy covering: repeatedly pick the block that covers the most uncovered t-subsets.
        Returns list of blocks.
        """
        blocks = []
        uncovered = set(uncovered_set)
        
        while uncovered and (time_limit is None or time.time() < time_limit):
            # Find the block that covers the most uncovered t-subsets
            best_block = None
            best_count = 0
            best_covered = set()
            
            # For efficiency, we try candidate blocks based on uncovered t-subsets
            # Pick a random uncovered t-subset to anchor our search
            sample_ts = random.choice(list(uncovered))
            anchor_points = set(sample_ts)
            
            # Try blocks that contain the anchor points
            # First, build blocks containing all anchor points
            remaining_points = [p for p in range(v) if p not in anchor_points]
            need = k - t
            
            if need <= 0:
                candidate_blocks = [tuple(sorted(anchor_points))]
            elif need == 1:
                candidate_blocks = [tuple(sorted(anchor_points | {p})) for p in remaining_points]
            else:
                # Try a sample of candidate blocks
                # Pick points that appear in many uncovered t-subsets
                point_scores = defaultdict(int)
                for ts in list(uncovered)[:200]:  # Sample for speed
                    for p in ts:
                        point_scores[p] += 1
                
                # Sort points by score
                scored_points = sorted(remaining_points, key=lambda p: -point_scores.get(p, 0))
                
                # Try blocks with top-scoring points
                top_points = scored_points[:min(need * 3, len(scored_points))]
                if len(top_points) >= need:
                    for combo in itertools.combinations(top_points, need):
                        candidate_blocks.append(tuple(sorted(anchor_points | set(combo))))
                else:
                    # Fallback: use all remaining points
                    for combo in itertools.combinations(remaining_points, need):
                        candidate_blocks.append(tuple(sorted(anchor_points | set(combo))))
                
                # Limit candidates for speed
                if len(candidate_blocks) > 500:
                    random.shuffle(candidate_blocks)
                    candidate_blocks = candidate_blocks[:500]
            
            # Evaluate candidates
            for block in candidate_blocks:
                if len(block) != k:
                    continue
                covered = set(get_tsubsets_in_block(block))
                new_covered = covered & uncovered
                count = len(new_covered)
                if count > best_count:
                    best_count = count
                    best_block = block
                    best_covered = new_covered
            
            if best_block is None or best_count == 0:
                # Fallback: just add a random block
                best_block = tuple(sorted(random.sample(range(v), k)))
                best_covered = set(get_tsubsets_in_block(best_block)) & uncovered
            
            blocks.append(list(best_block))
            uncovered -= best_covered
        
        return blocks
    
    def local_improve(blocks, time_limit=None):
        """
        Try to improve the covering by swapping points in blocks.
        """
        if time_limit is None or time.time() >= time_limit:
            return blocks
        
        # Build coverage map: for each t-subset, which blocks cover it
        tsubset_covered_by = defaultdict(list)
        for i, block in enumerate(blocks):
            for ts in itertools.combinations(block, t):
                tsubset_covered_by[ts].append(i)
        
        # Try swaps
        improved = True
        while improved and (time_limit is None or time.time() < time_limit):
            improved = False
            for i, block in enumerate(blocks):
                if time_limit is not None and time.time() >= time_limit:
                    break
                for pos in range(k):
                    if time_limit is not None and time.time() >= time_limit:
                        break
                    orig_point = block[pos]
                    # Try replacing with each other point
                    for new_point in range(v):
                        if new_point == orig_point or new_point in block:
                            continue
                        new_block = block[:]
                        new_block[pos] = new_point
                        new_block.sort()
                        
                        # Check if this swap is beneficial
                        # Count how many t-subsets are only covered by this block
                        old_covered = set(get_tsubsets_in_block(block))
                        new_covered = set(get_tsubsets_in_block(new_block))
                        
                        # T-subsets that would become uncovered
                        lost = old_covered - new_covered
                        # T-subsets that would be newly covered
                        gained = new_covered - old_covered
                        
                        # Check if any lost t-subset is not covered by other blocks
                        would_lose = 0
                        for ts in lost:
                            if len(tsubset_covered_by[ts]) == 1:
                                would_lose += 1
                        
                        if would_lose == 0 and len(gained) > 0:
                            # This swap is beneficial
                            # Update coverage map
                            for ts in lost:
                                tsubset_covered_by[ts].remove(i)
                            for ts in gained:
                                tsubset_covered_by[ts].append(i)
                            blocks[i] = new_block
                            improved = True
                            break
        
        return blocks
    
    # Try multiple strategies and keep the best
    best_blocks = None
    best_count = float('inf')
    
    # Strategy 1: Standard greedy
    if time.time() < deadline - 30:
        blocks = greedy_cover(set(range(num_tsubsets)), deadline - 10)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
    
    # Strategy 2: Multiple randomized restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() >= deadline - 20:
            break
        
        # Shuffle the order of uncovered t-subsets
        shuffled = list(range(num_tsubsets))
        random.shuffle(shuffled)
        
        blocks = greedy_cover(set(shuffled), deadline - 10)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        
        # Try local improvement
        if time.time() < deadline - 5:
            improved_blocks = local_improve(blocks[:], deadline - 2)
            if len(improved_blocks) < best_count:
                best_blocks = improved_blocks
                best_count = len(improved_blocks)
    
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = greedy_cover(set(range(num_tsubsets)), deadline)
    
    return best_blocks
# EVOLVE-BLOCK-END
