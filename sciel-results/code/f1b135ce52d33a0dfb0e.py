"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def construct(v, k, t):
    deadline = time.time() + 135.0  # safety margin before 150s kill
    rng = random.Random(42)
    
    best_blocks = None
    best_count = float('inf')
    
    def greedy_cover():
        """Randomized greedy set cover with reverse indexing."""
        # Map each t-subset to a unique index
        all_tsubs = list(itertools.combinations(range(v), t))
        tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
        n_tsubs = len(all_tsubs)
        
        # For each point, precompute which t-subsets it participates in
        point_tsubs = [[] for _ in range(v)]
        for ts in all_tsubs:
            for p in ts:
                point_tsubs[p].append(tsub_to_idx[ts])
        
        # Track coverage count for each t-subset
        coverage = [0] * n_tsubs
        blocks = []
        
        while True:
            if time.time() > deadline:
                return None  # timeout
            
            # Find an uncovered t-subset
            uncovered_idx = None
            for i in range(n_tsubs):
                if coverage[i] == 0:
                    uncovered_idx = i
                    break
            if uncovered_idx is None:
                break
            
            # Build a block that covers this t-subset and maximizes new coverage
            seed_tsub = all_tsubs[uncovered_idx]
            block_set = set(seed_tsub)
            
            # Greedily add points
            for _ in range(k - t):
                if time.time() > deadline:
                    return None
                
                best_point = None
                best_gain = -1
                candidates = [p for p in range(v) if p not in block_set]
                
                # Shuffle for randomization
                rng.shuffle(candidates)
                
                for p in candidates:
                    # Calculate gain: how many new t-subsets this point covers
                    # given the current block
                    gain = 0
                    for tsub_idx in point_tsubs[p]:
                        if coverage[tsub_idx] == 0:
                            # Check if this t-subset would be covered by adding p
                            # i.e., all other elements are already in block_set
                            ts = all_tsubs[tsub_idx]
                            others = [x for x in ts if x != p]
                            if all(o in block_set for o in others):
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                    if gain == n_tsubs:  # max possible
                        break
                
                if best_point is not None:
                    block_set.add(best_point)
            
            block = sorted(block_set)
            blocks.append(block)
            
            # Update coverage
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                coverage[idx] += 1
        
        return blocks

    def local_search_optimize(blocks):
        """Aggressive local search: remove redundant blocks and replace blocks to shift unique burden."""
        if not blocks:
            return blocks
        
        all_tsubs = list(itertools.combinations(range(v), t))
        tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
        n_tsubs = len(all_tsubs)
        
        def calculate_block_tsubs(block):
            return [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        
        improved = True
        while improved:
            improved = False
            
            # 1. Remove redundant blocks
            coverage_count = [0] * n_tsubs
            block_tsubs = []
            
            for block in blocks:
                tsubs_in_block = calculate_block_tsubs(block)
                block_tsubs.append(tsubs_in_block)
                for idx in tsubs_in_block:
                    coverage_count[idx] += 1
            
            keep = [True] * len(blocks)
            for i in range(len(blocks)):
                if not keep[i]:
                    continue
                can_remove = True
                for tsub_idx in block_tsubs[i]:
                    if coverage_count[tsub_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    keep[i] = False
                    improved = True
                    for tsub_idx in block_tsubs[i]:
                        coverage_count[tsub_idx] -= 1
            
            if improved:
                blocks = [blocks[i] for i in range(len(blocks)) if keep[i]]
                # Recalculate block_tsubs and coverage_count for next iteration if we removed blocks
                # Actually, the loop restarts, so we can just continue to the next phase or restart.
                # Restarting is safer for consistency.
                continue
            
            # 2. Try to replace blocks
            # Recalculate coverage and block_tsubs for current state
            coverage_count = [0] * n_tsubs
            block_tsubs = []
            for block in blocks:
                tsubs_in_block = calculate_block_tsubs(block)
                block_tsubs.append(tsubs_in_block)
                for idx in tsubs_in_block:
                    coverage_count[idx] += 1
            
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks
                
                # Find t-subsets uniquely covered by block i
                unique_tsubs = []
                for tsub_idx in block_tsubs[i]:
                    if coverage_count[tsub_idx] == 1:
                        unique_tsubs.append(tsub_idx)
                
                if not unique_tsubs:
                    continue # Should have been removed in step 1, but maybe not if logic changed
                
                # Determine mandatory points for a replacement block
                mandatory_points = set()
                for tsub_idx in unique_tsubs:
                    mandatory_points.update(all_tsubs[tsub_idx])
                
                if len(mandatory_points) > k:
                    continue # Cannot cover all unique tsubs with one block
                
                # Try to find a replacement block B' containing mandatory_points
                # We want B' to be "good". 
                # Heuristic: Fill remaining k - len(mandatory_points) slots with points 
                # that maximize coverage of t-subsets currently covered by exactly 2 blocks (to increase redundancy).
                
                remaining_slots = k - len(mandatory_points)
                current_block_set = set(blocks[i])
                
                best_new_block = None
                best_score = -1
                
                # Try a few random completions
                for _ in range(20):
                    candidates = [p for p in range(v) if p not in mandatory_points]
                    rng.shuffle(candidates)
                    extra_points = candidates[:remaining_slots]
                    new_block_set = mandatory_points.union(extra_points)
                    
                    # Check if this block is distinct from existing blocks? 
                    # The problem says blocks must be distinct.
                    # We can check against the current list.
                    # But we are replacing block i, so it's fine if it matches another block? 
                    # No, "blocks must be distinct".
                    # If new_block_set is in blocks (excluding i), it's invalid.
                    # However, if it's identical to another block, we could just remove one of them?
                    # Let's just skip if it's a duplicate of another block.
                    dup = False
                    for j in range(len(blocks)):
                        if j == i: continue
                        if set(blocks[j]) == new_block_set:
                            dup = True
                            break
                    if dup:
                        continue
                        
                    new_block = sorted(new_block_set)
                    
                    # Score: How many t-subsets does this new block cover that are 
                    # currently covered by exactly 2 blocks? (Potential for making them 3, 
                    # or if we remove the old block, they become 1? No, old block is gone.
                    # If we replace B_i with B', tsubs in B_i \ B' lose 1 coverage.
                    # tsubs in B' \ B_i gain 1 coverage.
                    # We want to avoid losing coverage of tsubs that are at 1.
                    # We already ensured B' covers all tsubs at 1 in B_i.
                    # So no tsub at 1 in B_i is lost.
                    # What about tsubs in B_i that are at 2? If not in B', they drop to 1.
                    # What about tsubs in B' that are at 1? They go to 2.
                    # What about tsubs in B' that are at 2? They go to 3.
                    
                    # A good replacement minimizes the number of tsubs that drop from 2 to 1.
                    # And maximizes tsubs that go from 1 to 2? (Not critical if they were 1).
                    
                    new_tsubs = set(calculate_block_tsubs(new_block))
                    old_tsubs = set(block_tsubs[i])
                    
                    lost_tsubs = old_tsubs - new_tsubs
                    gained_tsubs = new_tsubs - old_tsubs
                    
                    # Check safety: No tsub in lost_tsubs should have coverage 1.
                    # We already checked unique_tsubs (coverage 1) are in new_tsubs.
                    # So any tsub in lost_tsubs has coverage >= 2.
                    
                    # Score: Minimize count of lost_tsubs with coverage 2.
                    # Maximize count of gained_tsubs with coverage 1? (Makes them 2).
                    
                    risk = 0
                    for ts_idx in lost_tsubs:
                        if coverage_count[ts_idx] == 2:
                            risk += 1
                    
                    benefit = 0
                    for ts_idx in gained_tsubs:
                        if coverage_count[ts_idx] == 1:
                            benefit += 1
                        elif coverage_count[ts_idx] == 2:
                            benefit += 0.5 # Slight benefit
                    
                    score = benefit - 2 * risk
                    if score > best_score:
                        best_score = score
                        best_new_block = new_block
                
                if best_new_block is not None:
                    # Perform replacement
                    blocks[i] = best_new_block
                    improved = True
                    break # Restart the while loop to recalculate coverage
            
        return blocks

    # Run greedy with randomization and local search
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        blocks = greedy_cover()
        if blocks is None:
            continue
        
        # Local search to remove redundant blocks and optimize
        blocks = local_search_optimize(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks.copy()
    
    if best_blocks is not None:
        return best_blocks
    
    # Fallback
    blocks = greedy_cover()
    if blocks is not None:
        return blocks
    
    return []
# EVOLVE-BLOCK-END
