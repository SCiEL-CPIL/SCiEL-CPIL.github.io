import itertools
import random
import time
import sys

# Precomputation for v=19, k=8, t=5
# We will hardcode the precomputation logic to run once at import or first call.
# Since the task is specific to C(19,8,5), we can optimize for these parameters.
# However, the signature is construct(v, k, t), so we should be somewhat general or 
# check if v, k, t match the known case. 
# Given the 120s limit, precomputation for general v,k,t might be too slow if v is large.
# But the task is specifically C(19,8,5). 
# The prompt says "The evaluator passes this task's parameters v, k and t."
# It implies v=19, k=8, t=5.
# We will implement a general solution but with heavy optimization for the specific case.

# To be safe and fast, we will detect if v=19, k=8, t=5 and use a specialized path.
# Otherwise, fallback to a simpler greedy (though it will likely be slow/invalid for other large cases, 
# the task is fixed).

def _precompute_incidence(v, k, t):
    """
    Precomputes the mapping from k-subsets to the list of t-subsets they cover.
    Returns:
    all_blocks: list of all k-subsets
    block_to_tsubs: list of lists of t-subset indices
    tsub_to_blocks: not strictly needed if we use coverage counts, but useful for debugging.
    tsubs: list of all t-subsets
    """
    points = range(v)
    tsubs = list(itertools.combinations(points, t))
    tsub_index = {ts: i for i, ts in enumerate(tsubs)}
    
    all_blocks = list(itertools.combinations(points, k))
    block_to_tsubs = []
    
    for block in all_blocks:
        covered = []
        for tsub in itertools.combinations(block, t):
            covered.append(tsub_index[tsub])
        block_to_tsubs.append(covered)
        
    return all_blocks, block_to_tsubs, tsubs, tsub_index

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second buffer for safety
    
    # Check if this is the specific problem C(19,8,5)
    if v == 19 and k == 8 and t == 5:
        return _solve_19_8_5(deadline)
    else:
        # Fallback for other parameters: Simple Greedy
        # This is likely to be slow for large v, but correct.
        return _greedy_cover(v, k, t, deadline)

def _greedy_cover(v, k, t, deadline):
    """Standard greedy set cover."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered and time.time() < deadline:
        # Find block covering most uncovered
        best_block = None
        max_cov = -1
        # This is O(C(v,k) * C(k,t)) which is huge. 
        # For general case, this is too slow. 
        # We rely on the specific case being 19,8,5.
        # If other cases are tested, this might timeout.
        # To ensure validity, we return whatever we have if we run out of time? 
        # No, it must be valid. 
        # The task says "Valid when every t-subset ... lies in at least one block".
        # If we don't finish, we return invalid? 
        # "An invalid result ... scores 0".
        # So we must finish. 
        # For non-19,8,5 cases, we might just do a very basic greedy.
        
        # Optimization: Pick the first uncovered subset, and find a block containing it.
        # This is O(1) to pick subset, O(C(v-5, 8-5)) to find blocks? No.
        # Let's just use the naive greedy from the seed for non-specific cases, 
        # hoping they aren't tested or are small.
        
        target = next(iter(uncovered))
        # Find a block containing target
        # Generate all blocks containing target
        remaining_points = [p for p in range(v) if p not in target]
        # We need k - t more points
        for block_rest in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(list(target) + list(block_rest)))
            # Check if this block is good? 
            # Just take the first one? That's bad.
            # Let's calculate gain.
            gain = 0
            for ts in itertools.combinations(block, t):
                if ts in uncovered:
                    gain += 1
            # This inner loop is expensive.
            # Given the constraint, I will assume v=19, k=8, t=5 is the only case.
            # If not, I will return a simple valid cover if possible, or timeout.
            pass
        
        # To avoid infinite loop or complexity in fallback, 
        # I will implement a simple randomized greedy for fallback.
        target = next(iter(uncovered))
        remaining = [p for p in range(v) if p not in target]
        random.shuffle(remaining)
        block = list(target) + remaining[:k-t]
        block = tuple(sorted(block))
        blocks.append(list(block))
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
            
    return blocks

def _solve_19_8_5(deadline):
    v, k, t = 19, 8, 5
    # Precompute
    all_blocks, block_to_tsubs, tsubs, tsub_index = _precompute_incidence(v, k, t)
    num_tsubs = len(tsubs)
    num_blocks = len(all_blocks)
    
    # We represent a solution as a set of block indices.
    # We maintain coverage_count: array of length num_tsubs
    
    # 1. Initial Solution: Greedy Set Cover
    coverage = [0] * num_tsubs
    current_blocks = set()
    
    # To speed up greedy, we can precompute for each t-subset, the list of blocks that cover it.
    # tsub_to_blocks[s] = list of block indices b such that s is in block_to_tsubs[b]
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs_in_block in enumerate(block_to_tsubs):
        for s_idx in tsubs_in_block:
            tsub_to_blocks[s_idx].append(b_idx)
            
    # Greedy:
    while any(c == 0 for c in coverage) and time.time() < deadline - 5:
        # Find uncovered t-subset
        # Optimization: Keep a list of uncovered? 
        # Checking `any(c==0)` is O(N). N=11628. This is fast.
        # But we need to pick the block with max gain.
        # Gain of block b = sum(1 for s in block_to_tsubs[b] if coverage[s] == 0)
        
        best_b = -1
        best_gain = -1
        
        # We can restrict search to blocks that cover at least one uncovered subset.
        # Actually, just iterate all blocks? 75k blocks * 126 subsets = 9.5M ops. 
        # This might be slow in Python if done repeatedly.
        # Better: Pick a random uncovered subset s. 
        # Candidates are blocks in tsub_to_blocks[s].
        # Calculate gain for these candidates.
        
        # Find an uncovered subset
        uncovered_idx = -1
        for i in range(num_tsubs):
            if coverage[i] == 0:
                uncovered_idx = i
                break
        
        if uncovered_idx == -1:
            break
            
        candidates = tsub_to_blocks[uncovered_idx]
        # To improve greedy, we can sample a subset of candidates or all.
        # Let's try all candidates for the first few iterations, then sample.
        # Actually, 11628 subsets. tsub_to_blocks size is average 126 * (C(19,8)/C(19,5))?
        # Number of blocks covering a specific 5-subset: C(19-5, 8-5) = C(14,3) = 364.
        # So candidates list is ~364.
        # 364 * 126 = ~45k ops per step.
        # We need ~300 blocks. 300 * 45k = 13.5M ops. This is fine.
        
        for b in candidates:
            gain = 0
            for s in block_to_tsubs[b]:
                if coverage[s] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_b = b
                
        if best_b == -1:
            break
            
        current_blocks.add(best_b)
        for s in block_to_tsubs[best_b]:
            coverage[s] += 1
            
    # 2. Local Search: Minimize number of blocks
    # Strategy: 
    # While time remains:
    #   Try to remove a redundant block.
    #   If no redundant block, try to swap a block with another to enable removal.
    
    # Helper to check if block b is redundant given current coverage and set
    def is_redundant(b):
        for s in block_to_tsubs[b]:
            if coverage[s] <= 1:
                return False
        return True

    # Helper to remove block b
    def remove_block(b):
        current_blocks.remove(b)
        for s in block_to_tsubs[b]:
            coverage[s] -= 1

    # Helper to add block b
    def add_block(b):
        current_blocks.add(b)
        for s in block_to_tsubs[b]:
            coverage[s] += 1

    # Initial cleanup
    # We iterate through blocks and remove redundant ones.
    # Order matters. We can shuffle.
    
    best_size = len(current_blocks)
    last_improvement = time.time()
    
    while time.time() < deadline - 2:
        improved = False
        
        # Pass 1: Remove redundant blocks
        # We iterate over a copy of the list to allow removal
        blocks_list = list(current_blocks)
        random.shuffle(blocks_list)
        
        for b in blocks_list:
            if b not in current_blocks:
                continue
            if is_redundant(b):
                remove_block(b)
                improved = True
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    last_improvement = time.time()
        
        # Pass 2: Swap-based Local Search
        # Try to swap a block with a non-block to reduce size or maintain size for better structure.
        # Actually, simple removal is often sufficient if the initial greedy is good.
        # But greedy can get stuck.
        # Let's try: Pick a random block B. Try to replace it with a block B' such that 
        # the union of coverage doesn't decrease, and then try to remove more.
        
        # A more powerful move: 
        # 1. Pick a block B to remove.
        # 2. Identify uncovered subsets U (those with coverage 1 and covered by B).
        # 3. If U is empty, remove B (handled in Pass 1).
        # 4. If U is not empty, find a block B' that covers all of U.
        # 5. If such B' exists and B' is not in current_blocks, add B'.
        # 6. Now try to remove B.
        # 7. If B can be removed, and we added B', the size is same. 
        # 8. Then run cleanup to see if size decreases.
        
        # This is complex. Let's stick to a simpler heuristic:
        # Randomly swap a block with a random other block, if it doesn't break coverage.
        # Then cleanup.
        
        if len(current_blocks) > 0:
            b_remove = random.choice(list(current_blocks))
            # Find subsets only covered by b_remove
            critical_subsets = []
            for s in block_to_tsubs[b_remove]:
                if coverage[s] == 1:
                    critical_subsets.append(s)
            
            if not critical_subsets:
                # Should have been removed in Pass 1, but maybe race condition or logic gap?
                # Just in case:
                remove_block(b_remove)
                improved = True
            else:
                # We need to cover critical_subsets with another block
                # Find a candidate block that covers ALL critical_subsets
                # We can use the first critical subset to find candidates
                first_crit = critical_subsets[0]
                candidates = tsub_to_blocks[first_crit]
                
                best_candidate = -1
                # We want a candidate that covers all critical_subsets
                # And ideally, is not already in the set (to allow swap)
                # Or if it is in the set, we can't swap it in (we'd have duplicate blocks, which is not allowed? 
                # "blocks must be distinct". So we can't add a block that's already there.)
                
                # So we look for b' not in current_blocks that covers all critical_subsets.
                for b_cand in candidates:
                    if b_cand in current_blocks:
                        continue
                    
                    # Check if b_cand covers all critical_subsets
                    covers_all = True
                    for s in critical_subsets:
                        # Check if s is in block_to_tsubs[b_cand]
                        # We can precompute a set for b_cand? 
                        # block_to_tsubs is a list. Membership check is O(L).
                        # L=126. Critical subsets size is small (usually 1 or 2).
                        # Let's just check.
                        if s not in block_to_tsubs[b_cand]:
                            covers_all = False
                            break
                    
                    if covers_all:
                        best_candidate = b_cand
                        break # Take the first one found? Or best?
                        # "Best" could be one that covers most other non-critical subsets?
                        # For now, first found is okay.
                
                if best_candidate != -1:
                    # Perform swap
                    remove_block(b_remove)
                    add_block(best_candidate)
                    improved = True
                    # After swap, run a quick cleanup
                    # (We will do full cleanup in next iteration of while loop)
        
        if not improved:
            # No immediate improvement found. 
            # We can try a "kick" - randomly change the set slightly?
            # Or just break if we've been stuck for a while?
            # Given the time limit, if we can't improve, we stop.
            # However, sometimes we need to accept a neutral move to escape local minima.
            # Let's try a few neutral swaps if time permits.
            
            if time.time() < deadline - 5:
                # Try 10 random neutral swaps
                found_neutral = False
                for _ in range(10):
                    if not current_blocks: break
                    b_remove = random.choice(list(current_blocks))
                    critical_subsets = [s for s in block_to_tsubs[b_remove] if coverage[s] == 1]
                    
                    if critical_subsets:
                        first_crit = critical_subsets[0]
                        candidates = tsub_to_blocks[first_crit]
                        # Randomly pick a candidate that covers all criticals
                        valid_cands = []
                        for b_cand in candidates:
                            if b_cand in current_blocks: continue
                            covers_all = True
                            for s in critical_subsets:
                                if s not in block_to_tsubs[b_cand]:
                                    covers_all = False
                                    break
                            if covers_all:
                                valid_cands.append(b_cand)
                        
                        if valid_cands:
                            b_new = random.choice(valid_cands)
                            # Perform swap
                            remove_block(b_remove)
                            add_block(b_new)
                            found_neutral = True
                            break # Restart cleanup loop with new state
                    
                if not found_neutral:
                    # Stuck. Break to save time? Or continue?
                    # If we break, we return current best.
                    break

    return [list(all_blocks[b]) for b in current_blocks]

# To ensure the code is robust, we wrap the main logic
if __name__ == "__main__":
    # Self-test
    blocks = construct(19, 8, 5)
    print(f"Found {len(blocks)} blocks")
    # Verify
    v, k, t = 19, 8, 5
    covered = set()
    for b in blocks:
        for ts in itertools.combinations(b, t):
            covered.add(ts)
    total = sum(1 for _ in itertools.combinations(range(v), t))
    print(f"Covered {len(covered)}/{total} subsets")
    if len(covered) != total:
        print("INVALID COVER")
    else:
        print("VALID COVER")
