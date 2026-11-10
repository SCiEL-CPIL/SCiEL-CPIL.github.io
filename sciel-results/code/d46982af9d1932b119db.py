import itertools
import time
import random
import numpy as np

_DEADLINE_S = 120.0
_MARGIN_S = 10.0


def construct(v, k, t):
    start_time = time.time()
    rng = random.Random(42)

    def remaining():
        return max(0.0, _DEADLINE_S - _MARGIN_S - (time.time() - start_time))

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)
    blocks = []

    # Cache for t-subsets of a block (tuple -> frozenset of tuples)
    block_tsub_cache = {}

    def get_block_tsubs(block):
        key = tuple(block)
        if key not in block_tsub_cache:
            block_tsub_cache[key] = frozenset(itertools.combinations(block, t))
        return block_tsub_cache[key]

    def gain_for_point(block, x, target_set):
        """
        Calculate gain of adding x to block, specifically targeting t-subsets in target_set.
        """
        cnt = 0
        # We need to check if (rest + x) is in target_set
        # rest is a combination of t-1 elements from block
        for rest in itertools.combinations(block, t - 1):
            ts = tuple(sorted(rest + (x,)))
            if ts in target_set:
                cnt += 1
        return cnt

    # Point frequency for tie-breaking / difficulty scoring
    point_freq = np.zeros(v, dtype=np.int32)

    # --- Greedy construction ---
    while uncovered and remaining() > 1.0:
        # Pick a random uncovered t-subset to diversify slightly, or min for determinism?
        # Min is deterministic and often good for baseline. Let's stick to min for stability.
        target = min(uncovered)
        block = list(target)
        block.sort()
        
        # Build the block greedily
        while len(block) < k and remaining() > 0.5:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            best_x = None
            best_gain = -1
            best_freq = 10**9
            for x in candidates:
                g = gain_for_point(block, x, uncovered)
                # Tie-breaker: prefer points with lower frequency
                if g > best_gain or (g == best_gain and point_freq[x] < best_freq):
                    best_gain = g
                    best_freq = point_freq[x]
                    best_x = x
            if best_x is None or best_gain == 0:
                # Pick arbitrary if no gain (prefer low freq)
                best_x = min(candidates, key=lambda x: point_freq[x])
            block.append(best_x)
        
        block.sort()
        block = list(set(block))
        if len(block) < k:
            used = set(block)
            rem = [x for x in range(v) if x not in used]
            rem.sort()
            block.extend(rem[:k - len(block)])
            block.sort()
            
        if len(set(block)) != k:
            continue
            
        covered = get_block_tsubs(block)
        if not covered:
            continue
        
        for ts in covered:
            uncovered.discard(ts)
        
        for p in block:
            point_freq[p] += 1
            
        blocks.append(block)

    # --- Local Improvement Phase ---
    # We will iterate:
    # 1. Remove redundant blocks.
    # 2. Try to replace blocks with better ones (targeting unique coverage).
    # 3. Simulated Annealing for further optimization.
    
    if remaining() > 2.0:
        # Helper to check if a block is redundant
        def is_redundant(block_idx, current_blocks):
            b = current_blocks[block_idx]
            cov_b = get_block_tsubs(b)
            for ts in cov_b:
                # Check if ts is covered by any other block
                only_this = True
                for j in range(len(current_blocks)):
                    if block_idx == j:
                        continue
                    if ts in get_block_tsubs(current_blocks[j]):
                        only_this = False
                        break
                if only_this:
                    return False
            return True

        improved = True
        while improved and remaining() > 2.0:
            improved = False
            
            # Step 1: Remove redundant blocks
            new_blocks = []
            for i in range(len(blocks)):
                if is_redundant(i, blocks):
                    improved = True
                else:
                    new_blocks.append(blocks[i])
            blocks = new_blocks
            
            # Step 2: Try to improve individual blocks
            # For each block, find its unique coverage.
            # If unique coverage is non-empty, try to construct a new block that covers it better.
            n = len(blocks)
            for i in range(n):
                if remaining() < 2.0:
                    break
                b = blocks[i]
                cov_b = get_block_tsubs(b)
                
                # Find unique t-subsets covered by b
                unique_ts = set()
                for ts in cov_b:
                    only_this = True
                    for j in range(n):
                        if i == j:
                            continue
                        if ts in get_block_tsubs(blocks[j]):
                            only_this = False
                            break
                    if only_this:
                        unique_ts.add(ts)
                
                if not unique_ts:
                    continue # Should have been removed in Step 1, but safety check
                
                # Try to build a new block that covers unique_ts
                # Strategy: Start with a random ts from unique_ts, greedily add points to cover more unique_ts
                # If the new block is different from b and covers all unique_ts, replace.
                
                # To avoid infinite loops, we only replace if the new block is "better".
                # Better can mean: same size, but we just want to change the structure.
                # Actually, if we replace b with b_new that covers unique_ts, the count stays same.
                # But it might make other blocks redundant.
                
                # Let's try to construct a block covering unique_ts.
                # We need a block B_new such that unique_ts subset of C(B_new, t).
                
                # Heuristic:
                # 1. Pick a ts from unique_ts.
                # 2. Start B_new with ts.
                # 3. Add points to B_new to maximize coverage of remaining unique_ts.
                
                if len(unique_ts) == 0:
                    continue
                    
                target_ts = next(iter(unique_ts))
                new_block = list(target_ts)
                new_block.sort()
                
                while len(new_block) < k:
                    candidates = [x for x in range(v) if x not in new_block]
                    if not candidates:
                        break
                    best_x = None
                    best_gain = -1
                    for x in candidates:
                        g = gain_for_point(new_block, x, unique_ts)
                        if g > best_gain:
                            best_gain = g
                            best_x = x
                    if best_x is None or best_gain == 0:
                        # If we can't cover more unique_ts, pick random to fill
                        best_x = rng.choice(candidates)
                    new_block.append(best_x)
                
                new_block.sort()
                new_block = list(set(new_block))
                if len(new_block) < k:
                    used = set(new_block)
                    rem = [x for x in range(v) if x not in used]
                    rng.shuffle(rem)
                    new_block.extend(rem[:k - len(new_block)])
                    new_block.sort()
                
                # Verify new_block covers all unique_ts
                cov_new = get_block_tsubs(new_block)
                if unique_ts.issubset(cov_new):
                    # If new_block is different from old block, replace it
                    if new_block != b:
                        blocks[i] = new_block
                        improved = True

    # --- Simulated Annealing Phase ---
    # Objective: Minimize number of blocks.
    # Moves:
    # 1. Remove a block (if valid, i.e., doesn't uncover any t-subset).
    # 2. Swap a block for a random one (accept if valid and maybe better? No, swap keeps size same).
    # 3. Add a random block (if it helps? No, we want to minimize).
    # Actually, since we want to minimize blocks, the only beneficial move is removal.
    # However, removal might block further removals.
    # Let's try a different approach:
    # Randomly perturb the set of blocks, then try to remove redundant ones.
    # Or: Randomly select a block, replace it with a random block, then try to remove redundants.
    
    if remaining() > 5.0:
        # Define a function to check validity and count uncovered
        def check_validity(current_blocks):
            covered = set()
            for b in current_blocks:
                covered.update(get_block_tsubs(b))
            uncovered_count = 0
            for ts in all_tsubsets:
                if ts not in covered:
                    uncovered_count += 1
            return uncovered_count
        
        # Initial state
        current_blocks = [b[:] for b in blocks]
        best_blocks = [b[:] for b in blocks]
        best_uncovered = check_validity(best_blocks)
        
        # Temperature
        T = 1.0
        T_min = 0.01
        cooling = 0.99
        
        while remaining() > 5.0 and T > T_min:
            # Pick a move
            move_type = rng.choice(['remove', 'swap', 'add'])
            
            if move_type == 'remove' and len(current_blocks) > 0:
                # Try to remove a random block
                idx = rng.randint(0, len(current_blocks) - 1)
                new_blocks = current_blocks[:idx] + current_blocks[idx+1:]
                # Check if still valid
                if len(new_blocks) < len(best_blocks):
                    unc = check_validity(new_blocks)
                    if unc == 0:
                        current_blocks = new_blocks
                        best_blocks = [b[:] for b in current_blocks]
                        best_uncovered = 0
                    else:
                        # Accept with probability exp(-unc/T) ? No, we must be valid.
                        # If invalid, revert.
                        pass
                elif len(new_blocks) == len(best_blocks):
                    # No change in size, but might help future removals?
                    # For now, only accept if it doesn't increase uncovered (which is 0 for best).
                    # Actually, if current is valid, and we remove a block, it might become invalid.
                    # We only keep valid states.
                    unc = check_validity(new_blocks)
                    if unc == 0:
                        current_blocks = new_blocks
                        # Update best if we found a smaller one (not possible here)
            
            elif move_type == 'swap' and len(current_blocks) > 0:
                idx = rng.randint(0, len(current_blocks) - 1)
                # Generate a random block
                new_block = list(rng.sample(range(v), k))
                new_block.sort()
                new_blocks = current_blocks[:]
                new_blocks[idx] = new_block
                unc = check_validity(new_blocks)
                if unc == 0:
                    # Valid swap. Does it help?
                    # We can try to remove redundants from this new state.
                    # But for now, just accept if valid.
                    current_blocks = new_blocks
            
            elif move_type == 'add' and len(current_blocks) < len(all_tsubsets):
                # Add a random block
                new_block = list(rng.sample(range(v), k))
                new_block.sort()
                new_blocks = current_blocks + [new_block]
                unc = check_validity(new_blocks)
                if unc == 0:
                    # Valid. Might allow removals later?
                    # But we want to minimize, so adding is usually bad unless it enables removals.
                    # Let's skip adding for now to keep code simple and focused on minimization.
                    pass
            
            # Try to remove redundants from current state
            if remaining() > 5.0:
                improved = True
                while improved and remaining() > 5.0:
                    improved = False
                    for i in range(len(current_blocks)):
                        if is_redundant(i, current_blocks):
                            current_blocks.pop(i)
                            improved = True
                            break # Restart check
            
            # Update best
            if len(current_blocks) < len(best_blocks):
                unc = check_validity(current_blocks)
                if unc == 0:
                    best_blocks = [b[:] for b in current_blocks]
                    best_uncovered = 0
            
            T *= cooling
            
            # If we reached the theoretical lower bound or something, break?
            # No, just rely on time.

        blocks = best_blocks

    return blocks
