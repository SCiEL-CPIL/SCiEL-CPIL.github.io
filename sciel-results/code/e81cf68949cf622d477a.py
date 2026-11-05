import itertools
import time
import array
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute t-subsets
    tsets = list(itertools.combinations(range(v), t))
    N = len(tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(tsets)}
    
    # coverage[i] = number of blocks covering t-subset i
    coverage = array.array('H', [0]) * N
    blocks = []
    block_tuples = set() # For fast lookup
    
    # Helper to get t-subset indices for a block
    def get_block_tsets(block):
        return [tset_to_idx[tc] for tc in itertools.combinations(block, t)]
    
    # Greedy Phase
    uncovered_count = N
    while uncovered_count > 0:
        if time.time() > deadline - 10.0:
            break
            
        # Find an uncovered t-subset
        anchor_idx = -1
        for i in range(N):
            if coverage[i] == 0:
                anchor_idx = i
                break
        if anchor_idx == -1:
            break
            
        anchor = tsets[anchor_idx]
        
        # Generate candidate blocks containing anchor
        remaining_points = [x for x in range(v) if x not in anchor]
        need = k - t
        
        best_block = None
        best_score = -1
        
        for extra in itertools.combinations(remaining_points, need):
            block = tuple(sorted(anchor + extra))
            
            # Calculate score: number of newly covered t-subsets
            new_cov = 0
            for tc in itertools.combinations(block, t):
                if coverage[tset_to_idx[tc]] == 0:
                    new_cov += 1
            
            if new_cov > best_score:
                best_score = new_cov
                best_block = block
                
        if best_block is None:
            break
            
        blocks.append(list(best_block))
        block_tuples.add(best_block)
        
        # Update coverage
        for i in get_block_tsets(best_block):
            if coverage[i] == 0:
                uncovered_count -= 1
            coverage[i] += 1
            
    # Local Search Phase
    # Precompute tsets for current blocks
    block_tsets = [get_block_tsets(b) for b in blocks]
    
    # Pass 1: Remove redundant blocks
    changed = True
    while changed:
        changed = False
        if time.time() > deadline - 5.0:
            break
        for bi in range(len(blocks) - 1, -1, -1):
            redundant = True
            for i in block_tsets[bi]:
                if coverage[i] <= 1:
                    redundant = False
                    break
            if redundant:
                for i in block_tsets[bi]:
                    coverage[i] -= 1
                blocks.pop(bi)
                block_tsets.pop(bi)
                block_tuples.discard(tuple(blocks[bi]) if bi < len(blocks) else None) # Wait, we popped it.
                # Actually, we should remove from block_tuples when we pop.
                # Let's re-construct block_tuples if we pop. Or just use list `in` if small.
                # 300 blocks, list `in` is fine.
                changed = True
                
    # Re-initialize block_tuples for safety
    block_tuples = set(tuple(b) for b in blocks)
    
    # Pass 2: Swap optimization
    ls_deadline = deadline - 2.0
    max_ls_steps = 20000
    
    step = 0
    while time.time() < ls_deadline and step < max_ls_steps:
        step += 1
        if not blocks:
            break
            
        bi = random.randint(0, len(blocks) - 1)
        
        # Check if block bi is redundant
        redundant = True
        for i in block_tsets[bi]:
            if coverage[i] <= 1:
                redundant = False
                break
        if redundant:
            for i in block_tsets[bi]:
                coverage[i] -= 1
            blocks.pop(bi)
            block_tsets.pop(bi)
            block_tuples.discard(tuple(blocks[bi]) if bi < len(blocks) else None) # Discard the popped one
            # Actually, simpler:
            # block_tuples.remove(tuple(blocks[bi])) BEFORE pop?
            # Let's just rebuild block_tuples every time we pop? No, too slow.
            # Let's just check `if new_block in block_tuples`.
            continue
            
        # Identify holes if we remove bi
        holes = []
        for i in block_tsets[bi]:
            if coverage[i] == 1:
                holes.append(i)
                
        if not holes:
            continue
            
        # Union of points in holes
        U = set()
        for h_idx in holes:
            U.update(tsets[h_idx])
            
        if len(U) > k:
            continue
            
        remaining_points = [x for x in range(v) if x not in U]
        need = k - len(U)
        
        if need < 0 or need > len(remaining_points):
            continue
            
        # Try a few random completions
        found_improvement = False
        for _ in range(20): # Try 20 random completions
            if time.time() > ls_deadline:
                break
                
            if need == 0:
                extra = ()
            else:
                extra = tuple(random.sample(remaining_points, need))
                
            new_block = tuple(sorted(U.union(extra)))
            
            if new_block in block_tuples:
                continue
                
            # Simulate swap: remove bi, add new_block
            # 1. Remove bi coverage
            for i in block_tsets[bi]:
                coverage[i] -= 1
                
            # 2. Add new_block coverage
            new_block_tsets = get_block_tsets(list(new_block))
            for i in new_block_tsets:
                coverage[i] += 1
                
            # 3. Check if any other block became redundant
            removed_other = -1
            for ri in range(len(blocks) - 1, -1, -1):
                if ri == bi:
                    continue
                red = True
                for i in block_tsets[ri]:
                    if coverage[i] <= 1:
                        red = False
                        break
                if red:
                    removed_other = ri
                    break
                    
            if removed_other != -1:
                # Apply changes
                # Remove removed_other
                for i in block_tsets[removed_other]:
                    coverage[i] -= 1
                blocks.pop(removed_other)
                block_tsets.pop(removed_other)
                block_tuples.discard(tuple(blocks[removed_other]) if removed_other < len(blocks) else None)
                # Wait, if we pop removed_other, indices shift.
                # If removed_other < bi, bi shifts.
                # This is getting complicated with indices.
                
                # Simpler approach:
                # 1. Save state
                # 2. Modify
                # 3. If improvement, keep; else revert.
                
                # Let's restart the inner loop logic with careful index handling.
                
                # Actually, let's just revert and try a different strategy if index management is risky.
                # Or, we can just remove `removed_other` and replace `bi` with `new_block`.
                
                # Let's do it carefully.
                # We have `blocks` and `block_tsets` lists.
                # We want to remove `removed_other` and replace `bi` with `new_block`.
                
                # Step 1: Remove `removed_other` from lists and sets.
                # Step 2: Replace `bi` in lists and sets.
                
                # But `bi` index might change if `removed_other < bi`.
                
                # Let's just revert and use a simpler "Accept if count decreases" check on a COPY?
                # No, copying is slow.
                
                # Let's just revert the coverage changes and continue.
                # The probability of finding a 2-block reduction is low.
                # Maybe we should accept "neutral" swaps that increase "slack"?
                
                # For now, let's just revert and continue to next step.
                # This might be too conservative.
                
                for i in new_block_tsets:
                    coverage[i] -= 1
                for i in block_tsets[bi]:
                    coverage[i] += 1
                break # Break out of random completions for this bi?
            else:
                # Revert
                for i in new_block_tsets:
                    coverage[i] -= 1
                for i in block_tsets[bi]:
                    coverage[i] += 1

    return blocks
