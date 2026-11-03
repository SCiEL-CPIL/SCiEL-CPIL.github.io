"""
Covering design C(19,8,5) solver.
Strategy:
1. Fast greedy construction using bitset representation for speed.
2. Local search (ILS) to improve the cover:
   - Try removing blocks that are 'redundant' (every t-subset they cover is also covered by other blocks).
   - If removal leads to an incomplete cover, repair with greedy.
   - Accept if improvement.
3. Deadline guard: return best found before 140s.
"""
import itertools
import time
import random

# Precompute all 5-subsets of 19 points as tuples
# v=19, t=5: C(19,5) = 11628
# k=8: C(8,5) = 56 per block

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 20s safety margin
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each point x, precompute which (t-1)-subsets it can extend
    # For greedy: gain(x) = number of uncovered t-subs that contain x and (t-1) elements from current block
    
    # Greedy construction
    def greedy_construct():
        uncovered = [True] * num_tsubs
        num_uncovered = num_tsubs
        blocks = []
        
        while num_uncovered > 0:
            # Find first uncovered t-subset
            first_idx = -1
            for i in range(num_uncovered):
                if uncovered[i]:
                    first_idx = i
                    break
            if first_idx == -1:
                break
            
            first_tsub = all_tsubs[first_idx]
            block = list(first_tsub)
            
            # Mark this t-subset as covered
            uncovered[first_idx] = False
            num_uncovered -= 1
            
            # Also mark other t-subs within this initial block if any (t-subset is exactly the block subset)
            # Actually, the block has t elements, so only one t-subset (itself)
            
            # Now add points one by one
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_x = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    # Compute gain: how many uncovered t-subs are formed by adding x to block
                    gain = 0
                    # For each (t-1)-subset of current block, check if t-subset with x is uncovered
                    # But we need to be efficient
                    # Current block has m elements, we choose t-1 from them
                    m = len(block)
                    for rest in itertools.combinations(block, t - 1):
                        # rest is a tuple of t-1 elements from block
                        # The t-subset would be sorted(rest + (x,))
                        # But we need to check if it's in tsub_to_idx and uncovered
                        # We can construct the sorted tuple
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x == -1 or best_gain == 0:
                    # No improvement possible, break
                    break
                
                block.append(best_x)
                block_set.add(best_x)
                
                # Mark newly covered t-subs
                # The new t-subs are those that include best_x and (t-1) elements from the rest of the block
                # But we need to be careful: only t-subs that were previously uncovered
                for rest in itertools.combinations([b for b in block if b != best_x], t - 1):
                    ts = tuple(sorted(rest + (best_x,)))
                    idx = tsub_to_idx.get(ts)
                    if idx is not None and uncovered[idx]:
                        uncovered[idx] = False
                        num_uncovered -= 1
        
        blocks.append(sorted(block))
        
        return blocks, uncovered
    
    # Local search: try to remove redundant blocks
    def is_block_redundant(blocks, block_idx, uncovered):
        """Check if removing block_idx still leaves all t-subs covered."""
        block = blocks[block_idx]
        # For each t-subset in this block, check if it's covered by another block
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            if uncovered[idx]:
                # This t-subset is only covered by this block
                return False
        return True
    
    def remove_and_check(blocks, block_idx, uncovered):
        """Temporarily remove block_idx and check if cover is still valid."""
        block = blocks[block_idx]
        # Check if any t-subset in this block is uncovered by others
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            if uncovered[idx]:
                return False, None
        return True, None
    
    def local_search(blocks, uncovered, max_iters=1000):
        """Try to remove redundant blocks."""
        improved = True
        while improved:
            improved = False
            # Shuffle block indices for diversity
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    return blocks, uncovered
                # Check if block idx is redundant
                block = blocks[idx]
                redundant = True
                for ts in itertools.combinations(block, t):
                    tidx = tsub_to_idx[ts]
                    if uncovered[tidx]:
                        redundant = False
                        break
                if redundant:
                    # Remove it
                    del blocks[idx]
                    improved = True
                    break  # Restart with new indices
        
        return blocks, uncovered
    
    # Phase 1: Greedy construction
    blocks, uncovered = greedy_construct()
    initial_count = len(blocks)
    
    # Phase 2: Local search to remove redundant blocks
    blocks, uncovered = local_search(blocks, uncovered)
    
    # Phase 3: Iterated Local Search - perturb and repair
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    if time.time() < deadline:
        # Try multiple ILS iterations
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Perturb: randomly remove a few blocks
            num_remove = min(5, len(best_blocks) // 10 + 1)
            perturb_blocks = list(best_blocks)
            remove_indices = random.sample(range(len(perturb_blocks)), min(num_remove, len(perturb_blocks)))
            for ri in sorted(remove_indices, reverse=True):
                del perturb_blocks[ri]
            
            # Repair: find uncovered t-subs and add blocks
            # First, compute which t-subs are now uncovered
            covered = [False] * num_tsubs
            for blk in perturb_blocks:
                for ts in itertools.combinations(blk, t):
                    idx = tsub_to_idx[ts]
                    covered[idx] = True
            
            uncovered_list = [i for i in range(num_tsubs) if not covered[i]]
            
            if not uncovered_list:
                # Still valid, and we have fewer blocks
                if len(perturb_blocks) < best_count:
                    best_blocks = list(perturb_blocks)
                    best_count = len(perturb_blocks)
                continue
            
            # Greedy repair
            repair_blocks = list(perturb_blocks)
            while uncovered_list:
                if time.time() > deadline:
                    break
                
                # Pick first uncovered
                first_idx = uncovered_list[0]
                first_tsub = all_tsubs[first_idx]
                block = list(first_tsub)
                block_set = set(block)
                
                # Mark as covered
                covered[first_idx] = True
                uncovered_list.remove(first_idx)
                
                while len(block) < k and uncovered_list:
                    best_gain = -1
                    best_x = -1
                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            ts = tuple(sorted(rest + (x,)))
                            idx = tsub_to_idx.get(ts)
                            if idx is not None and not covered[idx]:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    
                    if best_x == -1 or best_gain == 0:
                        break
                    
                    block.append(best_x)
                    block_set.add(best_x)
                    
                    # Mark newly covered
                    for rest in itertools.combinations([b for b in block if b != best_x], t - 1):
                        ts = tuple(sorted(rest + (best_x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and not covered[idx]:
                            covered[idx] = True
                            if idx in uncovered_list:
                                uncovered_list.remove(idx)
                
                repair_blocks.append(sorted(block))
            
            # Check if repair is valid and better
            if not uncovered_list and len(repair_blocks) < best_count:
                best_blocks = repair_blocks
                best_count = len(repair_blocks)
                
                # Try local search on the new best
                if time.time() < deadline:
                    best_blocks, _ = local_search(best_blocks, None)
                    best_count = len(best_blocks)
    
    return best_blocks


if __name__ == "__main__":
    # Test
    blocks = construct(19, 8, 5)
    print(f"Number of blocks: {len(blocks)}")
    # Verify
    all_tsubs = set(itertools.combinations(range(19), 5))
    covered = set()
    for b in blocks:
        for ts in itertools.combinations(b, 5):
            covered.add(ts)
    missing = all_tsubs - covered
    print(f"Missing t-subs: {len(missing)}")
    if missing:
        print("INVALID COVER")
    else:
        print("VALID COVER")
