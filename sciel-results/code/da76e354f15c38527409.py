import itertools
import time
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) - choose as few k-element blocks
    as possible so every t-element subset is in some block.
    
    Strategy: Iterative compression with t-packing scaffold.
    1. Build a maximal t-packing (blocks with disjoint t-subsets) as lower bound structure
    2. Use greedy completion that prioritizes gaps in the packing's coverage
    3. Local search: remove redundant, merge pairs, swap to improve
    """
    deadline = time.time() + 115
    rng = random.Random(42)
    
    # Precompute all t-subsets of range(v)
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_ts = len(all_tsubsets)
    
    # Map each t-subset to its index for fast lookup
    ts_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # For each block (k-subset), precompute which t-subsets it covers
    # We'll use a lazy approach: generate t-subsets on the fly
    
    def block_to_tsubset_indices(block):
        """Return list of indices of t-subsets contained in block."""
        result = []
        for ts in itertools.combinations(block, t):
            result.append(ts_index[ts])
        return result
    
    # Build a t-packing: maximize number of blocks with pairwise disjoint t-subsets
    # This gives us a structural scaffold
    def build_t_packing(deadline, rng):
        """Build a maximal t-packing greedily."""
        covered_ts = set()
        packing = []
        
        # Generate candidate blocks in a shuffled order
        all_blocks = list(itertools.combinations(range(v), k))
        rng.shuffle(all_blocks)
        
        for block in all_blocks:
            if time.time() > deadline:
                break
            tsubs = block_to_tsubset_indices(block)
            # Check if any t-subset is already covered
            if any(ts in covered_ts for ts in tsubs):
                continue
            # Add this block to the packing
            packing.append(block)
            for ts in tsubs:
                covered_ts.add(ts)
        
        return packing, covered_ts
    
    # Greedy completion given a starting set of blocks
    def greedy_complete(initial_blocks, covered_ts, deadline, rng):
        """Complete the covering using greedy set cover."""
        blocks = list(initial_blocks)
        uncovered_count = n_ts - len(covered_ts)
        
        if uncovered_count <= 0:
            return blocks
        
        # Track which t-subsets are uncovered
        uncovered = set(range(n_ts))
        for ts in covered_ts:
            uncovered.discard(ts)
        
        # For each candidate block, compute how many uncovered t-subsets it covers
        # We'll use a priority approach: pick the block that covers the most uncovered
        
        # Pre-compute for efficiency: for each t-subset, which blocks contain it
        # But that's expensive. Instead, we'll do adaptive greedy.
        
        # Build reverse index: for each t-subset, list of blocks that contain it
        # We'll do this lazily
        
        while uncovered and time.time() < deadline:
            # Pick a random uncovered t-subset to focus on
            seed_ts_idx = rng.choice(list(uncovered))
            seed_ts = all_tsubsets[seed_ts_idx]
            
            # Find all blocks that contain this t-subset
            candidates = []
            remaining = [x for x in range(v) if x not in seed_ts]
            for extra in itertools.combinations(remaining, k - t):
                block = tuple(sorted(seed_ts + extra))
                # Check how many uncovered t-subsets this block covers
                gain = 0
                for ts in itertools.combinations(block, t):
                    ts_idx = ts_index[ts]
                    if ts_idx in uncovered:
                        gain += 1
                candidates.append((gain, block))
            
            if not candidates:
                break
            
            # Pick the best candidate (most uncovered covered)
            candidates.sort(key=lambda x: (-x[0], x[1]))
            best_gain, best_block = candidates[0]
            
            if best_gain <= 0:
                break
            
            blocks.append(list(best_block))
            
            # Update uncovered set
            for ts in itertools.combinations(best_block, t):
                ts_idx = ts_index[ts]
                uncovered.discard(ts_idx)
        
        return blocks
    
    # Local search: remove redundant blocks
    def remove_redundant(blocks, deadline):
        """Remove blocks that don't uniquely cover any t-subset."""
        # Build coverage count for each t-subset
        ts_counts = defaultdict(int)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                ts_counts[ts] += 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_blocks = []
            for block in blocks:
                # Check if this block has any uniquely covered t-subset
                unique = False
                for ts in itertools.combinations(block, t):
                    if ts_counts[ts] == 1:
                        unique = True
                        break
                if unique:
                    new_blocks.append(block)
                else:
                    # Remove this block and update counts
                    for ts in itertools.combinations(block, t):
                        ts_counts[ts] -= 1
                    changed = True
            blocks = new_blocks
        
        return blocks
    
    # Local search: merge two blocks into one if possible
    def try_merge(blocks, deadline):
        """Try to merge pairs of blocks into single blocks."""
        n = len(blocks)
        merged = False
        
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                bi = set(blocks[i])
                bj = set(blocks[j])
                
                # Find t-subsets that would become uncovered if we remove both
                uncovered_if_removed = set()
                
                for ts in itertools.combinations(blocks[i], t):
                    ts_t = tuple(sorted(ts))
                    # Check if covered by any other block
                    count = 0
                    for m, bm in enumerate(blocks):
                        if m == i or m == j:
                            continue
                        if all(x in bm for x in ts_t):
                            count += 1
                    if count == 0:
                        uncovered_if_removed.add(ts_t)
                
                for ts in itertools.combinations(blocks[j], t):
                    ts_t = tuple(sorted(ts))
                    count = 0
                    for m, bm in enumerate(blocks):
                        if m == i or m == j:
                            continue
                        if all(x in bm for x in ts_t):
                            count += 1
                    if count == 0:
                        uncovered_if_removed.add(ts_t)
                
                if not uncovered_if_removed:
                    # Both blocks are redundant together
                    blocks.pop(j)
                    blocks.pop(i)
                    merged = True
                    break
                
                # Check if we can form a new block covering all uncovered
                union_points = set()
                for ts in uncovered_if_removed:
                    union_points.update(ts)
                
                if len(union_points) <= k:
                    # Create new block
                    new_block = list(union_points)
                    remaining = [x for x in range(v) if x not in union_points]
                    if len(new_block) < k:
                        new_block.extend(random.sample(remaining, k - len(new_block)))
                    new_block.sort()
                    
                    blocks.pop(j)
                    blocks.pop(i)
                    blocks.append(new_block)
                    merged = True
                    break
            
            if merged:
                break
        
        return blocks, merged
    
    # Main algorithm
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Build t-packing scaffold
    packing_deadline = deadline - 80
    packing, packing_covered = build_t_packing(packing_deadline, rng)
    
    # Phase 2: Greedy completion from packing
    if time.time() < deadline - 20:
        completion_deadline = deadline - 20
        completed = greedy_complete(packing, packing_covered, completion_deadline, rng)
        
        if len(completed) < best_count:
            best_count = len(completed)
            best_blocks = completed
        
        # Phase 3: Local search
        if time.time() < deadline - 10:
            ls_deadline = deadline - 10
            ls_blocks = remove_redundant(completed, ls_deadline)
            ls_blocks, _ = try_merge(ls_blocks, ls_deadline)
            ls_blocks = remove_redundant(ls_blocks, ls_deadline)
            
            if len(ls_blocks) < best_count:
                best_count = len(ls_blocks)
                best_blocks = ls_blocks
    
    # Phase 4: Additional restarts with pure greedy (no packing)
    while time.time() < deadline - 15:
        # Pure greedy from scratch
        greedy_deadline = deadline - 15
        uncovered = set(range(n_ts))
        blocks = []
        
        while uncovered and time.time() < greedy_deadline:
            seed_ts_idx = rng.choice(list(uncovered))
            seed_ts = all_tsubsets[seed_ts_idx]
            
            candidates = []
            remaining = [x for x in range(v) if x not in seed_ts]
            for extra in itertools.combinations(remaining, k - t):
                block = tuple(sorted(seed_ts + extra))
                gain = 0
                for ts in itertools.combinations(block, t):
                    ts_idx = ts_index[ts]
                    if ts_idx in uncovered:
                        gain += 1
                candidates.append((gain, block))
            
            if not candidates:
                break
            
            candidates.sort(key=lambda x: (-x[0], x[1]))
            best_gain, best_block = candidates[0]
            
            if best_gain <= 0:
                break
            
            blocks.append(list(best_block))
            for ts in itertools.combinations(best_block, t):
                ts_idx = ts_index[ts]
                uncovered.discard(ts_idx)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        # Local search
        if time.time() < deadline - 5:
            ls_deadline = deadline - 5
            ls_blocks = remove_redundant(blocks, ls_deadline)
            ls_blocks, _ = try_merge(ls_blocks, ls_deadline)
            ls_blocks = remove_redundant(ls_blocks, ls_deadline)
            
            if len(ls_blocks) < best_count:
                best_count = len(ls_blocks)
                best_blocks = ls_blocks
    
    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = set(range(n_ts))
        blocks = []
        while uncovered and time.time() < deadline:
            seed_ts_idx = rng.choice(list(uncovered))
            seed_ts = all_tsubsets[seed_ts_idx]
            remaining = [x for x in range(v) if x not in seed_ts]
            best_block = None
            best_gain = -1
            for extra in itertools.combinations(remaining, k - t):
                block = tuple(sorted(seed_ts + extra))
                gain = 0
                for ts in itertools.combinations(block, t):
                    if ts_index[ts] in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = list(block)
            if best_gain <= 0:
                break
            blocks.append(best_block)
            for ts in itertools.combinations(best_block, t):
                uncovered.discard(ts_index[ts])
        best_blocks = blocks
    
    return best_blocks

if __name__ == "__main__":
    v, k, t = 19, 8, 5
    start = time.time()
    blocks = construct(v, k, t)
    end = time.time()
    print(f"Number of blocks: {len(blocks)}")
    print(f"Time taken: {end - start:.2f} seconds")
    
    # Verify validity
    covered = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered.add(tuple(sorted(ts)))
    
    all_ts = set(tuple(sorted(ts)) for ts in itertools.combinations(range(v), t))
    if covered == all_ts:
        print("Valid covering found.")
    else:
        print(f"Invalid covering! Missing {len(all_ts - covered)} t-subsets.")
