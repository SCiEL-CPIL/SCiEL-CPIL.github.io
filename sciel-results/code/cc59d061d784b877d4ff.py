"""Improved greedy covering for C(v,k,t) with deadline guard and multi-candidate selection."""
import itertools
import time

def construct(v, k, t):
    # Deadline: return best valid covering before 115s wall-clock (safety margin under 120s)
    deadline = time.time() + 115.0
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # Coverage count: how many times each t-subset is covered
    cover_count = [0] * n_tsubs
    
    # Set of uncovered t-subset indices
    uncovered = set(range(n_tsubs))
    
    blocks = []
    
    def gain_of_point(x, block_list):
        """Count uncovered t-subsets that would be covered by adding x to block_list."""
        block_set = set(block_list)
        need = t - len(block_list) - 1
        if need < 0:
            return 0
        if need == 0:
            sub = tuple(sorted(block_list + [x]))
            idx = tsub_to_idx.get(sub)
            if idx is not None and idx in uncovered:
                return 1
            return 0
        remaining = [p for p in range(v) if p not in block_set and p != x]
        count = 0
        for extra in itertools.combinations(remaining, need):
            sub = tuple(sorted(block_list + [x] + list(extra)))
            idx = tsub_to_idx.get(sub)
            if idx is not None and idx in uncovered:
                count += 1
        return count
    
    def make_block(start_sub):
        """Greedily extend start_sub to a full block of size k."""
        block = list(start_sub)
        block_set = set(block)
        while len(block) < k:
            best_x = -1
            best_g = -1
            for x in range(v):
                if x in block_set:
                    continue
                g = gain_of_point(x, block)
                if g > best_g:
                    best_g = g
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
            block_set.add(best_x)
        block.sort()
        return block
    
    def add_block(block):
        """Add a block and update coverage."""
        blocks.append(block)
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None:
                cover_count[idx] += 1
                if cover_count[idx] == 1:
                    uncovered.discard(idx)
    
    # Initial block
    first_idx = min(uncovered)
    first_sub = all_tsubs[first_idx]
    block = make_block(first_sub)
    add_block(block)
    
    # Main greedy loop with multi-candidate every 10 steps
    iteration = 0
    while uncovered:
        if time.time() > deadline - 3.0:
            break
        
        iteration += 1
        
        if iteration % 10 == 0 and time.time() < deadline - 10.0:
            # Try multi-candidate: pick 5 uncovered t-subsets, make blocks, pick best
            sample_indices = list(uncovered)[:5]
            best_block = None
            best_gain = -1
            for idx in sample_indices:
                sub = all_tsubs[idx]
                b = make_block(sub)
                # Compute gain
                g = 0
                for ts in itertools.combinations(b, t):
                    i = tsub_to_idx.get(ts)
                    if i is not None and i in uncovered:
                        g += 1
                if g > best_gain:
                    best_gain = g
                    best_block = b
            if best_block is not None:
                add_block(best_block)
        else:
            # Standard greedy
            first_idx = min(uncovered)
            first_sub = all_tsubs[first_idx]
            block = make_block(first_sub)
            add_block(block)
    
    # Final safety pass if still uncovered
    while uncovered and time.time() < deadline - 0.5:
        first_idx = min(uncovered)
        first_sub = all_tsubs[first_idx]
        block = make_block(first_sub)
        add_block(block)
    
    return blocks
