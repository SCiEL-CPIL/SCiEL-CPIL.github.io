"""Improved covering design C(19,7,4) with greedy + local search."""
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    """Construct a covering design C(v,k,t) with minimum blocks."""
    start_time = time.time()
    deadline = start_time + 110  # 110s to leave safety margin
    
    # Total t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to its index
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute: for each block (k-subset), which t-subsets it covers
    # We'll compute this on the fly for efficiency
    
    def block_covers(block):
        """Return set of t-subset indices covered by this block."""
        block_set = set(block)
        covered = set()
        for ts in itertools.combinations(block, t):
            covered.add(tsub_idx[ts])
        return covered
    
    # Greedy construction with better heuristics
    def greedy_construct():
        uncovered = set(range(n_tsubsets))
        blocks = []
        
        while uncovered:
            # Choose the t-subset that's hardest to cover (appears in fewest potential blocks)
            # For simplicity, pick the first uncovered one, but we can be smarter
            # Actually, let's pick randomly from uncovered to allow diversity
            target = random.choice(list(uncovered))
            target_ts = all_tsubsets[target]
            
            # Build a block containing target_ts that maximizes coverage
            best_block = None
            best_gain = -1
            
            # Start with the target t-subset
            base = list(target_ts)
            
            # Try to extend to k elements, choosing points that maximize new coverage
            block = list(base)
            
            while len(block) < k:
                block_set = set(block)
                # Current coverage of this partial block
                current_covered = set()
                for ts in itertools.combinations(block, t):
                    if tsub_idx[ts] in uncovered:
                        current_covered.add(tsub_idx[ts])
                
                best_point = None
                best_point_gain = -1
                
                for p in range(v):
                    if p in block_set:
                        continue
                    new_block = block + [p]
                    new_set = set(new_block)
                    # Count how many uncovered t-subsets this would cover
                    gain = 0
                    for ts in itertools.combinations(new_set, t):
                        if tsub_idx[ts] in uncovered:
                            gain += 1
                    if gain > best_point_gain:
                        best_point_gain = gain
                        best_point = p
                
                if best_point is not None:
                    block.append(best_point)
            
            block.sort()
            covered = block_covers(block)
            blocks.append(block)
            uncovered -= covered
        
        return blocks
    
    # Local search: try to remove redundant blocks
    def remove_redundant(blocks):
        """Remove blocks that don't cover any unique t-subset."""
        if not blocks:
            return blocks
        
        # Count how many blocks cover each t-subset
        coverage_count = defaultdict(int)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                coverage_count[tsub_idx[ts]] += 1
        
        # Try removing blocks that only cover t-subsets covered by others
        new_blocks = []
        for block in blocks:
            # Check if this block covers any t-subset that only it covers
            essential = False
            for ts in itertools.combinations(block, t):
                if coverage_count[tsub_idx[ts]] == 1:
                    essential = True
                    break
            
            if not essential:
                # Remove this block
                for ts in itertools.combinations(block, t):
                    coverage_count[tsub_idx[ts]] -= 1
            else:
                new_blocks.append(block)
        
        return new_blocks
    
    # Local search: try swapping points in blocks
    def local_search(blocks, max_iters=1000):
        """Try to improve by swapping points in blocks."""
        blocks = list(blocks)
        n = len(blocks)
        
        # Build coverage structure
        def build_coverage(block_list):
            cov = defaultdict(list)  # tsub_idx -> list of block indices
            for bi, block in enumerate(block_list):
                for ts in itertools.combinations(block, t):
                    cov[tsub_idx[ts]].append(bi)
            return cov
        
        cov = build_coverage(blocks)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            for bi in range(n):
                if time.time() > deadline:
                    break
                block = blocks[bi]
                for pi in range(len(block)):
                    old_point = block[pi]
                    # Try replacing with each other point
                    for new_point in range(v):
                        if new_point == old_point:
                            continue
                        if new_point in block:
                            continue
                        
                        # Check if this swap maintains coverage
                        new_block = block[:pi] + [new_point] + block[pi+1:]
                        
                        # Check: does every t-subset in old_block that was only covered by this block
                        # still get covered?
                        valid = True
                        for ts in itertools.combinations(block, t):
                            if new_point not in ts and old_point in ts:
                                # This t-subset is lost from this block
                                # Check if it's covered by another block
                                if len(cov[tsub_idx[ts]]) == 1:
                                    valid = False
                                    break
                        if not valid:
                            continue
                        
                        # Also check: new t-subsets in new_block that weren't in old_block
                        # are fine (they add coverage)
                        
                        # Perform the swap
                        blocks[bi] = sorted(new_block)
                        improved = True
                        # Rebuild coverage (expensive but safe)
                        cov = build_coverage(blocks)
                        break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return blocks
    
    # Main algorithm
    best_blocks = None
    best_len = float('inf')
    
    # Try multiple greedy constructions with different random seeds
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        
        random.seed(trial)
        blocks = greedy_construct()
        
        # Remove redundant blocks
        blocks = remove_redundant(blocks)
        
        # Local search
        if time.time() < deadline - 10:
            blocks = local_search(blocks, max_iters=200)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        if best_len <= 140:
            break
    
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = greedy_construct()
    
    return best_blocks
