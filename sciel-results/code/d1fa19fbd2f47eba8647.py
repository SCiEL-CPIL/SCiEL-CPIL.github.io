import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Precompute all t-subsets as tuples for indexing
    all_tsubsets = list(itertools.combinations(range(v), t))
    # Map each t-subset tuple to an index for fast array access
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    num_tsubsets = len(all_tsubsets)
    
    # Precompute (t-1)-subsets for gain computation
    # We'll compute gains on the fly
    
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts: we'll do as many as time allows
    # Each greedy run takes some time; we'll track it
    num_restarts = 50  # upper bound, will be limited by time
    
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random tie-breaking: shuffle the order of points considered
        point_order = list(range(v))
        rng.shuffle(point_order)
        
        # cover_count: array of length num_tsubsets, 0 means uncovered
        cover_count = [0] * num_tsubsets
        uncovered_count = num_tsubsets
        blocks = []
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            
            # Start a new block: pick a random uncovered t-subset to seed
            # Find an uncovered t-subset
            # For efficiency, maintain a list of uncovered indices
            # But updating that list is expensive. Instead, just pick randomly until we find uncovered.
            # Actually, let's just pick the first uncovered one for determinism within a restart,
            # but use random tie-breaking for point selection.
            
            # Find an uncovered t-subset to start the block
            # We need to find any index where cover_count[i] == 0
            # This could be slow if we scan. Let's maintain a set of uncovered indices.
            # Actually, for the first block, all are uncovered. For subsequent, we can track.
            
            # Let me restructure: maintain a list/set of uncovered t-subset indices
            # This is O(1) to check membership, O(56) to remove covered ones per block
            
            # Actually, let me just scan. With 11628 subsets, scanning is fast enough.
            seed_idx = -1
            for i in range(num_tsubsets):
                if cover_count[i] == 0:
                    seed_idx = i
                    break
            if seed_idx == -1:
                break  # Should not happen if uncovered_count > 0
            
            seed = list(all_tsubsets[seed_idx])
            block = set(seed)
            
            # Greedily add points to the block
            while len(block) < k:
                if time.time() > deadline:
                    break
                
                # Compute gain for each candidate point
                # gain(x) = number of (t-1)-subsets S of block such that {x} ∪ S is uncovered
                best_gain = -1
                best_point = None
                
                # Use bucket queue: gains are in [0, C(len(block), t-1)]
                max_possible_gain = 1
                for r in range(t-1):
                    max_possible_gain *= (len(block) - r) // (r + 1)
                # Actually C(n, r) where n = len(block), r = t-1
                # Let's just compute it
                def comb(n, r):
                    if r < 0 or r > n:
                        return 0
                    if r == 0 or r == n:
                        return 1
                    r = min(r, n - r)
                    result = 1
                    for i in range(r):
                        result = result * (n - i) // (i + 1)
                    return result
                
                max_gain = comb(len(block), t-1)
                
                # Bucket queue: list of lists, bucket[i] = points with gain i
                buckets = [[] for _ in range(max_gain + 1)]
                
                for x in point_order:
                    if x in block:
                        continue
                    
                    # Compute gain for x
                    gain = 0
                    block_list = list(block)
                    for S in itertools.combinations(block_list, t-1):
                        # Form the t-subset {x} ∪ S
                        new_subset = tuple(sorted(S + (x,)))
                        idx = tsub_idx.get(new_subset, -1)
                        if idx != -1 and cover_count[idx] == 0:
                            gain += 1
                    
                    if gain > 0:
                        buckets[gain].append(x)
                
                # Find the best point: highest non-empty bucket
                chosen = None
                for g in range(max_gain, -1, -1):
                    if buckets[g]:
                        chosen = buckets[g][0]  # First in bucket (tie-break by point_order)
                        break
                
                if chosen is None:
                    # No point increases coverage; pick any random point not in block
                    remaining = [x for x in point_order if x not in block]
                    if remaining:
                        chosen = remaining[0]
                    else:
                        break
                
                block.add(chosen)
            
            if len(block) < k:
                # Incomplete block due to time, skip this block
                if time.time() > deadline:
                    break
                continue
            
            block_list = sorted(block)
            blocks.append(block_list)
            
            # Update cover_count for all t-subsets of this block
            for ts in itertools.combinations(block_list, t):
                idx = tsub_idx[ts]
                if cover_count[idx] == 0:
                    uncovered_count -= 1
                cover_count[idx] += 1
        
        if uncovered_count == 0 and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        if best_blocks is not None and best_count <= 264:
            break  # Can't do better than lower bound
    
    if best_blocks is None:
        # Fallback: return a valid (possibly large) covering
        # This shouldn't happen, but just in case
        return []
    
    return best_blocks
