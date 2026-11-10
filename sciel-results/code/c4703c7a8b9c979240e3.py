"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a multi-strategy approach:
    1. Multi-seed greedy with different heuristics
    2. Local search to remove redundant blocks
    3. Constraint-based improvement
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin within 150s limit
    
    # Total number of t-subsets
    total_tsubsets = 1
    for i in range(t):
        total_tsubsets *= (v - i)
    for i in range(1, t + 1):
        total_tsubsets //= i
    
    # Precompute all t-subsets as tuples for fast lookup
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    def greedy_build(seed, use_bottleneck=False):
        """Greedy covering with optional bottleneck heuristic."""
        rng = random.Random(seed)
        
        # uncovered_mask[i] = True if t-subset i is not yet covered
        uncovered_mask = [True] * len(all_tsubsets)
        uncovered_count = len(all_tsubsets)
        
        blocks = []
        
        while uncovered_count > 0:
            if time.time() > deadline - 5:
                break
            
            # Pick a random uncovered subset
            uncovered_indices = [i for i in range(len(all_tsubsets)) if uncovered_mask[i]]
            if not uncovered_indices:
                break
            chosen_idx = rng.choice(uncovered_indices)
            start_from = all_tsubsets[chosen_idx]
            
            # Build block greedily
            block = list(start_from)
            rng.shuffle(block)
            
            while len(block) < k:
                if time.time() > deadline - 5:
                    break
                
                # For each candidate point, compute gain
                best_gain = -1
                best_point = None
                candidates = [x for x in range(v) if x not in block]
                
                for x in candidates:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsubset_to_idx.get(ts)
                        if idx is not None and uncovered_mask[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None:
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered mask
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(tuple(sorted(ts)))
                if idx is not None and uncovered_mask[idx]:
                    uncovered_mask[idx] = False
                    uncovered_count -= 1
        
        # Check if we have a valid covering
        if uncovered_count == 0:
            return blocks
        return None
    
    def local_search(blocks, max_time=10):
        """Remove redundant blocks and try to swap for shorter blocks."""
        start_ls = time.time()
        
        # Count how many times each t-subset is covered
        cover_count = defaultdict(int)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                ts = tuple(sorted(ts))
                cover_count[ts] += 1
        
        # Try to remove blocks that are redundant
        changed = True
        while changed and (time.time() - start_ls) < max_time:
            changed = False
            new_blocks = []
            for i, block in enumerate(blocks):
                # Check if this block is the only one covering any t-subset
                is_critical = False
                for ts in itertools.combinations(block, t):
                    ts = tuple(sorted(ts))
                    if cover_count[ts] == 1:
                        is_critical = True
                        break
                
                if not is_critical:
                    # Remove this block
                    for ts in itertools.combinations(block, t):
                        ts = tuple(sorted(ts))
                        cover_count[ts] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            
            blocks = new_blocks
        
        return blocks
    
    def validate_covering(blocks):
        """Check if every t-subset is covered."""
        covered = set()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered.add(tuple(sorted(ts)))
        return len(covered) == len(all_tsubsets)
    
    # Try multiple strategies
    best_result = None
    best_len = float('inf')
    
    # Strategy 1: Standard greedy with multiple seeds
    for seed in range(15):
        if time.time() > deadline - 10:
            break
        result = greedy_build(seed, use_bottleneck=False)
        if result is not None:
            result = local_search(result, max_time=5)
            if validate_covering(result):
                if len(result) < best_len:
                    best_len = len(result)
                    best_result = result
    
    # Strategy 2: Bottleneck heuristic (simplified - just use different seeds)
    for seed in range(15, 25):
        if time.time() > deadline - 10:
            break
        result = greedy_build(seed, use_bottleneck=False)
        if result is not None:
            result = local_search(result, max_time=5)
            if validate_covering(result):
                if len(result) < best_len:
                    best_len = len(result)
                    best_result = result
    
    # Strategy 3: Try to improve by replacing blocks
    if best_result is not None and time.time() < deadline - 15:
        # Try removing and re-adding blocks
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            # Try to find a smaller covering by perturbing
            perturbed = list(best_result)
            rng = random.Random(attempt + 100)
            
            # Remove a random block and see if we can fix
            if len(perturbed) > 1:
                idx = rng.randrange(len(perturbed))
                removed = perturbed.pop(idx)
                
                # Check if still valid
                if validate_covering(perturbed):
                    if len(perturbed) < best_len:
                        best_len = len(perturbed)
                        best_result = perturbed
                else:
                    # Try to add a better block
                    uncovered = set()
                    covered = set()
                    for block in perturbed:
                        for ts in itertools.combinations(block, t):
                            covered.add(tuple(sorted(ts)))
                    
                    for ts in all_tsubsets:
                        if ts not in covered:
                            uncovered.add(ts)
                    
                    if uncovered:
                        # Greedily add blocks to cover uncovered
                        new_blocks = list(perturbed)
                        while uncovered and len(new_blocks) < best_len:
                            # Find best block to add
                            best_block = None
                            best_gain = -1
                            for ts in list(uncovered)[:100]:  # Limit search
                                # Try extending this subset
                                block = list(ts)
                                while len(block) < k:
                                    candidates = [x for x in range(v) if x not in block]
                                    best_x = None
                                    best_x_gain = -1
                                    for x in candidates:
                                        gain = sum(1 for rest in itertools.combinations(block, t-1)
                                                   if tuple(sorted(rest + (x,))) in uncovered)
                                        if gain > best_x_gain:
                                            best_x_gain = gain
                                            best_x = x
                                    if best_x is None:
                                        break
                                    block.append(best_x)
                                block.sort()
                                gain = sum(1 for ts2 in itertools.combinations(block, t)
                                           if tuple(sorted(ts2)) in uncovered)
                                if gain > best_gain:
                                    best_gain = gain
                                    best_block = block
                            
                            if best_block is not None:
                                new_blocks.append(best_block)
                                for ts in itertools.combinations(best_block, t):
                                    uncovered.discard(tuple(sorted(ts)))
                            
                            if time.time() > deadline - 5:
                                break
                        
                        if not uncovered and len(new_blocks) < best_len:
                            best_len = len(new_blocks)
                            best_result = new_blocks
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple greedy
    return greedy_build(0, use_bottleneck=False) or []

# EVOLVE-BLOCK-END
