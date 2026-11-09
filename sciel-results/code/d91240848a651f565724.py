"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    """
    Covering design C(v,k,t): choose as few k-subsets of {0..v-1} as possible
    so that every t-subset is contained in at least one block.
    
    Uses a randomized greedy approach with multiple restarts and a deadline guard.
    """
    deadline = time.time() + 115  # 115s safety margin within 120s limit
    
    # Precompute all t-subsets for quick lookup
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    best_blocks = None
    best_count = float('inf')
    
    # Try multiple random seeds/restarts
    rng = np.random.RandomState(42)
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Randomized greedy covering
        uncovered = np.ones(num_tsubs, dtype=bool)
        uncovered_count = num_tsubs
        blocks = []
        
        while uncovered_count > 0:
            if time.time() > deadline - 2:
                break
            
            # Pick a random uncovered t-subset to anchor the block
            unc_idx = np.where(uncovered)[0]
            anchor_idx = unc_idx[rng.randint(len(unc_idx))]
            anchor = list(all_tsubs[anchor_idx])
            block = set(anchor)
            
            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block) < k:
                if time.time() > deadline - 2:
                    break
                
                # Compute gain for each candidate point
                candidates = [x for x in range(v) if x not in block]
                if not candidates:
                    break
                
                # For each candidate, count how many uncovered t-subsets it would cover
                # A t-subset is covered if it's a subset of block ∪ {x} and was uncovered
                gains = {}
                for x in candidates:
                    # Count uncovered t-subsets that contain x and are subsets of block ∪ {x}
                    # i.e., t-subsets S where x ∈ S ⊆ block ∪ {x}
                    # S must contain x, and S \ {x} ⊆ block, and |S| = t
                    # So S = {x} ∪ T where T is a (t-1)-subset of block
                    gain = 0
                    for T in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(T + (x,)))
                        if ts in tsub_index:
                            idx = tsub_index[ts]
                            if uncovered[idx]:
                                gain += 1
                    gains[x] = gain
                
                if max(gains.values()) == 0:
                    # No point adds coverage; pick randomly
                    x = candidates[rng.randint(len(candidates))]
                else:
                    # Weighted random selection based on gain
                    max_gain = max(gains.values())
                    weights = np.array([gains[x] + 0.1 for x in candidates], dtype=float)
                    # Use softmax-like weighting
                    weights = np.exp(weights / max(max_gain, 1.0))
                    weights /= weights.sum()
                    x = candidates[rng.choice(len(candidates), p=weights)]
                
                block.add(x)
            
            if len(block) < k:
                break
            
            block_list = sorted(block)
            
            # Check if this block is new
            if block_list in blocks:
                break
            
            blocks.append(block_list)
            
            # Update uncovered
            for ts in itertools.combinations(block_list, t):
                idx = tsub_index[ts]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
        
        if uncovered_count == 0 and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            # If we've reached a good solution, try to improve with local search
            if best_count <= 155:
                # Try local search: remove redundant blocks
                improved = True
                while improved:
                    improved = False
                    for i in range(len(best_blocks)):
                        if time.time() > deadline - 2:
                            break
                        # Try removing block i
                        test_blocks = best_blocks[:i] + best_blocks[i+1:]
                        # Check if still covers all t-subsets
                        covered = set()
                        for b in test_blocks:
                            for ts in itertools.combinations(b, t):
                                covered.add(ts)
                        if len(covered) == num_tsubs:
                            best_blocks = test_blocks
                            best_count = len(best_blocks)
                            improved = True
                            break
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(all_tsubs)
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
