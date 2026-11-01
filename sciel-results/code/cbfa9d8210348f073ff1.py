"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict

def _time_limit():
    return 130.0  # safety margin before 150s kill

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + _time_limit()
    
    # Precompute all t-subsets for quick reference
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # We'll track coverage using a bitset-like structure (list of booleans or integers)
    # For v=24, k=7, t=3: C(24,3) = 2024, manageable
    
    def greedy_cover(uncovered_set, random_seed=None):
        """Greedy covering algorithm."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
        
        uncovered = set(uncovered_set)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            start_ts = rng.choice(list(uncovered))
            block = list(start_ts)
            
            # Greedily add points to maximize coverage
            while len(block) < k:
                best_gain = -1
                best_point = None
                block_set = set(block)
                
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count how many new t-subsets this point covers
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and best_point is not None:
                        # Tie-breaking: prefer points that appear in more uncovered subsets
                        pass
                
                if best_point is None:
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        
        return blocks
    
    def local_search(blocks, uncovered_set, max_iters=50):
        """Try to remove redundant blocks or swap blocks to improve."""
        if not blocks:
            return blocks, uncovered_set
        
        # Count how many times each t-subset is covered
        cover_count = defaultdict(int)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cover_count[ts] += 1
        
        # Try removing blocks that have all their t-subsets covered by other blocks
        new_blocks = []
        for block in blocks:
            can_remove = True
            for ts in itertools.combinations(block, t):
                if cover_count[ts] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove this block
                for ts in itertools.combinations(block, t):
                    cover_count[ts] -= 1
            else:
                new_blocks.append(block)
        
        return new_blocks, uncovered_set
    
    # First, do a deterministic greedy to get a baseline
    baseline_blocks = greedy_cover(set(range(n_t)))
    
    # Try multiple random restarts
    best_blocks = baseline_blocks
    best_count = len(baseline_blocks)
    
    rng = random.Random(42)
    n_restarts = 20  # Number of random restarts to try
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Randomized greedy with different random seeds
        seed = rng.randint(0, 2**31)
        blocks = greedy_cover(set(range(n_t)), random_seed=seed)
        
        # Apply local search
        blocks, _ = local_search(blocks, set(range(n_t)))
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Convert indices back to point sets
    # Wait, I was working with indices of t-subsets, not points directly.
    # Let me redo this more carefully.
    
    # Actually, the greedy_cover above works with point sets directly.
    # Let me restructure.
    
    return best_blocks

# Let me rewrite this more carefully
def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    def greedy_cover(random_seed=None):
        """Greedy covering: open a block on an uncovered t-subset, add points greedily."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
        
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 5:
                break
            
            # Pick an uncovered t-subset
            if random_seed is not None:
                start_ts = rng.choice(list(uncovered))
            else:
                start_ts = min(uncovered)
            
            block = list(start_ts)
            
            while len(block) < k:
                best_gain = -1
                best_point = None
                block_set = set(block)
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None:
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        
        return blocks, uncovered
    
    def try_remove_redundant(blocks):
        """Remove blocks that are redundant (all their t-subsets covered by others)."""
        if not blocks:
            return blocks
        
        cover_count = defaultdict(int)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cover_count[ts] += 1
        
        new_blocks = []
        for block in blocks:
            can_remove = all(cover_count[ts] > 1 for ts in itertools.combinations(block, t))
            if can_remove:
                for ts in itertools.combinations(block, t):
                    cover_count[ts] -= 1
            else:
                new_blocks.append(block)
        
        return new_blocks
    
    # Deterministic greedy first
    baseline_blocks, baseline_uncovered = greedy_cover(None)
    best_blocks = baseline_blocks
    best_count = len(baseline_blocks)
    
    if baseline_uncovered:
        # Incomplete - shouldn't happen with greedy, but handle gracefully
        pass
    
    # Random restarts
    rng = random.Random(12345)
    
    for i in range(30):
        if time.time() > deadline - 15:
            break
        
        seed = rng.randint(0, 2**31)
        blocks, uncovered = greedy_cover(seed)
        
        if not uncovered and len(blocks) < best_count:
            blocks = try_remove_redundant(blocks)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
