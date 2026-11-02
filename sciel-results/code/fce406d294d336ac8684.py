import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate all blocks
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # 2. Generate all t-subsets
    all_tsubs = list(itertools.combinations(points, t))
    n_tsubs = len(all_tsubs)
    
    # 3. Map each t-subset to the list of blocks containing it
    # subset_to_blocks: dict[tuple, list[int]]
    subset_to_blocks = {ts: [] for ts in all_tsubs}
    
    # Iterate through all blocks and their t-subsets
    for i, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            subset_to_blocks[ts].append(i)
            
    # 4. Initialize counts
    # Each block covers C(k, t) t-subsets initially
    initial_count = 1
    for _ in range(t):
        initial_count *= 1 # placeholder, use math.comb if available or manual
    # C(k, t)
    def nCr(n, r):
        if r > n: return 0
        if r == 0 or r == n: return 1
        r = min(r, n - r)
        num = 1
        den = 1
        for i in range(r):
            num *= (n - i)
            den *= (i + 1)
        return num // den
        
    init_count = nCr(k, t)
    block_counts = [init_count] * n_blocks
    
    # 5. Greedy selection
    uncovered = set(all_tsubs)
    selected_blocks = []
    
    while uncovered:
        if time.time() > deadline:
            # If we run out of time, we might return an invalid set.
            # But the algorithm is fast, this shouldn't happen.
            # To be safe, if we break early, the result is invalid.
            # We must ensure we finish.
            pass
            
        # Find block with max count
        # Using max with key is O(N)
        max_idx = 0
        max_val = -1
        for i in range(n_blocks):
            if block_counts[i] > max_val:
                max_val = block_counts[i]
                max_idx = i
        
        if max_val == 0:
            # Should not happen if uncovered is not empty and problem is solvable
            break
            
        selected_blocks.append(all_blocks[max_idx])
        
        # Update counts
        # For each t-subset in the selected block
        for ts in itertools.combinations(all_blocks[max_idx], t):
            if ts in uncovered:
                uncovered.remove(ts)
                # Decrement counts for all blocks containing this t-subset
                for b_idx in subset_to_blocks[ts]:
                    block_counts[b_idx] -= 1
                    
    return [list(b) for b in selected_blocks]
