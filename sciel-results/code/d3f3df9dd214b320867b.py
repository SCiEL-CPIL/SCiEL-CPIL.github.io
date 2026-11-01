import itertools
import random
import time

# Precompute blocks and their triple sets for v=24, k=7, t=3
# This is specific to the task parameters, but the function takes v, k, t.
# We should make it generic enough or assume the evaluator uses 24, 7, 3.
# The prompt says "evaluator passes this task's parameters v, k and t".
# So we must handle v, k, t dynamically.

def construct(v, k, t):
    # For v=24, k=7, t=3, total blocks is ~346k.
    # Total triples is ~2024.
    
    # If parameters are too large, this precomputation might be too slow.
    # But 24 is small.
    
    # To be safe and fast, we use a lazy generation or precomputation if size allows.
    # Given the 120s limit, precomputing 346k blocks is fast (<0.1s).
    
    all_blocks = list(itertools.combinations(range(v), k))
    # Map block to index for fast lookup?
    # We need to know which triples a block covers.
    
    # Represent triples as tuples of sorted points.
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Precompute for each block, the list of triple indices it covers
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_idx[tr] for tr in itertools.combinations(b, t)]
        block_triples.append(ts)
        
    # Greedy with restarts
    best_blocks = None
    best_len = float('inf')
    
    # Number of restarts depends on time budget
    start_time = time.time()
    max_time = 110 # Leave 10s buffer
    
    # We will run greedy multiple times with random tie-breaking
    # We need to efficiently find the block with max gain.
    
    # Data structures for greedy:
    # uncovered: boolean array of size num_triples
    # gain[block_idx]: number of uncovered triples covered by this block
    
    num_triples = len(all_triples)
    num_blocks = len(all_blocks)
    
    # To make it faster, we can sort blocks by some heuristic?
    # No, gain changes dynamically.
    
    # Efficient Gain Calculation:
    # When we add a block, we mark its triples as covered.
    # We need to update the gain of all other blocks that share triples with the added block.
    # This is expensive if done naively.
    
    # Alternative: 
    # For each uncovered triple, find the blocks that cover it.
    # The "gain" of a block is the number of uncovered triples it covers.
    # We can maintain a list of uncovered triples.
    # For each step, we want to find the block maximizing gain.
    # We can iterate through uncovered triples and accumulate counts for blocks?
    # That is O(|Uncovered| * AvgBlocksPerTriple).
    # AvgBlocksPerTriple = C(v-t, k-t) = C(21, 4) = 5985.
    # |Uncovered| starts at 2024.
    # 2024 * 5985 ~ 12 million operations per step.
    # Number of steps ~ 76.
    # 76 * 12M = 912M operations. Too slow in Python.
    
    # We need a better greedy.
    # Standard Greedy:
    # 1. Calculate gain for ALL blocks.
    # 2. Pick max.
    # 3. Update gains.
    # Updating gains:
    # When block B is added, for every triple T in B:
    #   For every block B' that covers T:
    #     If T was the only uncovered triple in B' (or just decrement count):
    #        gain[B'] -= 1
    
    # We need: For each triple T, the list of blocks covering T.
    # triple_to_blocks = defaultdict(list)
    # for i, ts in enumerate(block_triples):
    #     for t_idx in ts:
    #         triple_to_blocks[t_idx].append(i)
        
    # This structure is:
    # 2024 triples * ~6000 blocks/triple = 12M entries.
    # Memory is okay (~100MB).
    # Speed:
    # Step 1: Init gains.
    #   For each block, gain = len(block_triples[i]) = 35.
    #   Actually, gain = number of *uncovered* triples. Initially all uncovered, so gain = 35 for all.
    #   Wait, if all gains are equal, we pick randomly or first.
    #   This is bad. We need to pick a block that is "good".
    #   In the first step, all blocks are symmetric.
    
    # Let's refine the gain update.
    # uncovered = [True] * num_triples
    # gains = [0] * num_blocks
    # for i in range(num_blocks):
    #     gains[i] = len(block_triples[i]) # 35
    
    # But this is static. We need dynamic.
    # 
    # Algorithm:
    # 1. Init `uncovered` as set of all triple indices.
    # 2. Init `gains` array.
    #    For each block, gain is number of triples in `uncovered` it covers.
    #    Initially 35.
    # 3. Loop:
    #    a. Find block `b` with max `gains[b]`.
    #       If tie, pick random among maxes.
    #    b. Add `b` to solution.
    #    c. For each triple `t` in `block_triples[b]`:
    #       If `t` is in `uncovered`:
    #          Remove `t` from `uncovered`.
    #          For each block `b_prime` in `triple_to_blocks[t]`:
    #             gains[b_prime] -= 1
    #    d. If `uncovered` is empty, break.
    
    # Complexity:
    # Finding max: O(num_blocks) = 346k.
    # Updating:
    #   Each block added covers 35 triples.
    #   Each triple is in ~6000 blocks.
    #   So for one block added, we update 35 * 6000 = 210,000 gains.
    #   Total steps ~ 76.
    #   Total updates: 76 * 210,000 = 16M updates.
    #   Total max-finds: 76 * 346,000 = 26M comparisons.
    #   Total ops ~ 42M.
    #   Python can do ~10-50M simple ops/sec.
    #   This might take 1-5 seconds per run.
    #   We can do ~10-50 restarts in 120s.
    
    # This is a viable plan.
    
    # Precomputation
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_to_blocks[t_idx].append(i)
            
    # Greedy Function
    def greedy_run(random_seed):
        random.seed(random_seed)
        uncovered = set(range(num_triples))
        gains = [len(ts) for ts in block_triples] # Initial gain is 35
        blocks_used = []
        
        # Optimization: If a block's gain becomes 0, it's useless.
        # We can maintain a list of "active" blocks?
        # Or just scan all 346k each time. 346k * 76 is fine.
        
        while uncovered:
            # Find max gain
            max_gain = -1
            candidates = []
            for i in range(num_blocks):
                if gains[i] > max_gain:
                    max_gain = gains[i]
                    candidates = [i]
                elif gains[i] == max_gain:
                    candidates.append(i)
            
            if max_gain == 0:
                # Should not happen if problem is solvable, but break to avoid infinite loop
                break
                
            # Random tie-breaking
            b = random.choice(candidates)
            blocks_used.append(b)
            
            # Update
            for t_idx in block_triples[b]:
                if t_idx in uncovered:
                    uncovered.remove(t_idx)
                    # Update gains for all blocks covering this triple
                    for b_prime in triple_to_blocks[t_idx]:
                        gains[b_prime] -= 1
                        
        return blocks_used

    # Run multiple restarts
    # Estimate time for one run.
    # Let's try 5 runs first to calibrate, then use remaining time.
    
    # To be safe, let's run 10-20 restarts.
    # We'll track the best (smallest) list.
    
    # Note: `blocks_used` is a list of indices. We need to convert to actual blocks.
    
    # Let's do a quick timing check in the code? No, just run a fixed number.
    # 20 restarts * 3s/run = 60s. Safe.
    
    best_solution = None
    best_size = float('inf')
    
    # Use different seeds
    for i in range(20):
        sol_idx = greedy_run(i)
        if len(sol_idx) < best_size:
            best_size = len(sol_idx)
            best_solution = sol_idx
        # Optional: Check time
        if time.time() - start_time > 100:
            break
            
    if best_solution is None:
        # Fallback to simple greedy if something went wrong
        # Just return first run
        sol_idx = greedy_run(0)
        best_solution = sol_idx
        
    # Convert indices to blocks
    result = [list(all_blocks[i]) for i in best_solution]
    return result
