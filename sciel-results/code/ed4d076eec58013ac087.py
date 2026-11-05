import itertools
import random
import sys

def construct(v, k, t):
       # Precompute all t-subsets
       all_t_subsets = list(itertools.combinations(range(v), t))
       t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
       M = len(all_t_subsets)
       
       # We will run multiple iterations of Greedy + Reduction
       best_blocks = None
       best_len = float('inf')
       
       # Time limit check
       import time
       start_time = time.time()
       time_limit = 110 # Leave 10s margin for finalization/return
        
       # Number of iterations? Let's try 200 fast ones.
       # Each greedy run is very fast.
       iterations = 0
       
       while time.time() - start_time < time_limit:
           iterations += 1
           uncovered = set(range(M)) # Use indices for speed
           blocks = []
           
           # Helper to get t-subsets from a block (list of points)
           def get_t_subsets_idx(block):
               return [t_subset_to_idx[tuple(s)] for s in itertools.combinations(block, t)]
           
           while uncovered:
               # Pick a random uncovered t-subset index
               # To be faster, we can pick a random element from the set
               # Set.pop() is O(1) average, but we need to keep the element for block construction
               # So we sample one, then remove it? No, we need to know which one we picked to start the block.
               # Let's pick a random element.
               # To avoid O(N) sampling from a set every time (which is fine for 11k), 
               # we can just pick one.
               # For better heuristic, we could pick the one with max "conflict", but let's start with random.
               target_idx = random.choice(list(uncovered)) # O(N) conversion to list is slow.
               
               # Optimization: Keep uncovered as a list? 
               # If we use a list, removing an element is O(N).
               # Using a set is better for O(1) removal, but sampling is O(N).
               # Given M=11628, O(N) sampling is ~10k ops.
               # 335 blocks * 10k = 3.3M ops per run.
               # 200 runs * 3.3M = 660M ops. Might be too slow in Python.
               
               # Better: Use a list of uncovered indices, and swap-remove.
               # But we need to pick a "good" one.
               
               # Let's stick to the seed's approach of picking min(uncovered) but randomize the *order* of points in the block construction?
               # No, let's optimize the sampling.
               
               # Alternative: Pick a random t-subset from the *uncovered* set.
               # We can maintain a list `uncovered_list` and a set `uncovered_set` for O(1) removal check.
               # To sample randomly from `uncovered_set` efficiently without converting to list:
               # We can just pick a random index from 0..M-1 and check if it's in `uncovered_set`.
               # Expected number of tries is M / |uncovered_set|.
               # Initially 1. Ends up being ~10x when few left.
               # This is O(1) expected time.
               
               # Let's implement this.
               while True:
                   candidate_idx = random.randint(0, M - 1)
                   if candidate_idx in uncovered:
                       break
               
               target_subset = all_t_subsets[candidate_idx]
               block = list(target_subset)
               
               # Greedy expansion
               while len(block) < k:
                   best_gain = -1
                   best_point = -1
                   # Candidates are points not in block
                   candidates = [x for x in range(v) if x not in block]
                   
                   # To speed up, we can precompute something?
                   # For each candidate x, gain = number of t-subsets in (block + x) that are in uncovered.
                   # t-subsets in (block + x) are formed by picking t-1 from block and 1 from {x} OR t from block (but x is new, so must be in t-subset? No, t-subset must be subset of block+x).
                   # The new t-subsets covered by adding x are those that contain x.
                   # So we look at combinations of (block, t-1) + (x,).
                   
                   # We can iterate over combinations of block of size t-1.
                   # There are C(len(block), t-1) of them.
                   # len(block) starts at t=5, so C(5,4)=5.
                   # Next len=6, C(6,4)=15.
                   # Next len=7, C(7,4)=35.
                   # Total checks per block: 5+15+35 = 55 combinations.
                   # For each combination, we form a tuple, sort it (or keep sorted), and check if its index is in `uncovered`.
                   
                   # We need to map (tuple of points) to index.
                   # `t_subset_to_idx` does this.
                   
                   for x in candidates:
                       gain = 0
                       # Generate t-1 subsets of current block
                       for rest in itertools.combinations(block, t - 1):
                           # Create the t-subset
                           # rest is a tuple of t-1 points. x is a point.
                           # We need to sort them to match the key in t_subset_to_idx
                           # Since block is sorted, and we add x, we can merge efficiently or just sort.
                           # Sorting 5 elements is fast.
                           s = tuple(sorted(rest + (x,)))
                           idx = t_subset_to_idx.get(s)
                           if idx is not None and idx in uncovered:
                               gain += 1
                               
                       if gain > best_gain:
                           best_gain = gain
                           best_point = x
                       elif gain == best_gain and best_point != -1:
                           # Tie break: random or smallest?
                           # Let's use random to explore
                           if random.random() < 0.5:
                               best_point = x
                               
                   block.append(best_point)
                   block.sort()
                   
               # Block is complete. Add to blocks.
               blocks.append(block)
               
               # Update uncovered
               # Get all t-subsets in this block
               block_t_subsets = get_t_subsets_idx(block)
               for idx in block_t_subsets:
                   uncovered.discard(idx)
                   
           # End while uncovered
           
           # Local Search: Reduction
           # Count coverage
           coverage_counts = [0] * M
           for b in blocks:
               for idx in get_t_subsets_idx(b):
                   coverage_counts[idx] += 1
                   
           improved = True
           while improved:
               improved = False
               # Try to remove each block
               # Iterate backwards to allow removal without index shifting issues
               for i in range(len(blocks) - 1, -1, -1):
                   b = blocks[i]
                   # Check if we can remove b
                   can_remove = True
                   # Decrement counts
                   for idx in get_t_subsets_idx(b):
                       coverage_counts[idx] -= 1
                       if coverage_counts[idx] == 0:
                           can_remove = False
                           break
                   
                   if can_remove:
                       # Keep removed
                       blocks.pop(i)
                       improved = True
                   else:
                       # Restore counts
                       for idx in get_t_subsets_idx(b):
                           coverage_counts[idx] += 1
           
           # Update best
           if len(blocks) < best_len:
               best_len = len(blocks)
               best_blocks = [list(b) for b in blocks]
               
           # If we reached a very good number, we might stop early?
           # Best known is 335. If we get <= 335, we can stop?
           # But we want to minimize.
           
       return best_blocks
