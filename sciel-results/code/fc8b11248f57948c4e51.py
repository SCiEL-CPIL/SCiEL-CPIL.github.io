import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    all_subsets = list(itertools.combinations(range(v), t))
    best_blocks = None
    best_count = float('inf')
    
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = set(all_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to start the block
            # Converting to list is O(N), but N is small enough (~11k max)
            u = random.choice(list(uncovered))
            block = list(u)
            
            # Extend block to size k
            while len(block) < k:
                best_x = -1
                best_gain = -1
                # Candidates are points not in block
                candidates = [x for x in range(v) if x not in block]
                
                # Pre-compute (t-1) subsets of current block for gain calculation
                block_t_minus_1 = list(itertools.combinations(block, t - 1))
                
                for x in candidates:
                    gain = 0
                    for rest in block_t_minus_1:
                        # Create the t-subset by adding x and sorting
                        new_sub = tuple(sorted(rest + (x,)))
                        if new_sub in uncovered:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        
                block.append(best_x)
                
            block.sort()
            blocks.append(block)
            
            # Remove all t-subsets of this block from uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
                
        return blocks

    def local_search_reduce(blocks, v, t):
        if not blocks:
            return blocks
            
        changed = True
        while changed:
            changed = False
            coverage_map = {}
            for i, b in enumerate(blocks):
                for sub in itertools.combinations(b, t):
                    if sub not in coverage_map:
                        coverage_map[sub] = set()
                    coverage_map[sub].add(i)
                    
            new_blocks = []
            for i, b in enumerate(blocks):
                is_redundant = True
                for sub in itertools.combinations(b, t):
                    # If this block is the ONLY one covering this subset, it's essential
                    if len(coverage_map[sub]) == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_blocks.append(b)
                else:
                    changed = True
                    
            blocks = new_blocks
            
        return blocks

    seed = 42
    while time.time() < deadline:
        blocks = greedy_construct(seed)
        blocks = local_search_reduce(blocks, v, t)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        seed += 1
        
    if best_blocks is None:
        # Fallback: should not happen, but return empty list to avoid crash
        return []
        
    return best_blocks
