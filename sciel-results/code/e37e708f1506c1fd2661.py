import itertools
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a frequency-based greedy heuristic.
    """
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    
    # Initialize point frequencies
    # Each point appears in C(v-1, t-1) subsets initially
    freq = np.full(v, np.binomial(v-1, t-1), dtype=np.int64)
    
    blocks = []
    
    # Pre-compute nothing heavy, but we can use a local variable for speed
    t_minus_1 = t - 1
    
    while uncovered:
        # 1. Start a new block with the point that has the highest frequency
        # among the uncovered subsets.
        start_point = int(np.argmax(freq))
        block = [start_point]
        
        # 2. Greedily extend the block
        while len(block) < k:
            # Calculate gain for each candidate x not in block
            # Gain(x) = number of uncovered t-subsets of the form S U {x}
            # where S is a (t-1)-subset of current block
            
            # Get all (t-1)-subsets of the current block
            block_combos = list(itertools.combinations(block, t_minus_1))
            
            best_x = -1
            best_gain = -1
            
            # Candidates are points not in the current block
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                for combo in block_combos:
                    # Create the t-subset tuple
                    new_subset = tuple(sorted(combo + (x,)))
                    
                    if new_subset in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            # Fallback if no gain (should not happen if uncovered is not empty)
            if best_x == -1:
                for x in range(v):
                    if x not in block:
                        best_x = x
                        break
            
            block.append(best_x)
        
        # 3. Finalize the block
        block.sort()
        blocks.append(block)
        
        # 4. Remove all t-subsets covered by this block from uncovered
        for subset in itertools.combinations(block, t):
            if subset in uncovered:
                uncovered.discard(subset)
                for p in subset:
                    freq[p] -= 1
            
    return blocks
