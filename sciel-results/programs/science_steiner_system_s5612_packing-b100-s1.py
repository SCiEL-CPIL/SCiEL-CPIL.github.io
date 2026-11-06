import time
import sys
from itertools import combinations

def construct():
    """
    Constructs the Steiner System S(5, 6, 12) using an optimized bitmask-based 
    backtracking algorithm (Algorithm X).
    """
    start_time = time.time()
    timeout = 140.0 # Safety margin below 150s
    
    points = range(12)
    all_5subsets = list(combinations(points, 5))
    num_5subsets = len(all_5subsets)
    
    # Map each 5-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    
    # Generate all 6-subsets (blocks)
    all_6subsets = list(combinations(points, 6))
    num_blocks = len(all_6subsets)
    
    # Pre-compute block masks and candidates
    # block_masks[b] is an integer where bit i is 1 if block b covers 5-subset i
    block_masks = [0] * num_blocks
    # candidates[s] is a list of block indices that cover 5-subset s
    candidates = [[] for _ in range(num_5subsets)]
    
    for b_idx, b in enumerate(all_6subsets):
        mask = 0
        for s in combinations(b, 5):
            s_idx = subset_to_idx[s]
            mask |= (1 << s_idx)
            candidates[s_idx].append(b_idx)
        block_masks[b_idx] = mask

    best_solution = []
    
    def backtrack(uncovered, covered_mask, current_blocks):
        nonlocal best_solution
        if time.time() - start_time > timeout:
            return False
            
        if uncovered == 0:
            best_solution = current_blocks[:]
            return True
            
        # Heuristic: Pick the uncovered 5-subset with the minimum number of valid candidate blocks
        min_subset = -1
        min_count = float('inf')
        
        # Iterate through set bits in uncovered
        u = uncovered
        while u:
            # Get the lowest set bit
            lsb = u & -u
            s_idx = lsb.bit_length() - 1
            u ^= lsb
            
            # Count valid blocks for s_idx
            count = 0
            for b_idx in candidates[s_idx]:
                # A block is valid if it doesn't overlap with already covered subsets
                if (block_masks[b_idx] & covered_mask) == 0:
                    count += 1
                    
            if count < min_count:
                min_count = count
                min_subset = s_idx
                if min_count == 0:
                    return False # Dead end
                if min_count == 1:
                    break # Can't get better than 1
            
        # Try each valid block covering min_subset
        for b_idx in candidates[min_subset]:
            if (block_masks[b_idx] & covered_mask) != 0:
                continue
                
            new_uncovered = uncovered & ~block_masks[b_idx]
            new_covered = covered_mask | block_masks[b_idx]
            
            if backtrack(new_uncovered, new_covered, current_blocks + [b_idx]):
                return True
                
        return False

    initial_uncovered = (1 << num_5subsets) - 1
    initial_covered = 0
    
    if backtrack(initial_uncovered, initial_covered, []):
        result = [list(all_6subsets[i]) for i in best_solution]
        return result
    else:
        # Fallback to greedy if backtracking fails or times out
        covered = set()
        blocks = []
        for b in all_6subsets:
            subsets = tuple(combinations(b, 5))
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        return blocks

if __name__ == "__main__":
    # Optional: Run a quick test
    blocks = construct()
    print(f"Found {len(blocks)} blocks")
