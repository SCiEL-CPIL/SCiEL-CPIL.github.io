<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Greedy lexicographic packing.
    from itertools import combinations
    covered, blocks = set(), []
    for b in combinations(range(12), 6):
        subsets = list(combinations(b, 5))
        if not any(s in covered for s in subsets):
            blocks.append(list(b))
            covered.update(subsets)
    return blocks
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    """Construct the Steiner system S(5,6,12) via backtracking with symmetry breaking."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Precompute all 6-subsets of 0..11
    all_blocks = list(combinations(range(12), 6))
    
    # Precompute all 5-subsets for each 6-subset
    block_5subs = {}
    for b in all_blocks:
        block_5subs[b] = list(combinations(b, 5))
    
    # Map each 5-subset to the blocks containing it
    from collections import defaultdict
    five_to_blocks = defaultdict(list)
    for b in all_blocks:
        for s in block_5subs[b]:
            five_to_blocks[s].append(b)
    
    # We'll use backtracking with the first block fixed to (0,1,2,3,4,5) for symmetry breaking
    # The Witt design S(5,6,12) has 132 blocks.
    # With symmetry breaking, we fix one block and search for the rest.
    
    # Actually, let's try a smarter approach: use the known structure of the Witt design.
    # The Witt design S(5,6,12) can be constructed from the Mathieu group M12.
    # But let's try backtracking with good heuristics first.
    
    # Use backtracking: select blocks one at a time, ensuring no 5-subset is repeated.
    # Key insight: with the first block fixed, the remaining structure is highly constrained.
    
    # Fix first block
    first_block = (0, 1, 2, 3, 4, 5)
    blocks = [first_block]
    covered = set(block_5subs[first_block])
    
    # Candidate blocks: those that don't share any 5-subset with covered
    def get_candidates(covered_set):
        cands = []
        for b in all_blocks:
            if b == first_block:
                continue
            valid = True
            for s in block_5subs[b]:
                if s in covered_set:
                    valid = False
                    break
            if valid:
                cands.append(b)
        return cands
    
    # Backtracking search
    def backtrack(blocks_list, covered_set, depth):
        if time.time() > deadline:
            return None
        
        if len(blocks_list) == 132:
            return blocks_list[:]
        
        # Get candidates
        cands = get_candidates(covered_set)
        
        if not cands:
            return None
        
        # Heuristic: try blocks that cover "rare" 5-subsets first
        # Count how many other uncovered 5-subsets each candidate block covers
        # Actually, just try in lexicographic order for simplicity first
        
        for b in cands:
            new_covered = covered_set | set(block_5subs[b])
            result = backtrack(blocks_list + [b], new_covered, depth + 1)
            if result is not None:
                return result
            if time.time() > deadline:
                return None
        
        return None
    
    result = backtrack(blocks, covered, 0)
    if result is not None:
        return [list(b) for b in result]
    
    # Fallback: return greedy result if backtracking fails or times out
    from itertools import combinations as comb
    covered_greedy, blocks_greedy = set(), []
    for b in comb(range(12), 6):
        subsets = list(comb(b, 5))
        if not any(s in covered_greedy for s in subsets):
            blocks_greedy.append(list(b))
            covered_greedy.update(subsets)
    return blocks_greedy

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
