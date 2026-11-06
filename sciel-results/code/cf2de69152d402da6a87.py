import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 12
    target_blocks = 132
    
    all_blocks = list(combinations(range(n), 6))
    block_5subs = [set(combinations(b, 5)) for b in all_blocks]
    
    covered = set()
    blocks = []
    
    # Symmetry breaking: Fix first block to be (0,1,2,3,4,5)
    # It is the first in lexicographic order
    first_idx = 0
    covered.update(block_5subs[first_idx])
    blocks.append(list(all_blocks[first_idx]))
    
    def backtrack(start_idx):
        if time.time() > deadline:
            return False
        
        if len(blocks) == target_blocks:
            return True
        
        for i in range(start_idx, len(all_blocks)):
            b_5s = block_5subs[i]
            
            # Check conflict
            conflict = False
            for s in b_5s:
                if s in covered:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Add block
            covered.update(b_5s)
            blocks.append(list(all_blocks[i]))
            
            if backtrack(i + 1):
                return True
            
            # Undo
            covered.difference_update(b_5s)
            blocks.pop()
            
        return False
    
    if backtrack(1):
        return blocks
    else:
        # Fallback to greedy if backtracking fails (should not happen)
        covered_greedy = set()
        blocks_greedy = []
        for i in range(len(all_blocks)):
            if not any(s in covered_greedy for s in block_5subs[i]):
                blocks_greedy.append(list(all_blocks[i]))
                covered_greedy.update(block_5subs[i])
        return blocks_greedy
