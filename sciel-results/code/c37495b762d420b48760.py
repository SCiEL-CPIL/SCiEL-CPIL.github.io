# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 140.0
    
    all_blocks = list(combinations(range(12), 6))
    block_5sets = [set(combinations(b, 5)) for b in all_blocks]
    
    used_5sets = set()
    blocks_found = []
    
    def backtrack(idx, current_blocks):
        if time.time() - start_time > timeout:
            return False
        
        if len(current_blocks) == 132:
            return True
            
        # Heuristic: Find the smallest point not yet "forced" by symmetry?
        # Simpler: Just try blocks in lexicographic order, but skip those that don't contain the smallest point not in the first block?
        
        # Strong Symmetry Breaking:
        # If len(current_blocks) == 0:
        #   Must be (0,1,2,3,4,5)
        # If len(current_blocks) == 1:
        #   Must contain 6.
        # If len(current_blocks) == 2:
        #   Must contain 7? (If 7 is not in B0 or B1)
        
        # Let's just use the "must contain smallest uncovered point" rule for the first few steps.
        
        if len(current_blocks) == 0:
            target_point = None # Handled outside
        elif len(current_blocks) == 1:
            target_point = 6
        elif len(current_blocks) == 2:
            # Check if 7 is in B0 or B1
            if 7 not in current_blocks[0] and 7 not in current_blocks[1]:
                target_point = 7
            else:
                # Find smallest point not in B0 U B1
                union = set(current_blocks[0]).union(current_blocks[1])
                for p in range(12):
                    if p not in union:
                        target_point = p
                        break
                else:
                    target_point = None
        else:
            target_point = None
            
        for i in range(idx, len(all_blocks)):
            b = all_blocks[i]
            b5s = block_5sets[i]
            
            if target_point is not None and target_point not in b:
                continue
                
            if any(s in used_5sets for s in b5s):
                continue
                
            blocks_found.append(b)
            used_5sets.update(b5s)
            
            if backtrack(i + 1, current_blocks + [b]):
                return True
                
            blocks_found.pop()
            used_5sets.difference_update(b5s)
            
        return False

    first_block = (0, 1, 2, 3, 4, 5)
    first_idx = all_blocks.index(first_block)
    
    blocks_found.append(first_block)
    used_5sets.update(block_5sets[first_idx])
    
    if backtrack(first_idx + 1, [first_block]):
        return [list(b) for b in blocks_found]
        
    return [list(b) for b in blocks_found]

# EVOLVE-BLOCK-END
