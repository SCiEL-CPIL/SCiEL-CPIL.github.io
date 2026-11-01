"""Superpermutation construction for n symbols."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    def build_superperm(n):
        """Build superpermutation for n symbols using standard algorithm."""
        if n == 1:
            return [1]
        
        prev = build_superperm(n - 1)
        perms = list(permutations(range(1, n)))
        
        blocks = []
        for p in perms:
            block = list(p) + [n] + list(p[1:])
            blocks.append(block)
        
        result = list(blocks[0])
        for i in range(1, len(blocks)):
            block = blocks[i]
            max_overlap = 0
            max_possible = min(len(result), len(block))
            for overlap in range(max_possible, -1, -1):
                if result[len(result) - overlap:] == block[:overlap]:
                    max_overlap = overlap
                    break
            result.extend(block[max_overlap:])
        
        return result
    
    word = build_superperm(n)
    
    # Optimize by removing redundant characters
    # Use efficient checking: maintain a count of each permutation's occurrences
    
    if n <= 7 and len(word) > 0:
        all_perms = list(permutations(range(1, n+1)))
        perm_set = set(all_perms)
        
        # Count occurrences of each permutation in current word
        perm_count = {p: 0 for p in perm_set}
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i+n])
            if window in perm_count:
                perm_count[window] += 1
        
        # Verify all permutations are present
        missing = [p for p in perm_set if perm_count[p] == 0]
        if missing:
            # Shouldn't happen, but return word as is
            return "".join(map(str, word))
        
        # Try to remove characters one by one
        improved = list(word)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            
            for i in range(len(improved)):
                if time.time() > deadline:
                    break
                
                # Determine which windows contain position i
                # Window starting at s contains position i if s <= i < s+n
                # So s ranges from max(0, i-n+1) to min(i, len(improved)-n)
                s_min = max(0, i - n + 1)
                s_max = min(i, len(improved) - n)
                
                # Check if removing position i would make any permutation have count 0
                removable = True
                for s in range(s_min, s_max + 1):
                    window = tuple(improved[s:s+n])
                    if window in perm_count and perm_count[window] == 1:
                        # This permutation only appears in this window
                        # But wait, we need to check if it appears in other windows
                        # that don't contain position i
                        # Actually, if perm_count[window] == 1, and this window contains position i,
                        # then removing position i would remove the only occurrence
                        removable = False
                        break
                
                if removable:
                    # Remove position i
                    # Update perm_count: decrement counts for windows containing position i
                    for s in range(s_min, s_max + 1):
                        window = tuple(improved[s:s+n])
                        if window in perm_count:
                            perm_count[window] -= 1
                    
                    improved = improved[:i] + improved[i+1:]
                    changed = True
                    break
        
        word = improved
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
