import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Base case
    if n == 1:
        return "1"
    
    # We build the word iteratively.
    # word is a list of integers.
    word = [1]
    
    for m in range(2, n + 1):
        # Extract distinct (m-1)-permutations from word in order of appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # If m is large, we might want to reorder perms to maximize overlap.
        # For m=2,3,4,5,6 the number of perms is small (1, 2, 6, 24, 120).
        # For m=7, it is 720.
        
        if m <= 5:
            # Standard order is likely good enough or we can just use greedy for all
            # Let's use greedy for all m >= 2 to be safe and consistent
            pass
            
        # Greedy reordering of perms
        # We want to order perms P1, P2, ... Pk such that overlap(Pi, Pi+1) is maximized.
        # Overlap(A, B) is length of longest suffix of A that is prefix of B.
        
        def overlap_len(A, B):
            max_ov = 0
            for i in range(1, len(A) + 1):
                if A[-i:] == B[:i]:
                    max_ov = i
            return max_ov

        # Precompute overlaps? 720*720 is ~500k operations, which is fast.
        # But we need to do this for each m.
        # For m=7, perms size is 720.
        
        if len(perms) > 1:
            # Build adjacency or just compute on the fly
            # Greedy Nearest Neighbor
            remaining = set(range(len(perms)))
            # Start with the first perm from original order? Or one with high degree?
            # Let's start with perms[0]
            current_idx = 0
            remaining.remove(current_idx)
            order = [current_idx]
            
            while remaining:
                current_perm = perms[current_idx]
                best_next = -1
                best_score = -1
                # Tie breaker: fewest remaining connections?
                # For simplicity, just max overlap.
                for idx in remaining:
                    next_perm = perms[idx]
                    ov = overlap_len(current_perm, next_perm)
                    # We want to maximize ov
                    if ov > best_score:
                        best_score = ov
                        best_next = idx
                
                if best_next == -1:
                    # Should not happen
                    break
                    
                order.append(best_next)
                remaining.remove(best_next)
                current_idx = best_next
                
            # Reorder perms list according to order
            new_perms = [perms[i] for i in order]
        else:
            new_perms = perms

        # Construct new word
        new_word = []
        # Start with the first block
        first_p = new_perms[0]
        block = list(first_p) + [m] + list(first_p)
        new_word = block
        
        for p in new_perms[1:]:
            block = list(p) + [m] + list(p)
            # Find max overlap between new_word suffix and block prefix
            max_ov = 0
            # Max possible overlap is min(len(new_word), len(block))
            # But block is 2*(m-1)+1.
            # new_word is getting long.
            # We only need to check up to len(block) - 1
            limit = min(len(new_word), len(block) - 1)
            for o in range(limit, 0, -1):
                if new_word[-o:] == block[:o]:
                    max_ov = o
                    break
            
            new_word += block[max_ov:]
            
        word = new_word
        
        # Check time
        if time.time() > deadline:
            break

    return "".join(map(str, word))
