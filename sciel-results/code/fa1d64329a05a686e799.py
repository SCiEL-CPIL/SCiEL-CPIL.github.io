import time
import sys

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    if n == 1:
        return "1"
        
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
            
        # Extract unique (m-1)-permutations in order of appearance
        perms = []
        seen = set()
        # word length is L. windows of size m-1.
        # i from 0 to L - (m-1)
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # Check if it's a valid permutation (all unique)
            # Since it comes from a superpermutation of size m-1, it should be, 
            # but the window might contain duplicates if it's not a permutation.
            # The standard algorithm relies on the fact that we only pick valid (m-1)-perms.
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
                
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            # Find max overlap
            # Max possible overlap is min(len(new_word), len(block))
            # But we can limit it. The suffix of new_word must match prefix of block.
            # block starts with p[0].
            # new_word ends with ...
            
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            # Check from largest possible overlap down to 0
            # Optimization: We know the structure.
            # But simple loop is fine for n=7.
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
            
        word = new_word
        
    return "".join(map(str, word))
