"""
Constructs the shortest known superpermutation for small n using the classic recursive method.
For n=7, this yields length 5913.
"""
import time
import sys

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Base cases
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Classic recursive construction
    # word is a list of ints
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
            
        # Extract unique permutations of length m-1 from current word
        # The order is determined by their first appearance in the word
        perms = []
        seen = set()
        # Optimization: Use a set for seen, and list for order
        # We iterate through all windows of length m-1
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # A valid permutation must have distinct elements
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
                
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap between new_word and block
            # new_word ends with ...
            # block starts with ...
            max_overlap = 0
            # The overlap cannot be longer than min(len(new_word), len(block))
            # Also, overlap must be a valid suffix/prefix match
            limit = min(len(new_word), len(block))
            # We check from largest possible overlap down to 0
            # However, for superpermutations, the overlap is usually small (1 or 2)
            # But we must be correct.
            for o in range(limit, -1, -1):
                if o == 0:
                    new_word.extend(block)
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    new_word.extend(block[o:])
                    break
            
        word = new_word
        
    return "".join(map(str, word))

if __name__ == "__main__":
    # For testing
    n = 7
    res = construct(n)
    print(len(res))
