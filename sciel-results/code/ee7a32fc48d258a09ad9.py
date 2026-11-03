"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct(n):
    """
    Returns the shortest known superpermutation for n symbols using the classic recursive construction.
    Length is sum_{i=1}^n i!.
    For n=7, length is 5913.
    """
    if n <= 0:
        return ""
    if n == 1:
        return "1"
    
    # Classic recursive construction
    # We maintain the word as a list of integers for efficient manipulation
    word = [1]
    
    # Deadline guard, though this construction is very fast (O(n * n! * n) roughly)
    # For n=7, it's well under 1 second.
    start_time = time.time()
    
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all distinct (m-1)-length permutations from the current word
        # The number of such permutations is (m-1)!
        # We scan the word to find them in order of appearance.
        for i in range(len(word) - (m - 2)):
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation of 1..m-1
            # Since we build it recursively, all windows of length m-1 are permutations
            # of 1..m-1 if they contain distinct elements.
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            # Block is p, m, p
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap between end of 'new_word' and start of 'block'
            # Overlap length o means new_word[-o:] == block[:o]
            # We want to append block[o:]
            max_possible_overlap = min(len(new_word), len(block))
            overlap = 0
            # Check from largest possible overlap down to 0
            for o in range(max_possible_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
        
        word = new_word
        
        # Safety check for time, though not strictly necessary for n<=7
        if time.time() - start_time > 110:
            break
            
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
