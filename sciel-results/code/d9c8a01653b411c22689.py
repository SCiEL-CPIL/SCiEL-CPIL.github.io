"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def _find_max_overlap(s1, s2, max_len):
    """Find the maximum overlap where suffix of s1 matches prefix of s2."""
    max_ov = min(len(s1), len(s2), max_len)
    for o in range(max_ov, 0, -1):
        if s1[-o:] == s2[:o]:
            return o
    return 0

def construct(n):
    start_time = time.time()
    deadline = start_time + 110

    if n == 1:
        return "1"

    word = "1"
    # For n=1, word is "1". Permutations of 1 is [1].
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break

        # Extract all (m-1)-permutations from word in order of appearance
        perms, seen = [], set()
        # The word contains permutations of 1..(m-1)
        # We need to extract them as they appear
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            # Check if window is a permutation of 1..(m-1)
            # Since we are building recursively, all windows of length m-1 
            # in the previous step's word that were permutations are valid.
            # However, the word might contain non-permutation windows if 
            # we didn't enforce it, but the recursive construction 
            # ensures all windows of length m-1 are permutations?
            # No, only the ones we placed.
            # Actually, in the recursive construction, the word is formed 
            # by concatenating blocks. The blocks are perms + m + perms.
            # The overlaps are between the end of one block and start of next.
            # The resulting word contains all (m-1)-perms.
            
            # Efficient check: window must have m-1 distinct chars
            if len(set(window)) == m - 1:
                # Also check if it's a valid permutation of 1..m-1
                # Since symbols are 1..m-1, distinctness is enough.
                if window not in seen:
                    seen.add(window)
                    perms.append(window)

        if m < n:
            new = ""
            for p in perms:
                block = p + str(m) + p
                ov = _find_max_overlap(new, block, len(new))
                new += block[ov:]
            word = new
        else:
            # For the final step, we just build the word
            # The logic is the same
            new = ""
            for p in perms:
                block = p + str(m) + p
                ov = _find_max_overlap(new, block, len(new))
                new += block[ov:]
            word = new

    return word
# EVOLVE-BLOCK-END
