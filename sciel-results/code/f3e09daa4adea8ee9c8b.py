"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120.0 # 20s margin for safety

    # Classic recursive construction of length 1! + 2! + ... + n!
    word = [1]
    
    for m in range(2, n + 1):
        # Check time
        if time.time() > deadline:
            return "".join(map(str, word))

        perms, seen = [], set()
        # Extract unique permutations of length m-1 from the current word
        # The current word contains all permutations of 1..m-1
        # We iterate through windows of length m-1
        # We only want those that are valid permutations (distinct elements)
        # and we want them in the order of first appearance.
        
        # Optimization: Use a set for O(1) lookup
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation of 1..m-1
            # Since we build recursively, all windows of length m-1 
            # that are permutations have already been validated in previous steps?
            # No, new windows appear.
            # But we can just check len(set(window)) == m-1.
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        # Construct the new word by inserting m
        for p in perms:
            # Block is p + [m] + p
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap between end of new_word and start of block
            # Max possible overlap is len(block) - 1
            # But practically, it's limited by the structure.
            # We check overlaps from largest to smallest.
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            # Optimization: Start from a reasonable guess?
            # The overlap is at least the length of the longest common suffix/prefix.
            # We can iterate o from max_overlap down to 0.
            # To speed up, we can break early if found.
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            # Append the non-overlapping part
            new_word += block[overlap:]
            
            # Check time periodically
            if time.time() > deadline:
                # If we run out of time, we can't complete the construction.
                # Returning a partial word will be INVALID.
                # It is better to let it finish if close, or return best effort?
                # The task says "Valid when every permutation... occurs".
                # A partial word is invalid.
                # So we should not return early with an invalid word.
                # We should only return early if we are sure it's valid?
                # No, we must return a valid word.
                # If we time out, we score 0.
                # So we must finish.
                # The loop is fast enough for n=7.
                pass

        word = new_word

    return "".join(map(str, word))
# EVOLVE-BLOCK-END
