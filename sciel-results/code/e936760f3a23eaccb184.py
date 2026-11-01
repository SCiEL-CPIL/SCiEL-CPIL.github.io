import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    # Classic recursive construction
    # S_1 = "1"
    # S_m = concat( P + str(m) + P for P in perms_of_S_{m-1} ) with overlap
    
    if n == 1:
        return "1"
        
    word = [1]
    
    for m in range(2, n + 1):
        # Extract unique permutations of length m-1 from word in order of appearance
        # The classic construction relies on the specific order of permutations found in the previous superpermutation
        perms = []
        seen = set()
        
        # Optimization: use tuple for hashing
        # Iterate through all windows of size m-1
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if it's a valid permutation of 1..m-1
            # Since it comes from a superpermutation of 1..m-1, it should be a permutation
            # But we need to ensure we only take each permutation once, in order of first appearance
            if window not in seen:
                # Verify it's a permutation of 1..m-1
                # This check is O(m). m is small (<=7).
                if sorted(window) == list(range(1, m)):
                    seen.add(window)
                    perms.append(window)
                    
        new_word = []
        # We need to append blocks P + [m] + P
        # And find maximal overlap with the current new_word
        
        for p in perms:
            block = list(p) + [m] + list(p)
            
            # Find maximal overlap between end of new_word and start of block
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            # Check from largest possible overlap down to 0
            # Optimization: break early if possible? 
            # Usually overlap is small.
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word.extend(block[overlap:])
            
        word = new_word
        
        # Deadline check
        if time.time() > deadline:
            # Return what we have? No, it must be valid for n.
            # If we run out of time, we can't finish the construction.
            # But n=7 is fast.
            pass

    return "".join(map(str, word))
