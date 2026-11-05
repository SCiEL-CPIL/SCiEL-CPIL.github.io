import time

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit, leave 10s margin
    
    word = [1]
    
    for m in range(2, n + 1):
        # Check time
        if time.time() > deadline:
            break
            
        # Collect unique (m-1)-permutations in order of appearance
        perms = []
        seen = set()
        # Iterate through windows of length m-1 in the current word
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # A valid permutation must contain distinct elements
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            # Construct the block: p[0]...p[m-2] m p[0]...p[m-3]
            # Wait, the standard block is p + [m] + p[:-1]
            # Let's verify:
            # m=2, p=[1]. Block: 1, 2, 1. (p + [2] + p[:-1] -> [1] + [2] + [] -> [1,2]. Missing last 1?)
            # Standard: 1, 2, 1.
            # p = [1]. p + [2] + p[:-1] is [1, 2]. Length 2.
            # The block should be length 2m-1.
            # For m=2, length 3.
            # So block = p + [m] + p
            # Let's check m=3.
            # p=[1,2]. Block: 1, 2, 3, 1, 2.
            # p + [3] + p[:-1] -> [1,2] + [3] + [1] -> [1,2,3,1]. Length 4.
            # But we need 1,2,3,1,2.
            # So block = p + [m] + p
            # Let's re-verify the standard construction formula.
            # Replace pi with pi1...pi_{n-1} n pi1...pi_{n-2}
            # This is p + [n] + p[:-1].
            # For n=2, p=[1]. p + [2] + p[:-1] = [1] + [2] + [] = [1, 2].
            # But the word for n=2 is 1, 2, 1.
            # Why?
            # The window is 1.
            # In the word 1, 2, 1, the windows of length 2 are 12 and 21.
            # The construction appends the block.
            # If block is [1, 2], and we start with empty, we get [1, 2].
            # Then next perm?
            # In W_1=[1], there is only one perm [1].
            # So W_2 should be [1, 2].
            # But [1, 2] does not contain 21.
            # So the standard construction for n=2 must be different or my understanding of the "block" is slightly off regarding the overlap.
            
            # Let's trace the standard algorithm carefully.
            # W_1 = 1
            # Perms of 1 in W_1: [1]
            # For p=[1], block is 1, 2, 1.
            # Why 1, 2, 1?
            # The permutation 1 becomes 1, 2, 1.
            # This contains 12 and 21.
            # So block = p + [m] + p.
            
            # Let's check n=3.
            # W_2 = 1, 2, 1.
            # Perms of 2 in W_2:
            # i=0: 1, 2.
            # i=1: 2, 1.
            # p1 = [1, 2]. Block: 1, 2, 3, 1, 2.
            # p2 = [2, 1]. Block: 2, 1, 3, 2, 1.
            
            # Start new_word = []
            # Add block1: 1, 2, 3, 1, 2.
            # new_word = [1, 2, 3, 1, 2]
            
            # Add block2: 2, 1, 3, 2, 1.
            # Overlap with end of new_word.
            # new_word ends with ... 1, 2.
            # block2 starts with 2, 1...
            # Overlap 1: '2' matches '2'.
            # Append 1, 3, 2, 1.
            # new_word = [1, 2, 3, 1, 2, 1, 3, 2, 1].
            
            # Length 9. Correct.
            
            block = list(p) + [m] + list(p)
            
            # Find max overlap
            # We want to find the largest k such that new_word[-k:] == block[:k]
            # k can be from min(len(new_word), len(block)) down to 0
            max_overlap = 0
            min_len = min(len(new_word), len(block))
            for k in range(min_len, -1, -1):
                if k == 0:
                    max_overlap = 0
                    break
                if new_word[-k:] == block[:k]:
                    max_overlap = k
                    break
            
            new_word += block[max_overlap:]
        
        word = new_word
    
    return "".join(map(str, word))
