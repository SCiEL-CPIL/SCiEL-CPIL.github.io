import time

def construct(n):
    """
    Constructs the shortest known superpermutation using the classic recursive method.
    Length is sum_{k=1}^n k!.
    """
    start_time = time.time()
    
    # For very small n, we can hardcode or use simple search, but the recursive method works for all n.
    # The recursive method produces:
    # n=1: "1" (len 1)
    # n=2: "121" (len 3)
    # n=3: "1321312" is NOT the standard one.
    # Let's trace the seed's logic carefully.
    
    word = [1]
    for m in range(2, n + 1):
        # Find all distinct permutations of size m-1 in the current word, in order of appearance
        perms, seen = [], set()
        # The window size is m-1
        w_size = m - 1
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            # Ensure it's a permutation (all distinct)
            if len(set(window)) == w_size:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            # The block to add is p, then m, then p
            # But we overlap with the existing new_word
            block = list(p) + [m] + list(p)
            
            # Find the maximum overlap between the end of new_word and the start of block
            # Max possible overlap is m-1 (length of p)
            max_overlap = min(m - 1, len(new_word))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
        
        word = new_word
        
        # Safety check for time (though this is very fast)
        if time.time() - start_time > 140:
            break
            
    return "".join(map(str, word))
