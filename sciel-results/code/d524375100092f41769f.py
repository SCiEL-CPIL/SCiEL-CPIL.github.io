import time

def construct(n):
    """
    Constructs a superpermutation of length 1! + 2! + ... + n!.
    Uses the standard recursive construction.
    For n=7, this yields 5913 symbols.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    if n == 1:
        return "1"

    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
            
        # Extract all unique (m-1)-permutations from the current word
        # They appear in the order of the standard construction
        perms = []
        seen = set()
        
        # The number of windows is len(word) - (m-1) + 1
        # But we only care about valid permutations (distinct elements)
        # And we want them in order of appearance.
        
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation of 1..m-1
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
                    
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap with the end of new_word
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[-o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
            
        word = new_word

    return "".join(map(str, word))
