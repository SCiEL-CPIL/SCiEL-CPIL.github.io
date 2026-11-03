import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Recursive construction
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            # Return best effort so far (which is valid for m-1, but not n)
            # But we must return a valid superpermutation for n.
            # If we time out, we might return an invalid one.
            # Better to just return the recursive one if it's fast enough.
            # The recursive construction for n=7 is very fast (<1s).
            pass
            
        perms, seen = [], set()
        # Extract all unique (m-1)-permutations from current word
        # The seed logic:
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between end of 'new' and start of 'block'
            # Max possible overlap is min(len(new), len(block))
            # But actually, overlap is at most m-1 because the new symbol m is unique?
            # No, the block starts with p[0].
            # The overlap must match a suffix of 'new' with a prefix of 'block'.
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    return "".join(map(str, word))
