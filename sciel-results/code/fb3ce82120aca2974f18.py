"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    # Classic recursive construction of length 1! + 2! + ... + n!
    # This yields a valid superpermutation.
    # For n=7, length is 5913.
    # For n=6, length is 873.
    # For n=5, length is 153.
    # For n=4, length is 33.
    
    word = [1]
    for m in range(2, n + 1):
        # Check time
        if time.time() > deadline:
            pass
        
        perms, seen = [], set()
        # Extract all (m-1)-permutations from the current word in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between the end of `new` and start of `block`
            # In the classic construction, the overlap is typically m-1.
            # Try m-1 first (most common), then fall back to smaller overlaps.
            max_possible = min(len(new), len(block))
            overlap = 0
            # Fast path: check m-1 overlap first
            if m - 1 <= max_possible:
                if new[len(new) - (m - 1):] == block[:m - 1]:
                    overlap = m - 1
                else:
                    # Fallback: check smaller overlaps
                    for o in range(m - 2, -1, -1):
                        if o <= max_possible and new[len(new) - o:] == block[:o]:
                            overlap = o
                            break
            else:
                for o in range(max_possible, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
            new += block[overlap:]
        word = new
        
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
