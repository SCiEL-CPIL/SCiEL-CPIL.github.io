import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate classic superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Overlap logic: find max overlap between end of new and start of block
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new)-o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # Optimization: The classic construction is known to be suboptimal for n>=5.
    # Try local search: greedy deletion.
    
    def is_valid(w):
        if len(w) < n:
            return False
        target = set()
        # Generate all permutations of 1..n
        # For n=7, 5040 permutations.
        # We can just check if all permutations are present.
        # Generating all permutations is 5040 * 7! ops? No, just itertools.permutations.
        import itertools
        all_perms = set(itertools.permutations(range(1, n+1)))
        
        present = set()
        for i in range(len(w) - n + 1):
            present.add(tuple(w[i:i+n]))
        return all_perms.issubset(present)
    
    # Initial check
    if not is_valid(word):
        return "".join(map(str, word))
        
    # Greedy deletion
    changed = True
    while changed and (time.time() < deadline):
        changed = False
        for i in range(len(word)):
            if time.time() > deadline:
                break
            # Try removing word[i]
            temp_word = word[:i] + word[i+1:]
            if is_valid(temp_word):
                word = temp_word
                changed = True
                break # Restart scan as indices changed
    
    return "".join(map(str, word))
