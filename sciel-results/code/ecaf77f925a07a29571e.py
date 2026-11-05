import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 140
    
    # Standard superpermutation construction
    word = [1]
    for m in range(2, n + 1):
        # Extract all unique permutations of 1..m-1 from the current word
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            # Find max overlap with the end of 'new'
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        
        word = new
        
        if time.time() > deadline:
            break

    if time.time() > deadline:
        return "".join(map(str, word))
    
    # Greedy deletion loop
    # We try to remove characters that are not the unique occurrence of any permutation
    # We repeat until no more deletions are possible
    
    max_iterations = 100 # Safety
    for _ in range(max_iterations):
        if time.time() > deadline - 5:
            break
            
        L = len(word)
        if L <= n:
            break
            
        # Calculate frequency of all n-windows
        freq = {}
        for i in range(L - n + 1):
            p = tuple(word[i:i+n])
            freq[p] = freq.get(p, 0) + 1
        
        # Find a valid deletion
        deleted = False
        # We can iterate in any order. Let's go left to right.
        for pos in range(1, L - 1):
            # Check windows affected by deleting word[pos]
            # Windows starting at i such that i <= pos < i+n
            # i_min = pos - n + 1
            # i_max = pos
            i_min = max(0, pos - n + 1)
            i_max = min(pos, L - n)
            
            valid = True
            for i in range(i_min, i_max + 1):
                p = tuple(word[i:i+n])
                if freq[p] == 1:
                    valid = False
                    break
            
            if valid:
                # Delete word[pos]
                word = word[:pos] + word[pos+1:]
                deleted = True
                break # Restart loop with new word/freq
        
        if not deleted:
            break
            
    return "".join(map(str, word))
