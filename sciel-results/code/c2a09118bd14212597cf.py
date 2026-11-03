import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if n <= 1:
        return "1"
    
    # 1. Generate standard superpermutation
    # S_1 = 1
    # S_m = S_{m-1} with m inserted
    # This produces length n! + (n-1)! + ... + 1!
    
    word = [1]
    for m in range(2, n + 1):
        # Extract unique permutations of 1..m-1 from current word
        # The standard construction inserts m after each occurrence of a permutation of 1..m-1
        # effectively. 
        # Actually, the recursive definition is:
        # S_1 = 1
        # S_n = S_{n-1} with n inserted before the last element of each permutation of 1..n-1?
        # No, the standard "insertion" construction is:
        # S_n = S_{n-1} \to insert n after every occurrence of each permutation of 1..n-1?
        
        # Let's use the standard algorithm:
        # S_1 = 1
        # S_n = S_{n-1} with n inserted after each occurrence of each (n-1)-permutation?
        # Actually, the length is n! + (n-1)! + ...
        
        # Standard construction:
        # S_1 = 1
        # S_2 = 121
        # S_3 = 1213121
        # S_4 = 121312141213121
        
        # Algorithm:
        # Start with S_{n-1}.
        # Find all occurrences of permutations of 1..n-1.
        # Insert n after each one?
        
        # Let's stick to the code provided in the prompt which works:
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Max overlap
            overlap = 0
            max_ov = min(len(new_word), len(block))
            for o in range(max_ov, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    L = len(word)
    
    # Precompute permutation to ID mapping
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    counts = [0] * num_perms
    
    # Helper to get ID
    # Using a local variable for speed
    pid = perm_to_id.get
    
    # Initial count pass
    for j in range(L - n + 1):
        t = tuple(word[j:j+n])
        id_val = pid(t, -1)
        if id_val != -1:
            counts[id_val] += 1
            
    # Greedy Deletion Loop
    # We scan left to right. If we delete, we stay at the same index.
    # If we can't delete, we move right.
    # We repeat passes until no changes or time out.
    
    # To potentially escape local minima, we can randomize the start position
    # or do multiple passes.
    
    passes = 0
    while time.time() < deadline:
        changed = False
        L = len(word)
        i = 0
        # Randomize start index to break symmetries?
        # For n=7, symmetry breaking might help.
        # But standard left-to-right is usually robust.
        
        while i < L:
            if time.time() > deadline - 2:
                break
                
            # Check safety at i
            # Affected windows: j in [max(0, i-n+1), min(i, L-n)]
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            
            safe = True
            affected_ids = []
            
            # Check counts
            for j in range(start_min, start_max + 1):
                t = tuple(word[j:j+n])
                id_val = pid(t, -1)
                if id_val != -1:
                    affected_ids.append(id_val)
                    if counts[id_val] == 1:
                        safe = False
                        break
            
            if safe:
                # Perform deletion
                for id_val in affected_ids:
                    counts[id_val] -= 1
                    
                word.pop(i)
                L -= 1
                changed = True
                
                # Increment counts for new windows
                new_start_min = max(0, i - n + 1)
                new_start_max = min(i, L - n)
                
                for j in range(new_start_min, new_start_max + 1):
                    t = tuple(word[j:j+n])
                    id_val = pid(t, -1)
                    if id_val != -1:
                        counts[id_val] += 1
                
                # Do not increment i, re-check new character at i
            else:
                i += 1
                
        passes += 1
        if not changed:
            # If no changes in a full pass, we are at a local minimum.
            # We can try a different strategy:
            # 1. Try deleting from right to left?
            # 2. Try random restarts?
            # 3. Just break, as further passes won't change anything.
            break
            
    return "".join(map(str, word))
