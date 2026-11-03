import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    if n <= 1:
        return "1"
        
    # 1. Generate standard superpermutation
    # Standard construction: S_1 = 1, S_n = S_{n-1} with n inserted
    # Efficient generation for n=7
    word = [1]
    for m in range(2, n + 1):
        # Extract unique permutations of 1..m-1
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
    # n! permutations
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    counts = [0] * num_perms
    
    # Helper to get ID
    def get_id(t):
        return perm_to_id.get(t, -1)
        
    # Initial count pass
    for j in range(L - n + 1):
        t = tuple(word[j:j+n])
        id_val = get_id(t)
        if id_val != -1:
            counts[id_val] += 1
            
    # Greedy Deletion Loop
    # We scan left to right. If we delete, we stay at the same index.
    # If we can't delete, we move right.
    # We repeat passes until no changes or time out.
    
    while time.time() < deadline:
        changed = False
        L = len(word)
        i = 0
        while i < L:
            if time.time() > deadline - 2:
                break
                
            # Check safety at i
            # Affected windows: j in [max(0, i-n+1), min(i, L-n)]
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            
            safe = True
            # We need to check counts for these windows
            # And store them for update if safe
            affected_ids = []
            for j in range(start_min, start_max + 1):
                t = tuple(word[j:j+n])
                id_val = get_id(t)
                if id_val != -1:
                    affected_ids.append(id_val)
                    if counts[id_val] == 1:
                        safe = False
                        break
            
            if safe:
                # Perform deletion
                # 1. Decrement counts for affected windows
                for id_val in affected_ids:
                    counts[id_val] -= 1
                    
                # 2. Remove character
                word.pop(i)
                L -= 1
                changed = True
                
                # 3. Increment counts for new windows
                # New windows span the gap at i
                # New L is L
                # New start_max for windows containing i: min(i, L-n)
                # New start_min: max(0, i-n+1)
                new_start_min = max(0, i - n + 1)
                new_start_max = min(i, L - n)
                
                for j in range(new_start_min, new_start_max + 1):
                    t = tuple(word[j:j+n])
                    id_val = get_id(t)
                    if id_val != -1:
                        counts[id_val] += 1
                
                # Do not increment i, re-check new character at i
            else:
                i += 1
                
        if not changed:
            break
            
    return "".join(map(str, word))
