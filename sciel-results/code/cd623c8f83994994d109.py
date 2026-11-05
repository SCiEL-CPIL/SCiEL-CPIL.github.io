import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    if n < 7:
        return standard_construct(n)
        
    if n > 7:
        # Fallback to standard for larger n, though unlikely
        return standard_construct(n)

    # n == 7
    # 1. Get standard order of perms of 1..6
    perms_6 = get_standard_perm_order(6)
    
    # 2. Precompute overlap costs
    # Cost[i][j] = 13 - overlap(perms_6[i], perms_6[j])
    # Actually, we want to maximize overlap.
    # Let's store overlap matrix O.
    N = len(perms_6)
    O = [[0]*N for _ in range(N)]
    for i in range(N):
        pi = perms_6[i]
        for j in range(N):
            pj = perms_6[j]
            # Calculate overlap
            # Suffix of pi vs Prefix of pj
            max_ov = 0
            # pi is tuple of ints
            for k in range(1, 7): # 1 to 6
                if pi[6-k:] == pj[:k]:
                    max_ov = k
            O[i][j] = max_ov
    
    # 3. Local Search
    # Start with standard order
    tour = list(range(N))
    
    # 2-opt
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(N-1):
            for j in range(i+1, N):
                # Check swap i..j
                # Current cost part: C[i-1][i] + C[j][j+1] (with bounds check)
                # New cost part: C[i-1][j] + C[i][j+1]
                
                # Indices in tour
                # Note: tour is a list of indices into perms_6
                # We are swapping the segment tour[i:j+1]
                
                # Let's use indices in the tour list
                # A = tour[i-1] (if i>0 else None)
                # B = tour[i]
                # C = tour[j]
                # D = tour[j+1] (if j<N-1 else None)
                
                # Overlap sum delta
                old_ov = 0
                new_ov = 0
                
                if i > 0:
                    old_ov += O[tour[i-1]][tour[i]]
                if j < N-1:
                    old_ov += O[tour[j]][tour[j+1]]
                    
                if i > 0:
                    new_ov += O[tour[i-1]][tour[j]]
                if j < N-1:
                    new_ov += O[tour[i]][tour[j+1]]
                    
                if new_ov > old_ov:
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
                    # Restart loop? Or continue?
                    # Usually restart or just continue.
                    # Continuing is faster.
        
    # 4. Construct final word
    word = []
    # Add first block
    p = perms_6[tour[0]]
    block = list(p) + [7] + list(p)
    word = block
    
    for idx in tour[1:]:
        p = perms_6[idx]
        block = list(p) + [7] + list(p)
        # Find overlap
        # Suffix of word vs Prefix of block
        max_ov = 0
        # word is list, block is list
        # Max possible overlap is 6 (length of p)
        # Actually, block starts with p.
        # So we check suffix of word against prefix of block.
        for k in range(min(len(word), len(block)), -1, -1):
            if k == 0:
                max_ov = 0
                break
            if word[-k:] == block[:k]:
                max_ov = k
                break
        word += block[max_ov:]
        
    return "".join(map(str, word))

def standard_construct(n):
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
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def get_standard_perm_order(n):
    # Returns list of tuples representing perms of 1..n in standard order
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
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    # Extract perms of 1..n from word
    perms, seen = [], set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n and window not in seen:
            seen.add(window)
            perms.append(window)
    return perms
