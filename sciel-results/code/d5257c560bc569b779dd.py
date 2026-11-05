import itertools
import time
import random

def construct(n):
    if n < 7:
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    start_time = time.time()
    deadline = start_time + 110
    
    symbols = list(range(1, n))
    perms = list(itertools.permutations(symbols))
    m = len(perms)
    
    # Cache for overlaps
    overlap_cache = {}
    
    def get_overlap(pi_idx, sigma_idx):
        key = (pi_idx, sigma_idx)
        if key in overlap_cache:
            return overlap_cache[key]
        
        pi = perms[pi_idx]
        sigma = perms[sigma_idx]
        max_k = 0
        for k in range(n - 1, 0, -1):
            if pi[-k:] == sigma[:k]:
                max_k = k
                break
        overlap_cache[key] = max_k
        return max_k

    # 1. Get standard path (lexicographic order of perms? No, standard construction order)
    # The standard construction order is the order in which permutations appear in S_{n-1}.
    # We can generate S_{n-1} using the recursive method to get the "standard" order.
    # Or we can just use the lexicographic order of perms and see if it's close.
    # Actually, the recursive construction for S_{n-1} produces permutations in a specific order.
    # Let's generate the standard word for n-1 and extract the order.
    
    def get_standard_order(n_val):
        word = [1]
        for m_val in range(2, n_val + 1):
            perms_list, seen = [], set()
            for i in range(len(word) - m_val + 2):
                window = tuple(word[i:i + m_val - 1])
                if len(set(window)) == m_val - 1 and window not in seen:
                    seen.add(window)
                    perms_list.append(window)
            new = []
            for p in perms_list:
                block = list(p) + [m_val] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        
        # Extract order of permutations of 1..n_val-1
        # The permutations of 1..n_val-1 appear as windows of length n_val-1 in word
        # But they appear multiple times.
        # In the standard construction, each permutation of 1..n_val-1 appears exactly once as a "block center" before n_val is inserted?
        # No, the permutations of 1..n_val-1 are the "previous word" permutations.
        # The list `perms_list` in the last iteration of the loop (where m_val = n_val) contains the permutations of 1..n_val-1 in the order they are used.
        # So we just need to return `perms_list` from the last iteration.
        
        # Let's re-run to capture it
        word = [1]
        last_perms = []
        for m_val in range(2, n_val + 1):
            perms_list, seen = [], set()
            for i in range(len(word) - m_val + 2):
                window = tuple(word[i:i + m_val - 1])
                if len(set(window)) == m_val - 1 and window not in seen:
                    seen.add(window)
                    perms_list.append(window)
            new = []
            for p in perms_list:
                block = list(p) + [m_val] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
            if m_val == n_val:
                last_perms = perms_list
        return last_perms

    std_perms_list = get_standard_order(n - 1)
    # Map std_perms_list to indices in `perms`
    perm_index = {p: i for i, p in enumerate(perms)}
    std_path = [perm_index[p] for p in std_perms_list]
    
    # Calculate score for std_path
    std_score = 0
    for i in range(len(std_path) - 1):
        std_score += get_overlap(std_path[i], std_path[i+1])
        
    best_path = std_path
    best_score = std_score
    
    # Try greedy improvements
    # We can try to improve the standard path using 2-opt or greedy restarts.
    # Let's do greedy restarts from random starts.
    
    for run in range(50): # Limit restarts
        if time.time() > deadline:
            break
        
        unvisited = set(range(m))
        start_idx = random.choice(list(unvisited))
        unvisited.remove(start_idx)
        
        path = [start_idx]
        current = start_idx
        
        while unvisited:
            best_next = -1
            best_overlap = -1
            # To speed up, we can pre-sort unvisited by overlap with current?
            # But overlap changes as we go? No, overlap(current, sigma) is fixed.
            # So we can just scan.
            for sigma_idx in unvisited:
                overlap = get_overlap(current, sigma_idx)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_next = sigma_idx
            
            if best_next == -1:
                break
            
            path.append(best_next)
            unvisited.remove(best_next)
            current = best_next
            
        score = 0
        for i in range(len(path) - 1):
            score += get_overlap(path[i], path[i+1])
            
        if score > best_score:
            best_score = score
            best_path = path[:]
            
    # Construct final word
    word = []
    for i, idx in enumerate(best_path):
        p = perms[idx]
        block = list(p) + [n] + list(p)
        if i == 0:
            word.extend(block)
        else:
            prev_idx = best_path[i-1]
            overlap = get_overlap(prev_idx, idx)
            word.extend(block[overlap:])
            
    return "".join(map(str, word))
