import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110
    
    if n == 1:
        return "1"
    
    # Build S_{n-1}
    def build_superperm_recursive(n_val):
        if n_val == 1:
            return [1]
        word = [1]
        for m in range(2, n_val + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    base_word = build_superperm_recursive(n - 1)
    base_len = len(base_word)
    
    # Extract permutations of 1..n-1 from base_word
    perms, seen = [], set()
    for i in range(base_len - (n - 2)):
        window = tuple(base_word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    m = len(perms)
    if n == 2:
        return "121" # 1, 2, 1. Perms: 12, 21.
        
    # Precompute overlap matrix
    # perms[i] is a tuple of length n-1
    # O[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # Max k is n-2 (since if k=n-1, perms[i]==perms[j])
    # Actually, if perms[i] != perms[j], max overlap is n-2.
    # Example: 123 and 231. Suffix 23, Prefix 23. Overlap 2.
    # 123 and 124 (not a perm).
    # For perms of 1..6, max overlap is 5.
    
    O = [[0] * m for _ in range(m)]
    for i in range(m):
        pi = perms[i]
        for j in range(m):
            if i == j:
                O[i][j] = n - 1
                continue
            pj = perms[j]
            # Check overlaps from n-2 down to 0
            max_ov = n - 2
            ov = 0
            for k in range(max_ov, -1, -1):
                if pi[-k:] == pj[:k]:
                    ov = k
                    break
            O[i][j] = ov
            
    # Start overlap
    # The base_word ends with some suffix.
    # We need overlap between base_word and perms[j].
    # base_word is a list of ints.
    # We can just check suffix of base_word.
    # Since base_word is long, we only need the last n-1 elements.
    base_suffix = tuple(base_word[-(n-1):])
    O_start = [0] * m
    for j in range(m):
        pj = perms[j]
        max_ov = n - 1
        ov = 0
        for k in range(max_ov, -1, -1):
            if base_suffix[-k:] == pj[:k]:
                ov = k
                break
        O_start[j] = ov
        
    # Initial order: 0 to m-1
    order = list(range(m))
    
    def calc_score(ord):
        total = O_start[ord[0]]
        for k in range(1, m):
            total += O[ord[k-1]][ord[k]]
        return total
        
    current_score = calc_score(order)
    
    # 2-Opt Local Search
    # We want to maximize score.
    # Time limit check
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        for i in range(m - 1):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, m):
                # Calculate delta for reversing order[i..j]
                # Old internal sum: sum_{k=i}^{j-1} O[ord[k]][ord[k+1]]
                # New internal sum: sum_{k=i}^{j-1} O[ord[k+1]][ord[k]]
                # Old boundary: 
                #   left: O_start[ord[i]] if i==0 else O[ord[i-1]][ord[i]]
                #   right: O[ord[j]][ord[j+1]] if j < m-1 else 0
                # New boundary:
                #   left: O_start[ord[j]] if i==0 else O[ord[i-1]][ord[j]]
                #   right: O[ord[i]][ord[j+1]] if j < m-1 else 0
                
                old_int = 0
                new_int = 0
                for k in range(i, j):
                    old_int += O[order[k]][order[k+1]]
                    new_int += O[order[k+1]][order[k]]
                    
                old_left = O_start[order[i]] if i == 0 else O[order[i-1]][order[i]]
                new_left = O_start[order[j]] if i == 0 else O[order[i-1]][order[j]]
                
                old_right = O[order[j]][order[j+1]] if j < m - 1 else 0
                new_right = O[order[i]][order[j+1]] if j < m - 1 else 0
                
                delta = (new_int - old_int) + (new_left - old_left) + (new_right - old_right)
                
                if delta > 0:
                    order[i:j+1] = reversed(order[i:j+1])
                    current_score += delta
                    improved = True
                    
    # Construct final word
    final_word = list(base_word)
    
    # Add first block
    first_idx = order[0]
    first_block = list(perms[first_idx]) + [n] + list(perms[first_idx])
    ov = O_start[first_idx]
    final_word += first_block[ov:]
    
    # Add remaining blocks
    for k in range(1, m):
        prev_idx = order[k-1]
        curr_idx = order[k]
        curr_block = list(perms[curr_idx]) + [n] + list(perms[curr_idx])
        ov = O[prev_idx][curr_idx]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))
