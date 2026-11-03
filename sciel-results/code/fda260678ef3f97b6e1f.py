import time
import itertools

def construct(n):
    start_time = time.time()
    time_limit = 110.0
    
    # 1. Generate standard superpermutation
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
        
    if n <= 4:
        return "".join(map(str, word))

    all_perms = list(itertools.permutations(range(1, n+1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        # Recalculate counts
        counts = [0] * num_perms
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i+n])
            idx = perm_to_idx[p]
            counts[idx] += 1
            
        improved = False
        # Try to delete from end to start
        for i in range(len(word) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
                
            # Identify destroyed windows
            # s in [i-n+1, i]
            min_s = max(0, i - n + 1)
            max_s = min(i, len(word) - n)
            
            # Check validity
            valid = True
            # We need to check if counts[idx] > 1 for all destroyed windows?
            # No, if multiple destroyed windows are the same permutation, 
            # we need counts[idx] > number_of_times_destroyed.
            
            # Let's count how many times each perm is destroyed
            destroyed_counts = {}
            for s in range(min_s, max_s + 1):
                p = tuple(word[s:s+n])
                idx = perm_to_idx[p]
                destroyed_counts[idx] = destroyed_counts.get(idx, 0) + 1
                
            for idx, cnt in destroyed_counts.items():
                if counts[idx] <= cnt:
                    valid = False
                    break
            
            if valid:
                word.pop(i)
                improved = True
                break # Restart while loop
                
        if not improved:
            break
            
    return "".join(map(str, word))
