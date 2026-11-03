import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate standard superpermutation
    # This is the same as the seed program
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
            max_overlap = min(len(new), len(block))
            for k in range(max_overlap, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if new[len(new) - k:] == block[:k]:
                    overlap = k
                    break
            new += block[overlap:]
        word = new
        
    # If n is small, return immediately as standard is optimal
    if n <= 6:
        return "".join(map(str, word))
        
    # 2. Prepare for local search (greedy deletion)
    # Map permutations to indices
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    counts = [0] * num_perms
    
    def update_counts(w):
        for i in range(len(w) - n + 1):
            t = tuple(w[i:i + n])
            if t in perm_to_idx:
                counts[perm_to_idx[t]] += 1
            else:
                # This shouldn't happen for a valid superpermutation
                # but if it does, it's not a permutation
                pass
                
    update_counts(word)
    
    # 3. Greedy Deletion
    # We try to remove characters if it doesn't reduce any count to 0
    # We repeat until no more deletions are possible
    
    # To speed up, we can keep a list of indices to check
    # But simple restart is fine for n=7
    
    while True:
        if time.time() > deadline:
            break
            
        improved = False
        L = len(word)
        
        for i in range(L):
            # Check if word[i] can be removed
            # Affected windows start at j in [max(0, i-n+1), i]
            start_j = max(0, i - n + 1)
            end_j = i
            
            can_remove = True
            affected_indices = []
            
            for j in range(start_j, end_j + 1):
                # Ensure j is a valid start index for a window of length n
                # In the current word of length L, valid j are 0..L-n
                if j > L - n:
                    continue
                    
                t = tuple(word[j:j + n])
                if t in perm_to_idx:
                    idx = perm_to_idx[t]
                    if counts[idx] == 1:
                        can_remove = False
                        break
                    else:
                        affected_indices.append(idx)
                else:
                    # Not a permutation, ignore
                    pass
            
            if can_remove:
                # Perform removal
                # Decrement counts
                for idx in affected_indices:
                    counts[idx] -= 1
                
                # Remove from word
                word.pop(i)
                improved = True
                break # Restart scan from 0
        
        if not improved:
            break
            
    return "".join(map(str, word))
