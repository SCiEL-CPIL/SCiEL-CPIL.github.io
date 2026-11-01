import time
import itertools

def construct(n):
    start_time = time.time()
    # Deadline guard
    deadline = start_time + 140.0
    
    if n == 1:
        return "1"
        
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Initialize prefix maps
    # prefix_maps[k] maps prefix of length k to set of indices
    prefix_maps = [{} for _ in range(n)] # index k-1 for length k
    
    for i, p in enumerate(perms):
        for k in range(1, n):
            pref = p[:k]
            if pref not in prefix_maps[k-1]:
                prefix_maps[k-1][pref] = set()
            prefix_maps[k-1][pref].add(i)
            
    remaining = set(range(N))
    word = [1]
    
    while remaining:
        if time.time() > deadline:
            # Should not happen, but safety
            break
            
        chosen_idx = None
        best_k = 0
        
        # Find best overlap
        # Check k from n-1 down to 1
        for k in range(n - 1, 0, -1):
            if len(word) < k:
                continue
            suffix = tuple(word[-k:])
            candidates = prefix_maps[k-1].get(suffix)
            if candidates:
                # Pick an arbitrary candidate
                chosen_idx = next(iter(candidates))
                best_k = k
                break
                
        if chosen_idx is None:
            # No overlap, pick any remaining
            chosen_idx = next(iter(remaining))
            best_k = 0
            
        # Append
        p = perms[chosen_idx]
        word.extend(p[best_k:])
        
        # Remove chosen
        remaining.remove(chosen_idx)
        for k in range(1, n):
            pref = p[:k]
            if pref in prefix_maps[k-1]:
                prefix_maps[k-1][pref].discard(chosen_idx)
                if not prefix_maps[k-1][pref]:
                    del prefix_maps[k-1][pref]
                    
        # Check newly covered permutations
        new_len = n - best_k
        # Windows are at the end
        # The new windows are those ending at the last new_len positions
        # Actually, any window of length n that includes at least one new character
        # The new characters are at indices len(word)-new_len to len(word)-1
        # So windows starting at i where i+n > len(word)-new_len
        # i > len(word) - new_len - n
        # i >= len(word) - new_len - n + 1
        
        start_check = max(0, len(word) - new_len - n + 1)
        for i in range(start_check, len(word) - n + 1):
            w = tuple(word[i:i+n])
            idx = perm_to_idx.get(w)
            if idx is not None and idx in remaining:
                remaining.remove(idx)
                p_remove = perms[idx]
                for k in range(1, n):
                    pref = p_remove[:k]
                    if pref in prefix_maps[k-1]:
                        prefix_maps[k-1][pref].discard(idx)
                        if not prefix_maps[k-1][pref]:
                            del prefix_maps[k-1][pref]
                            
    return "".join(map(str, word))
