import itertools

def construct(n):
    if n == 1:
        return "1"
    
    perms = sorted(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map prefix to list of indices
    prefix_to_indices = {}
    for i, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_to_indices:
                prefix_to_indices[prefix] = []
            prefix_to_indices[prefix].append(i)
            
    # Also need prefix of length 0? No, k goes down to 1.
    # If k=0, we pick any.
    
    visited = [False] * N
    current_idx = 0
    visited[0] = True
    current_word = list(perms[0])
    
    # Initialize last used
    last_used = [0] * (n + 1)
    for i, sym in enumerate(perms[0]):
        last_used[sym] = i + 1 # 1-based time for initialization
    
    time = n
    
    for _ in range(N - 1):
        found = False
        # Try max overlap first
        for k in range(n - 1, 0, -1):
            suffix = tuple(current_word[-k:])
            candidates = prefix_to_indices.get(suffix, [])
            
            if not candidates:
                continue
                
            # Filter unvisited
            unvisited_candidates = [idx for idx in candidates if not visited[idx]]
            
            if not unvisited_candidates:
                continue
                
            # Select best candidate based on LRU of the first new symbol
            # The first new symbol is perms[idx][k]
            best_idx = -1
            min_last_used = float('inf')
            
            for idx in unvisited_candidates:
                first_new_sym = perms[idx][k]
                if last_used[first_new_sym] < min_last_used:
                    min_last_used = last_used[first_new_sym]
                    best_idx = idx
                    
            if best_idx != -1:
                visited[best_idx] = True
                next_perm = perms[best_idx]
                new_syms = next_perm[k:]
                current_word.extend(new_syms)
                
                # Update last_used for appended symbols
                for sym in new_syms:
                    time += 1
                    last_used[sym] = time
                    
                current_idx = best_idx
                found = True
                break
        
        if not found:
            # Fallback: pick unvisited perm with LRU first symbol?
            # Or just lexicographically smallest?
            # To be safe and valid, we must pick an unvisited perm.
            # Let's try to pick the one with the least recently used first symbol.
            best_idx = -1
            min_last_used = float('inf')
            for i in range(N):
                if not visited[i]:
                    first_sym = perms[i][0]
                    if last_used[first_sym] < min_last_used:
                        min_last_used = last_used[first_sym]
                        best_idx = i
                        
            if best_idx != -1:
                visited[best_idx] = True
                next_perm = perms[best_idx]
                current_word.extend(next_perm)
                for sym in next_perm:
                    time += 1
                    last_used[sym] = time
                current_idx = best_idx
            else:
                # Should not happen
                break

    return "".join(map(str, current_word))
