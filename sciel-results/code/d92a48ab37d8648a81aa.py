import itertools

def construct(n):
    if n == 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map from prefix to list of indices of permutations starting with that prefix
    prefix_to_indices = {}
    for i, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_to_indices:
                prefix_to_indices[prefix] = []
            prefix_to_indices[prefix].append(i)
            
    # Also map permutation tuple to index for quick lookup if needed?
    # We use a set of visited tuples.
    visited = set()
    
    current = perms[0]
    visited.add(current)
    
    word = list(current)
    current_len = len(word)
    
    # LRU tracking
    last_seen = {}
    for i, sym in enumerate(current):
        last_seen[sym] = i
        
    for _ in range(N - 1):
        # 1. Try to find unvisited neighbor (overlap n-1)
        # Neighbors are perms p' such that p' = current[1:] + (x,)
        # x must not be in current[1:]
        
        best_x = None
        best_perm = None
        best_score = float('inf')
        
        for x in range(1, n + 1):
            if x not in current[1:]:
                next_perm = current[1:] + (x,)
                if next_perm not in visited:
                    score = last_seen.get(x, -1)
                    if score < best_score or (score == best_score and (best_x is None or x < best_x)):
                        best_score = score
                        best_x = x
                        best_perm = next_perm
                        
        if best_perm is not None:
            # Use the LRU choice
            word.append(best_x)
            current_len += 1
            last_seen[best_x] = current_len - 1
            visited.add(best_perm)
            current = best_perm
        else:
            # Fallback: Find unvisited perm with max overlap
            found = False
            for k in range(n - 2, 0, -1):
                suffix = tuple(current[-k:])
                candidates = prefix_to_indices.get(suffix, [])
                
                next_idx = -1
                # We want the one with max overlap, which is k.
                # Among them, pick the one that is "best" by some heuristic.
                # Seed picked lexicographically smallest.
                # We can pick the one whose first new symbol (at index k) is LRU?
                # Or just lexicographically smallest to be safe/fast.
                
                for idx in candidates:
                    if perms[idx] not in visited:
                        next_idx = idx
                        break
                        
                if next_idx != -1:
                    next_perm = perms[next_idx]
                    # Append new symbols
                    new_symbols = next_perm[k:]
                    for sym in new_symbols:
                        word.append(sym)
                        last_seen[sym] = len(word) - 1
                    visited.add(next_perm)
                    current = next_perm
                    found = True
                    break
                    
            if not found:
                # No overlap > 0, pick any unvisited
                for i in range(N):
                    if perms[i] not in visited:
                        next_perm = perms[i]
                        for sym in next_perm:
                            word.append(sym)
                            last_seen[sym] = len(word) - 1
                        visited.add(next_perm)
                        current = next_perm
                        found = True
                        break
                        
    return "".join(map(str, word))
