import itertools
import sys

def construct(n):
    if n == 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map from suffix to list of indices
    # We need to check overlaps from n-1 down to 0
    # For efficiency, we can precompute or just search.
    # Since n=7, N=5040, it's small enough to just filter if needed, 
    # but a dict is better.
    
    # We will use a dictionary mapping tuple(suffix) to list of indices
    # But we need to check multiple suffix lengths.
    # It's easier to just iterate candidates and check prefix match.
    # Or precompute for each k.
    
    # Let's precompute prefix_to_indices for all k from 1 to n-1
    # Actually, we need suffix match.
    # current_word ends with S. We want perms starting with S.
    # So we need a map: prefix -> indices.
    
    prefix_map = {}
    for i, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_map:
                prefix_map[prefix] = []
            prefix_map[prefix].append(i)
            
    visited = [False] * N
    current_word = list(perms[0])
    visited[0] = True
    
    # Recency tracking
    # last_seen[symbol] = index in current_word
    last_seen = [0] * (n + 1)
    for idx, sym in enumerate(current_word):
        last_seen[sym] = idx
        
    current_len = len(current_word)
    
    for _ in range(N - 1):
        found = False
        # Try max overlap first
        # Overlap k means we take suffix of length k from current_word
        # and it must be prefix of length k of next_perm
        # New symbols added: n - k
        
        for k in range(n - 1, -1, -1):
            if k == 0:
                suffix = ()
            else:
                suffix = tuple(current_word[-k:])
                
            candidates = prefix_map.get(suffix, [])
            
            # Find unvisited candidates
            unvisited_cands = [idx for idx in candidates if not visited[idx]]
            
            if not unvisited_cands:
                continue
                
            # Select best candidate based on LRU heuristic
            # The new symbols are perms[idx][k:]
            # We want to minimize the recency of the first new symbol?
            # Or the sum of recencies?
            # "greedily appending the least recently used symbol"
            # This suggests the first new symbol should be the LRU.
            
            best_idx = -1
            best_score = float('inf')
            
            for idx in unvisited_cands:
                new_syms = perms[idx][k:]
                if not new_syms:
                    # This shouldn't happen if k < n
                    continue
                    
                # Heuristic: Minimize last_seen of the first new symbol
                # If tie, minimize last_seen of the second, etc.
                # To make it a single score, we can use a tuple of last_seen values
                # and find the min.
                # But we want the *least* recently used, so smallest last_seen value.
                
                score_tuple = tuple(last_seen[s] for s in new_syms)
                
                if best_idx == -1 or score_tuple < best_score:
                    best_score = score_tuple
                    best_idx = idx
                    
            if best_idx != -1:
                visited[best_idx] = True
                next_perm = perms[best_idx]
                new_syms = next_perm[k:]
                current_word.extend(new_syms)
                
                # Update last_seen
                for sym in new_syms:
                    last_seen[sym] = current_len
                    current_len += 1
                    
                found = True
                break
                
        if not found:
            # Fallback: pick any unvisited
            for i in range(N):
                if not visited[i]:
                    visited[i] = True
                    next_perm = perms[i]
                    current_word.extend(next_perm)
                    for sym in next_perm:
                        last_seen[sym] = current_len
                        current_len += 1
                    found = True
                    break
                    
    return "".join(map(str, current_word))
