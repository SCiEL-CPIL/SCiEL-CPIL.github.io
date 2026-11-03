import itertools
import time

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Generate all permutations
    perms = sorted(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Build prefix index for quick lookup
    prefix_to_indices = {}
    for i, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_to_indices:
                prefix_to_indices[prefix] = []
            prefix_to_indices[prefix].append(i)
    
    # Also build suffix-to-permutation map for overlap checking
    # suffix of length k maps to the permutation that has that suffix
    # Actually, we need: given a suffix of the current word, find perms starting with that suffix
    # That's exactly prefix_to_indices
    
    # Strategy: Use the classic recursive construction as initial solution,
    # then try to improve via local search (swaps, insertions)
    
    # First: classic recursive construction
    def recursive_superperm(n):
        if n == 1:
            return [1]
        word = recursive_superperm(n - 1)
        # Find all distinct (n-1)-length windows in word
        windows = []
        seen = set()
        for i in range(len(word) - n + 2):
            w = tuple(word[i:i + n - 1])
            if len(set(w)) == n - 1 and w not in seen:
                seen.add(w)
                windows.append(w)
        new_word = []
        for w in windows:
            block = list(w) + [n] + list(w)
            # Find maximum overlap
            overlap = 0
            max_ov = min(len(new_word), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[-o:] == block[:o]:
                    overlap = o
                    break
            new_word.extend(block[overlap:])
        return new_word
    
    # Get initial word from recursive construction
    word = recursive_superperm(n)
    best_word = list(word)
    best_len = len(best_word)
    
    # Verify the word contains all permutations
    def verify(word_list):
        if len(word_list) < n:
            return False
        seen = set()
        for i in range(len(word_list) - n + 1):
            window = tuple(word_list[i:i + n])
            if len(set(window)) == n:  # Only count actual permutations
                seen.add(window)
        return len(seen) == N
    
    # Local search: try to remove redundant symbols
    # A symbol at position i can be removed if the word still contains all perms
    # But that's expensive. Instead, try 2-opt style moves.
    
    # Alternative: use greedy with look-ahead to build a better word
    # Let's try multiple greedy strategies and keep the best
    
    def greedy_with_lookahead(perms, prefix_to_indices, n, N, start_perm_idx=0):
        visited = [False] * N
        current_idx = start_perm_idx
        visited[start_perm_idx] = True
        current_word = list(perms[start_perm_idx])
        
        for step in range(N - 1):
            found = False
            # Try max overlap first
            for k in range(n - 1, 0, -1):
                suffix = tuple(current_word[-k:])
                candidates = prefix_to_indices.get(suffix, [])
                
                if not candidates:
                    continue
                
                # Score candidates by look-ahead
                best_idx = -1
                best_score = -1
                best_next_sym = n + 1
                
                for idx in candidates:
                    if not visited[idx]:
                        next_perm = perms[idx]
                        # Look-ahead: how many unvisited perms start with suffix of next_perm?
                        next_suffix = next_perm[n - 1:]
                        next_cands = prefix_to_indices.get(next_suffix, [])
                        unvisited_next = sum(1 for i in next_cands if not visited[i])
                        
                        # Prefer: high unvisited_next (continuity), low next symbol
                        score = unvisited_next * 10 - next_perm[k]
                        
                        if score > best_score or (score == best_score and next_perm[k] < best_next_sym):
                            best_score = score
                            best_idx = idx
                            best_next_sym = next_perm[k]
                
                if best_idx != -1:
                    visited[best_idx] = True
                    next_perm = perms[best_idx]
                    current_word.extend(next_perm[k:])
                    current_idx = best_idx
                    found = True
                    break
            
            if not found:
                # No overlap, pick lexicographically smallest unvisited
                for i in range(N):
                    if not visited[i]:
                        visited[i] = True
                        next_perm = perms[i]
                        current_word.extend(next_perm)
                        current_idx = i
                        found = True
                        break
        
        return current_word
    
    # Try greedy from different starting permutations
    for start_idx in range(min(10, N)):
        if time.time() > deadline:
            break
        g_word = greedy_with_lookahead(perms, prefix_to_indices, n, N, start_idx)
        if verify(g_word) and len(g_word) < best_len:
            best_word = g_word
            best_len = len(g_word)
    
    # Try the recursive construction result
    if verify(word) and len(word) < best_len:
        best_word = word
        best_len = len(word)
    
    # Local improvement: try to find and remove unnecessary symbols
    # For each position, check if removing it still gives a valid superpermutation
    # This is O(n * N) per removal attempt, but we can be smart about it
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(1, len(best_word) - 1):  # Don't remove first or last
            if time.time() > deadline:
                break
            # Try removing position i
            test_word = best_word[:i] + best_word[i + 1:]
            if verify(test_word):
                best_word = test_word
                best_len = len(best_word)
                improved = True
                break  # Restart from beginning after removal
    
    return "".join(map(str, best_word))
