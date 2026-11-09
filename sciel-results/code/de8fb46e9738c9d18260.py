import random
import time
import itertools

def construct(n):
    if n <= 1:
        return "1"
    
    # Precompute all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    num_perms = len(perms)
    
    # Index permutations by prefixes
    # prefix_map[k] maps prefix tuple to set of indices of perms with that prefix
    # k ranges from 1 to n-1
    prefix_map = [dict() for _ in range(n)]
    for i, p in enumerate(perms):
        for k in range(1, n):
            pref = p[:k]
            if pref not in prefix_map[k-1]:
                prefix_map[k-1][pref] = set()
            prefix_map[k-1][pref].add(i)
            
    best_word = None
    best_len = float('inf')
    
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    while time.time() - start_time < timeout:
        visited = set()
        word = []
        
        # Start with a random permutation
        start_idx = random.randint(0, num_perms - 1)
        p = perms[start_idx]
        word.extend(p)
        visited.add(start_idx)
        
        current_len = n
        
        for _ in range(num_perms - 1):
            best_next_idx = -1
            best_overlap = 0
            best_next_next_overlap = 0
            found = False
            
            # Check overlaps from max to min
            for k in range(n - 1, 0, -1):
                if len(word) < k:
                    continue
                prefix = tuple(word[-k:])
                if prefix in prefix_map[k-1]:
                    candidates = prefix_map[k-1][prefix]
                    # Filter unvisited
                    # Optimization: if len(candidates) == 1, check that one
                    unvisited_candidates = []
                    for idx in candidates:
                        if idx not in visited:
                            unvisited_candidates.append(idx)
                    
                    if unvisited_candidates:
                        # Evaluate each candidate with lookahead
                        best_cand = None
                        best_score = -1
                        
                        # Limit candidates to check if too many (e.g., top 10 random)
                        # For k=1, there are 720. Checking all is 720 * 6 lookups = 4320 ops.
                        # 5040 steps * 4320 = 21M ops. Might be too slow in Python.
                        # Let's sample if len > 20.
                        if len(unvisited_candidates) > 20:
                            unvisited_candidates = random.sample(unvisited_candidates, 20)
                            
                        for idx in unvisited_candidates:
                            p_cand = perms[idx]
                            # Simulate next step overlap
                            next_max_overlap = 0
                            for k2 in range(n - 1, 0, -1):
                                pref2 = tuple(p_cand[-k2:])
                                if pref2 in prefix_map[k2-1]:
                                    # Check if any unvisited perm exists for this prefix
                                    # We don't check visited here for speed, just existence
                                    # This is an optimistic heuristic
                                    if prefix_map[k2-1][pref2]:
                                        next_max_overlap = k2
                                        break
                            
                            # Heuristic: maximize next_max_overlap
                            # Tie-break: random or min index
                            score = next_max_overlap
                            if score > best_score:
                                best_score = score
                                best_cand = idx
                                best_next_next_overlap = next_max_overlap
                                
                        if best_cand is not None:
                            best_next_idx = best_cand
                            best_overlap = k
                            found = True
                            break
            
            if not found:
                # No overlap, pick any unvisited
                # Use a precomputed list of indices?
                # Just scan
                for i in range(num_perms):
                    if i not in visited:
                        best_next_idx = i
                        best_overlap = 0
                        break
            
            p = perms[best_next_idx]
            visited.add(best_next_idx)
            
            # Append
            if best_overlap > 0:
                word.extend(p[best_overlap:])
                current_len += (n - best_overlap)
            else:
                word.extend(p)
                current_len += n
        
        # Check if this is better
        if current_len < best_len:
            best_len = current_len
            best_word = word[:]

    return "".join(map(str, best_word))
