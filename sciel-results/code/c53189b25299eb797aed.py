import itertools

def construct(n):
    if n == 1:
        return "1"
    
    total_perms = 1
    for i in range(2, n + 1):
        total_perms *= i
        
    # LRU Heuristic
    current_word = list(range(1, n + 1))
    
    # Track all unique windows of length n to avoid duplicates
    all_windows_seen = {tuple(current_word)}
    
    # Track which permutations have been found
    perms_seen = set()
    if len(current_word) == n:
        perms_seen.add(tuple(current_word))
        
    last_used = {i: i for i in range(1, n + 1)}
    
    # Safety limit
    max_steps = total_perms * 20 
    
    steps = 0
    while len(perms_seen) < total_perms:
        if steps > max_steps:
            break
            
        suffix = current_word[-(n-1):]
        
        candidates = []
        for s in range(1, n + 1):
            new_window = tuple(suffix + [s])
            
            if new_window not in all_windows_seen:
                # Valid move (no duplicate window)
                candidates.append((last_used[s], s, new_window))
                
        if not candidates:
            # Stuck
            break
        
        # Sort by last_used (ascending), then by symbol (ascending)
        candidates.sort(key=lambda x: (x[0], x[1]))
        
        best_s = candidates[0][1]
        best_window = candidates[0][2]
        
        current_word.append(best_s)
        all_windows_seen.add(best_window)
        
        # Check if it's a permutation
        # A window is a permutation if it contains all distinct symbols 1..n
        # Since n is small, we can check len(set) == n and min==1, max==n
        # Or just sort and compare
        if len(set(best_window)) == n:
             perms_seen.add(best_window)
             
        last_used[best_s] = len(current_word) - 1
        steps += 1
        
    return "".join(map(str, current_word))
