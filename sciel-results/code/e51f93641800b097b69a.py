def construct(n):
    if n == 1:
        return "1"
    
    seen_windows = set()
    current = list(range(1, n + 1))
    seen_windows.add(tuple(current))
    
    total_perms = 1
    for i in range(2, n + 1):
        total_perms *= i
    
    last_used = {i: i - 1 for i in range(1, n + 1)}
    
    while len(seen_windows) < total_perms:
        suffix = tuple(current[-(n-1):])
        
        candidates = []
        for sym in range(1, n + 1):
            new_window = suffix + (sym,)
            if new_window not in seen_windows:
                candidates.append((last_used[sym], sym))
        
        if not candidates:
            break
        
        candidates.sort()
        best_sym = candidates[0][1]
        
        current.append(best_sym)
        seen_windows.add(tuple(current[-n:]))
        last_used[best_sym] = len(current) - 1
    
    return "".join(map(str, current))
