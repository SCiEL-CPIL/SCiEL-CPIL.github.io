def find_max_sidon_exact(n):
    best_set = []
    
    def backtrack(current_set, start):
        nonlocal best_set
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
        if start >= n:
            return
            
        for i in range(start, n):
            new_set = current_set + [i]
            if is_sidon(new_set):
                backtrack(new_set, i + 1)
                
    backtrack([], 0)
    return best_set
