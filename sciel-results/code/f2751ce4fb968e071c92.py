def construct():
    # 1. Generate smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    smooth.sort()
    n_smooth = len(smooth)
    
    # 2. Precompute best representation (min terms) for each sum 1..128
    best_rep = {}
    
    # We iterate by number of terms k
    # We use DFS to generate antichains of size k, pruning sums > 128
    def dfs(start_idx, k, current_sum, current_list):
        if k == 0:
            if 1 <= current_sum <= 128 and current_sum not in best_rep:
                best_rep[current_sum] = list(current_list)
            return
        
        for i in range(start_idx, n_smooth):
            x = smooth[i]
            if current_sum + x > 128:
                break
            
            # Check antichain property with current_list
            is_valid = True
            for y in current_list:
                if x % y == 0 or y % x == 0:
                    is_valid = False
                    break
            
            if is_valid:
                current_list.append(x)
                dfs(i + 1, k - 1, current_sum + x, current_list)
                current_list.pop()
    
    # Try k from 1 upwards
    for k in range(1, 11):
        if len(best_rep) == 128:
            break
        dfs(0, k, 0, [])
        
    # Check for missing
    missing = [n for n in range(1, 129) if n not in best_rep]
    if missing:
        # Extend search for missing numbers if necessary
        for k in range(11, 15):
            if not missing:
                break
            dfs(0, k, 0, [])
            
    # Final construction
    answers = []
    for n in range(1, 129):
        if n in best_rep:
            answers.append(best_rep[n])
        else:
            raise ValueError(f"Failed to find representation for {n}")
            
    return answers
