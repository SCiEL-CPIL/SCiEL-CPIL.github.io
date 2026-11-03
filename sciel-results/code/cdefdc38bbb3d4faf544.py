import time

def construct(n_rows, k):
    # n_rows=4, k=7 (8 marks)
    
    # Step 1: Generate all valid rulers of order k+1 with max value up to a limit
    # Limit can be dynamic or fixed. Let's fix to 160 to be safe.
    max_limit = 160
    
    # Generate rulers
    # A ruler is a tuple of length k+1, starting with 0.
    # We use DFS to generate all valid rulers.
    
    all_rulers = []
    
    def generate_rulers(M_limit):
        row = [0]
        used = set()
        
        def dfs(idx):
            if idx == k + 1:
                all_rulers.append(tuple(row))
                return
                
            # Pruning: 
            # The next mark x must be > row[-1] and <= M_limit
            # Also, the remaining marks must fit.
            
            for x in range(row[-1] + 1, M_limit + 1):
                # Check differences
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                    
                row.append(x)
                for d in new_diffs:
                    used.add(d)
                    
                dfs(idx + 1)
                
                for d in new_diffs:
                    used.remove(d)
                row.pop()
                
        dfs(1)

    generate_rulers(max_limit)
    
    # Step 2: Select n_rows disjoint rulers minimizing the max value
    
    # Sort rulers by max value
    all_rulers.sort(key=lambda r: r[-1])
    
    # We want to find a set of n_rows disjoint rulers.
    # We can iterate M from lower_bound to max_limit.
    lower_bound = n_rows * (k * (k + 1) // 2)
    
    # To speed up selection, we can group rulers by max value.
    # Or just filter.
    
    # Backtracking selection
    def select_rulers(M_threshold, start_idx):
        # We need to pick n_rows rulers from all_rulers[start_idx:] 
        # such that their max value <= M_threshold and they are disjoint.
        # We use a set for used differences.
        used = set()
        chosen = []
        
        def dfs_selection(idx):
            if len(chosen) == n_rows:
                return True
                
            # Try each ruler from start_idx onwards?
            # This is inefficient. 
            # Better: iterate through the list of valid rulers (max <= M_threshold)
            # and pick one.
            
            # Optimization: We can pre-filter rulers with max <= M_threshold
            # But we are calling this inside the loop.
            
            # Let's pass the list of candidate indices.
            
            # Actually, let's restructure:
            # For a given M, get list of valid rulers.
            # Then backtrack to pick n_rows disjoint ones.
            
            # This function is called for a specific M.
            
            # We need to avoid re-using rulers.
            # Since the list is small, we can just iterate.
            
            # To avoid symmetry (order of rows doesn't matter), we can enforce an order?
            # No, the rulers are distinct.
            
            # Simple approach:
            # Iterate i from 0 to len(candidates)
            # Pick candidates[i]
            # Recurse with candidates[i+1:] to avoid picking the same ruler twice?
            # Yes, because rows are indistinguishable in the final output?
            # The output is a list of rows. The order of rows doesn't matter for validity.
            # So we can enforce that we pick rulers in increasing index order.
            
            for i in range(idx, len(candidates)):
                r = candidates[i]
                diffs = set()
                valid = True
                for j in range(len(r)):
                    for l in range(j+1, len(r)):
                        d = r[l] - r[j]
                        if d in used:
                            valid = False
                            break
                        diffs.add(d)
                    if not valid:
                        break
                        
                if not valid:
                    continue
                
                # Pick this ruler
                chosen.append(r)
                used.update(diffs)
                
                if dfs_selection(i + 1):
                    return True
                    
                # Backtrack
                chosen.pop()
                used.difference_update(diffs)
                
            return False
            
        candidates = [r for r in all_rulers if r[-1] <= M_threshold]
        
        if dfs_selection(0):
            return chosen
            
        return None

    # Iterate M
    start_time = time.time()
    
    for M in range(lower_bound, max_limit + 1):
        if time.time() - start_time > 100:
            break
            
        result = select_rulers(M, 0)
        if result:
            # Convert tuples to lists
            return [list(r) for r in result]
            
    # Fallback to greedy if no solution found
    def greedy_construct():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    return greedy_construct()
