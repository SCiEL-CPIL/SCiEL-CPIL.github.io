import itertools

def construct(n_rows, k):
    C = (k + 1) * k // 2
    if C == 0:
        return [[0] for _ in range(n_rows)]
        
    M_min = n_rows * C
    
    def solve(M_val):
        def dfs(idx, prev_max, used_diffs):
            if idx == n_rows:
                return []
            
            # Pruning
            if hasattr(int, 'bit_count'):
                used_count = used_diffs.bit_count()
            else:
                used_count = bin(used_diffs).count('1')
            
            if M_val - used_count < (n_rows - idx) * C:
                return None

            start_L = max(prev_max, k)
            
            for L in range(start_L, M_val + 1):
                if L - 1 < k - 1:
                    continue
                
                # Generate combinations
                # Optimization: If k-1 is 0, combo is empty tuple
                if k - 1 == 0:
                    combos = [()]
                else:
                    combos = itertools.combinations(range(1, L), k - 1)
                    
                for combo in combos:
                    row = [0] + list(combo) + [L]
                    
                    diffs = 0
                    # Calculate diffs
                    # Unroll for small k? No, generic is fine.
                    for i in range(len(row)):
                        ri = row[i]
                        for j in range(i + 1, len(row)):
                            d = row[j] - ri
                            diffs |= (1 << d)
                    
                    if diffs & used_diffs:
                        continue
                    
                    res = dfs(idx + 1, L, used_diffs | diffs)
                    if res is not None:
                        return [row] + res
            
            return None

        return dfs(0, 0, 0)

    # Iterative deepening on M
    # Start from M_min
    # The gap between M_min and optimal is usually small.
    M = M_min
    while M < M_min + 100: # Cap search range for safety
        res = solve(M)
        if res is not None:
            return res
        M += 1
        
    # Fallback to greedy if exact search fails or is too slow
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
