import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All differences within rows must be distinct across all rows.
    Minimize the largest entry.
    """
    # Heuristic fallback: Greedy
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    greedy_rows = greedy_construct()
    initial_max = max(max(r) for r in greedy_rows)
    
    # Lower bound
    min_possible_max = n_rows * (k * (k + 1) // 2)
    
    # If the problem is trivial or small, we can search.
    # For the specific task (7, 5), backtracking is very fast.
    
    def generate_rulers(max_val):
        rulers = []
        # Use bitmask for current row diffs
        # Max diff is max_val
        def backtrack_row(current_row, last_val, current_mask):
            if len(current_row) == k + 1:
                rulers.append((tuple(current_row), current_mask, current_row[-1]))
                return

            # Pruning: if we can't reach k+1 elements with remaining range?
            # Not strictly needed for small k.
            
            for x in range(last_val + 1, max_val + 1):
                new_mask = 0
                valid = True
                for a in current_row:
                    d = x - a
                    bit = 1 << d
                    if current_mask & bit:
                        valid = False
                        break
                    new_mask |= bit
                if valid:
                    backtrack_row(current_row + [x], x, current_mask | new_mask)

        backtrack_row([0], 0, 0)
        return rulers

    all_rulers = generate_rulers(initial_max)
    all_rulers.sort(key=lambda x: x[2])
    
    # Search for minimal M
    for M in range(min_possible_max, initial_max + 1):
        candidates = [r for r in all_rulers if r[2] <= M]
        if len(candidates) < n_rows:
            continue
        
        def find_disjoint_rulers(cands, n_needed):
            def backtrack(idx, used_mask, count, current_selection):
                if count == n_needed:
                    return current_selection
                
                if len(cands) - idx < n_needed - count:
                    return None
                
                for i in range(idx, len(cands)):
                    row, mask, m_val = cands[i]
                    if used_mask & mask == 0:
                        res = backtrack(i + 1, used_mask | mask, count + 1, current_selection + [row])
                        if res:
                            return res
                return None

            return backtrack(0, 0, 0, [])

        sol = find_disjoint_rulers(candidates, n_rows)
        if sol:
            return [list(r) for r in sol]
            
    return greedy_rows
