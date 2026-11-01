import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin for safety
    
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    def greedy_row(used, k, max_val_hint=1000):
        """Constructs a row greedily avoiding 'used' differences."""
        row = [0]
        for _ in range(k):
            found = False
            last = row[-1]
            # Try increasing values
            # Heuristic: start from last+1, but maybe we can jump?
            # Greedy usually works best starting small.
            for x in range(last + 1, max_val_hint + 1):
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break
            if not found:
                return None
        return row

    def find_best_row(used, k, time_limit):
        """
        Finds a row of length k+1 starting at 0 that avoids 'used' differences.
        Minimizes the last element. Uses DFS with pruning.
        """
        best_row = None
        best_max = float('inf')
        
        # We need to search for the row.
        # Since we want to minimize the max, we can do a BFS or DFS with increasing max value.
        # Or just DFS and keep the best found so far, pruning if current path max > best_max.
        
        def dfs(row, current_max):
            nonlocal best_row, best_max
            
            if time.time() > time_limit:
                return
            
            if len(row) == k + 1:
                if current_max < best_max:
                    best_max = current_max
                    best_row = row[:]
                return
            
            last = row[-1]
            # The next value x must be > last.
            # Also, the new differences {x - a for a in row} must not be in 'used'.
            # And they must not be in the differences already formed by 'row' (which is guaranteed if we build incrementally and check against 'used' + current row diffs? No, 'used' is external. Internal diffs are unique by construction of a ruler? No, a row must be a ruler itself? 
            # The task says: "all differences between two entries of the same row are distinct over all rows".
            # This implies differences within a row must be distinct (it's a ruler) AND distinct from other rows.
            # So yes, we must ensure new_diffs are not in 'used' AND not in existing diffs of 'row'.
            
            # Let's pre-compute existing diffs of 'row' to check against?
            # Actually, if we add x, the new diffs are x - a_i.
            # We need x - a_i != x - a_j for i!=j (always true if a_i != a_j).
            # We need x - a_i != a_j - a_i for any previous pair (a_j, a_i) with j>i?
            # i.e., x != a_j - a_i + a_i = a_j? No.
            # The condition is: the set of all diffs in the final row must be unique.
            # So {x - a_i for all i < len(row)} must be disjoint from {a_j - a_i for all 0<=i<j<len(row)}.
            
            existing_diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    existing_diffs.add(row[j] - row[i])
            
            for x in range(last + 1, best_max if best_max < float('inf') else 200):
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                if new_diffs & existing_diffs:
                    continue
                
                row.append(x)
                dfs(row, x)
                row.pop()
                
                # Pruning: if we found a solution, we don't need to try larger x for this position?
                # Not necessarily, because a larger x here might allow smaller values later?
                # No, if we are minimizing the max, and we just found a solution with max M,
                # any solution extending this path with x' > x will have max >= x'.
                # If x' >= best_max, we can prune.
                # But wait, we are iterating x upwards.
                # If we found a solution in the first branch, best_max is updated.
                # Subsequent x values are larger, so they can't improve the max if the rest of the row is built greedily?
                # Actually, if we pick a larger x now, the remaining elements must be even larger, so the final max will be larger than x.
                # So if x >= best_max, we can break.
                if best_max < float('inf') and x >= best_max:
                    break

        dfs([0], 0)
        return best_row

    # Initial Solution
    rows = []
    used = set()
    for _ in range(n_rows):
        row = greedy_row(used, k)
        if row is None:
            # Fallback: random restart or just use a very high limit
            row = greedy_row(used, k, 500)
            if row is None:
                # Should not happen for small n,k
                return [[0, 1, 2, 3, 4, 5] for _ in range(n_rows)] # Invalid fallback
        rows.append(row)
        used.update(get_diffs(row))
    
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    
    # Local Search
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart with a different initial configuration?
            # Or just randomize one row completely?
            # Let's try a random restart of the greedy phase with some noise
            rows = []
            used = set()
            for i in range(n_rows):
                # Add some randomness to greedy
                row = [0]
                for _ in range(k):
                    last = row[-1]
                    found = False
                    # Try a few random candidates before falling back to greedy
                    candidates = list(range(last + 1, last + 20))
                    random.shuffle(candidates)
                    for x in candidates:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            # Check internal uniqueness
                            internal_diffs = {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
                            if not new_diffs & internal_diffs:
                                row.append(x)
                                used.update(new_diffs)
                                found = True
                                break
                    if not found:
                        # Fallback to greedy
                        for x in range(last + 1, 500):
                            new_diffs = {x - a for a in row}
                            if not new_diffs & used:
                                internal_diffs = {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
                                if not new_diffs & internal_diffs:
                                    row.append(x)
                                    used.update(new_diffs)
                                    found = True
                                    break
                rows.append(row)
            current_max = max(max(r) for r in rows)
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
            else:
                no_improve_count += 100
            continue

        # Pick a random row to optimize
        i = random.randint(0, n_rows - 1)
        
        # Remove row i from used set
        row_diffs = get_diffs(rows[i])
        used_excluding_i = used - row_diffs
        
        # Find best row for position i
        new_row = find_best_row(used_excluding_i, k, deadline)
        
        if new_row is not None:
            # Update rows and used set
            used = used_excluding_i | get_diffs(new_row)
            rows[i] = new_row
            
            current_max = max(max(r) for r in rows)
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

    return best_rows
