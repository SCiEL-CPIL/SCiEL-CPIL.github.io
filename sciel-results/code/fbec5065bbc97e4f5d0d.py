import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer

    total_diffs = n_rows * (k * (k + 1) // 2)
    lb = total_diffs

    def get_cost(rows):
        diffs = {}
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    diffs[d] = diffs.get(d, 0) + 1
        # Cost is number of duplicates (excess count)
        cost = 0
        for d, cnt in diffs.items():
            if cnt > 1:
                cost += cnt - 1
        return cost

    def is_valid(rows):
        return get_cost(rows) == 0

    def max_val(rows):
        return max(max(row) for row in rows)

    def local_search(limit, time_limit=5.0):
        # Initialize
        rows = []
        for _ in range(n_rows):
            # Generate random increasing sequence
            # Simple way: sort random distinct values
            # But we need to ensure they fit in [0, limit]
            # And start with 0.
            vals = [0]
            # We need k more values.
            # Randomly pick from 1..limit, sort, ensure distinct.
            # If limit is small, this might fail or be hard.
            # Better: Greedy random.
            last = 0
            for _ in range(k):
                # Pick random x in [last+1, limit]
                # Bias towards smaller? Or uniform?
                # Uniform might spread them out.
                if last >= limit:
                    return None
                x = random.randint(last + 1, limit)
                vals.append(x)
                last = x
            rows.append(vals)
            
        start_ls = time.time()
        best_cost = get_cost(rows)
        
        while time.time() - start_ls < time_limit:
            if best_cost == 0:
                return rows
            
            # Pick a random row and position to change
            r = random.randint(0, n_rows - 1)
            # Position 0 is fixed at 0. Change 1..k
            i = random.randint(1, k)
            
            # Determine valid range for new value
            lo = rows[r][i - 1] + 1
            hi = rows[r][i + 1] - 1 if i < k else limit
            
            if lo > hi:
                continue
                
            # Try a few random candidates
            improved = False
            for _ in range(10):
                new_val = random.randint(lo, hi)
                if new_val == rows[r][i]:
                    continue
                
                old_val = rows[r][i]
                rows[r][i] = new_val
                
                new_cost = get_cost(rows)
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    improved = True
                    break
                else:
                    rows[r][i] = old_val
            
            if not improved:
                # Stuck, restart or random kick
                # Random kick: accept a worse move occasionally?
                # Or just restart initialization?
                # Let's do a small random kick
                r = random.randint(0, n_rows - 1)
                i = random.randint(1, k)
                lo = rows[r][i - 1] + 1
                hi = rows[r][i + 1] - 1 if i < k else limit
                if lo <= hi:
                    new_val = random.randint(lo, hi)
                    rows[r][i] = new_val
                    # Update cost, even if worse
                    best_cost = get_cost(rows)

        if best_cost == 0:
            return rows
        return None

    best_rows = None
    best_max = float('inf')

    # Search for smallest limit
    # We know LB is 112. Best known is 134.
    # Try limits from LB to LB + 30
    for limit in range(lb, lb + 40):
        if time.time() > deadline:
            break
            
        # Try a few restarts for this limit
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            res = local_search(limit, time_limit=5.0)
            if res is not None:
                # Valid solution found
                m = max_val(res)
                if m < best_max:
                    best_max = m
                    best_rows = [row[:] for row in res]
                found = True
                # If we found a solution, we can break out of attempts
                # But maybe a later attempt finds a smaller max?
                # The max is bounded by `limit`.
                # If we find a solution, its max is <= limit.
                # If we found one with max 120, and limit is 125, we might find one with max 122?
                # No, we want the global minimum.
                # If we find a valid solution for `limit`, the max is <= limit.
                # We are iterating `limit` upwards.
                # The first `limit` that yields a solution gives a max <= limit.
                # Is it guaranteed to be the minimum?
                # If there is a solution with max 112, it would be found at limit=112.
                # If not, limit=113, etc.
                # So the first success gives the smallest possible max (within the search power).
                # So we can break out of the limit loop?
                # Yes, if we are confident the local search can find it.
                # But local search might miss it.
                # So we should probably try a bit harder for the current limit before moving to next.
                # Let's break out of `attempt` loop, but continue to next limit?
                # No, if we found a solution with limit=115, max is <=115.
                # If we try limit=116, we might find max=116? Or 115?
                # We want the *smallest* max.
                # If we found a valid solution, its max is M.
                # We should record M.
                # If we then find a solution with smaller M, we update.
                # But we are increasing `limit`.
                # A solution with `limit=115` has max <= 115.
                # A solution with `limit=116` has max <= 116.
                # It is possible the `limit=116` search finds a solution with max 112 (if we missed it in 115).
                # So we should NOT break the limit loop immediately.
                # We should continue searching for smaller max values?
                # But `limit` is a hard cap.
                # If we found a solution with max 115 at limit 115, we know max >= 112.
                # We can try to improve it by local search minimizing max value directly?
                # For now, let's just record the best found so far.
                pass

        # If we found a solution, we can break if we are close to LB?
        # Let's just let it run until time out or we hit a high limit.
        # To save time, if we found a solution, we can reduce time per limit.

    # Fallback: If no solution found (should not happen), generate a large valid one
    if best_rows is None:
        # Greedy with large upper
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                for x in range(row[-1] + 1, 1000):
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                        found = True
                        break
                if not found:
                    break
            if len(row) < k + 1:
                return [[0, 1, 2, 3, 4, 5, 6, 7]] * n_rows
            rows.append(row)
        best_rows = rows

    return best_rows
