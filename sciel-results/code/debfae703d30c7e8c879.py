import time
import sys
import random
import math

def construct(n_rows, k):
    """
    Construct a Difference Triangle Set with n_rows rows, each of size k+1.
    Returns a list of rows.
    """
    start_time = time.time()
    deadline = 145.0 # 5s safety margin
    
    # The problem is (7, 5): 7 rows, 6 elements per row (indices 0 to 5).
    # Elements are a_0 < a_1 < ... < a_k with a_0 = 0.
    # We want to minimize max(a_k).
    
    # Best known for (7,5) is 112. Counting bound is 105.
    # We will search for solutions with M < 120.
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_solution(rows):
        used = set()
        for row in rows:
            diffs = get_diffs(row)
            if len(diffs) != len(row) * (len(row) - 1) // 2:
                return False
            if diffs & used:
                return False
            used |= diffs
        return True

    def backtrack_solve(n_rows, k, max_val_limit):
        """
        Attempts to find a valid DTS with max entry <= max_val_limit.
        Returns list of rows or None.
        """
        rows = []
        used_diffs = set()
        
        def solve_row(row_idx, current_row):
            # Check time
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                rows.append(current_row)
                # Recurse to next row
                if row_idx + 1 < n_rows:
                    if solve_row(row_idx + 1, [0]):
                        return True
                else:
                    return True
                # Backtrack
                rows.pop()
                # Remove diffs of this row from used_diffs
                diffs = get_diffs(current_row)
                used_diffs.difference_update(diffs)
                return False
            
            # Determine next value
            # The next value must be > current_row[-1]
            # And < max_val_limit (if it's the last element, it must be <= limit)
            # Actually, we just need to ensure the final element is <= limit.
            # Intermediate elements can be anything as long as the final one fits.
            # However, to keep it simple, we restrict all elements to be <= max_val_limit.
            
            last_val = current_row[-1] if current_row else 0
            is_last_pos = (len(current_row) == k)
            
            # Heuristic: Try small values first to keep the row compact
            # But we need to ensure we can fill the rest of the row within max_val_limit.
            # Simple heuristic: iterate from last_val + 1 upwards.
            
            # Optimization: If we are at the last position, we just need a valid value.
            # If we are not, we might need to leave room.
            
            # Let's try candidates in increasing order.
            # To speed up, we can limit the range.
            # If current_row is [0, 1, 2, 3, 4], next must be >= 5.
            
            # Forward checking:
            # If we pick x, the diffs {x - a for a in current_row} must not be in used_diffs.
            
            start_x = last_val + 1
            # Upper bound for x:
            # If this is the last element, x <= max_val_limit.
            # If not, we still want x to be reasonably small, but we must be able to fit
            # the remaining (k - len(current_row)) elements after x, all <= max_val_limit.
            # So x + (k - len(current_row)) <= max_val_limit is a loose bound.
            # Let's just use max_val_limit as the hard upper bound for all elements.
            
            for x in range(start_x, max_val_limit + 1):
                # Check time
                if time.time() > deadline:
                    return False
                
                new_diffs = set()
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Check internal consistency of current_row + [x]
                # The diffs within current_row are already checked when they were added.
                # We only need to check diffs involving x.
                # Wait, we also need to ensure that the diffs among current_row elements
                # are distinct from each other? No, that was checked when they were added.
                # But wait, if we backtrack, do we re-check?
                # The set `used_diffs` contains diffs from previous rows.
                # The diffs within `current_row` are not in `used_diffs` yet.
                # We must ensure that new_diffs does not intersect with diffs already in current_row.
                
                # Let's maintain a set of diffs for the current row as well.
                # Actually, it's easier to just check if any new diff conflicts with
                # existing diffs in current_row OR used_diffs.
                
                current_row_diffs = set()
                for i in range(len(current_row)):
                    for j in range(i+1, len(current_row)):
                        current_row_diffs.add(current_row[j] - current_row[i])
                        
                # This is O(k^2) per step, which is fine for k=5.
                # But we can optimize by maintaining current_row_diffs incrementally.
                # For simplicity and correctness, let's just compute it or maintain it.
                # Let's pass it as a parameter or maintain it in closure.
                
                # Let's refactor to maintain current_row_diffs.
                
                # If we use the set comprehension above, it's safe.
                
                if new_diffs & current_row_diffs:
                    continue
                if new_diffs & used_diffs:
                    continue
                
                # Valid move
                current_row.append(x)
                # Update used_diffs? No, only when row is complete.
                # But we need to ensure subsequent elements in this row don't conflict
                # with each other. The check above (new_diffs & current_row_diffs) handles that.
                
                # Recurse
                if solve_row(row_idx, current_row):
                    return True
                
                # Backtrack
                current_row.pop()
            
            return False
        
        # Initial call
        # We need to handle the first row specially or just let the loop handle it.
        # current_row starts as [0].
        if solve_row(0, [0]):
            return rows
        return None

    # Strategy:
    # 1. Try to find a solution with M close to the lower bound (105).
    # 2. Increase M slightly if it fails.
    # 3. Use randomization in the order of rows or values if deterministic fails?
    #    Backtracking with "smallest value first" is deterministic.
    #    To escape local minima (if any, though CSP doesn't have them in the same way),
    #    we can try different permutations of row construction or randomize value selection slightly.
    
    # Since the problem is small, a well-optimized backtracker should find M=112 quickly.
    
    best_solution = None
    
    # Try limits from 105 upwards
    for limit in range(105, 125):
        if time.time() > deadline - 5:
            break
            
        # Add a small randomization to avoid getting stuck in the same pattern
        # if the deterministic search is slow.
        # However, for DTS, the structure is quite rigid.
        
        # Let's try a randomized backtracking approach for variety.
        # We will shuffle the candidate values slightly? No, that breaks the "compactness" heuristic.
        # Instead, we can try different starting rows.
        
        # For now, let's just run the deterministic backtracker.
        # It should be fast enough for M=112.
        
        sol = backtrack_solve(n_rows, k, limit)
        if sol:
            best_solution = sol
            break # Found a solution for this limit, which is minimal for this limit.
            
    if best_solution:
        return best_solution
    
    # Fallback: If backtracking fails (e.g., due to time or complexity), use a greedy + local search
    # similar to the seed, but with a tighter time budget.
    
    def greedy_init(rng, n_rows, k, max_limit):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                x = row[-1] + 1
                while x <= max_limit:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        # Check internal row consistency
                        row_diffs = set()
                        for i in range(len(row)):
                            for j in range(i+1, len(row)):
                                row_diffs.add(row[j] - row[i])
                        if not (new_diffs & row_diffs):
                            row.append(x)
                            used |= new_diffs
                            found = True
                            break
                    x += 1
                if not found:
                    return None
            rows.append(row)
        return rows

    def local_search(rows, used_diffs, n_rows, k, rng, deadline):
        best_rows = [r[:] for r in rows]
        best_M = max(max(r) for r in rows)
        current_M = best_M
        
        T = 20.0
        cooling = 0.99
        
        while time.time() < deadline and T > 0.5:
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            row = rows[ri]
            
            lo = row[pi - 1] + 1
            hi = row[pi + 1] - 1 if pi < k else current_M + 5
            
            if lo > hi:
                continue
            
            old_val = row[pi]
            
            # Generate new value
            if random.random() < 0.8:
                new_val = rng.randint(lo, min(hi, current_M))
            else:
                new_val = rng.randint(lo, hi)
            
            if new_val == old_val:
                continue
            
            new_row = row[:]
            new_row[pi] = new_val
            if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                continue
            
            # Check validity
            new_diffs = get_diffs(new_row)
            expected = len(new_row) * (len(new_row) - 1) // 2
            if len(new_diffs) != expected:
                continue
            
            # Check against other rows
            other_diffs = set()
            for i, r in enumerate(rows):
                if i != ri:
                    other_diffs |= get_diffs(r)
            
            if new_diffs & other_diffs:
                continue
            
            # Valid move
            new_M = max(max(new_row), max(max(r) for i, r in enumerate(rows) if i != ri))
            delta_E = new_M - current_M
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                rows[ri] = new_row
                current_M = new_M
                
                if current_M < best_M:
                    best_M = current_M
                    best_rows = [r[:] for r in rows]
            
            T *= cooling
        
        return best_rows, best_M

    rng = random.Random(42)
    rows = None
    # Try greedy with increasing limits
    for limit in range(110, 130):
        if time.time() > deadline - 10:
            break
        rows = greedy_init(rng, n_rows, k, limit)
        if rows:
            break
            
    if rows:
        used_diffs = set()
        for r in rows:
            used_diffs |= get_diffs(r)
        
        final_rows, _ = local_search(rows, used_diffs, n_rows, k, rng, deadline - 2)
        if is_valid_solution(final_rows):
            return final_rows
        return rows
    else:
        # Absolute fallback: return a trivial valid solution if possible, or empty
        # This case should not happen for (7,5)
        return []

# Test code (not run during evaluation)
if __name__ == "__main__":
    sol = construct(7, 5)
    print(sol)
    max_val = max(max(row) for row in sol)
    print(f"Max value: {max_val}")
    # Validate
    used = set()
    for row in sol:
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    print("Conflict found!")
                used.add(d)
                diffs.add(d)
        if len(diffs) != 15:
            print("Internal conflict in row!")
    print("Valid" if is_valid_solution(sol) else "Invalid")
