"""Rewritten program: randomized greedy construction with time-bounded restarts."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within each row are distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    # Each row has C(k+1, 2) = k*(k+1)//2 differences
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # The simple counting bound: need at least total_diffs distinct positive integers
    # up to the max entry M, so M >= total_diffs
    # But we also need M >= k (since row goes 0, ..., M with k steps)
    counting_bound = total_diffs

    def greedy_randomized(rows_seed, max_val):
        """
        Try to build rows greedily. rows_seed is a list of partial rows
        (each a list starting with [0]). max_val is the upper bound for entries.
        Returns a complete set of rows or None if stuck.
        """
        used = set()
        rows = []
        
        for seed_row in rows_seed:
            row = list(seed_row)
            if not row:
                row = [0]
            if row[0] != 0:
                row = [0] + row
            
            while len(row) < k + 1:
                # Try to find the next value
                # We need to pick x > row[-1] such that all x - a for a in row are not in used
                # and x <= max_val
                found = False
                # Randomize the order of trying values to escape local minima
                # Start from row[-1] + 1 up to max_val
                # But to be efficient, we can try a few random candidates first
                
                start = row[-1] + 1
                end = min(max_val, start + 200)  # limit search window for speed
                candidates = list(range(start, end + 1))
                random.shuffle(candidates)
                
                for x in candidates:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                
                if not found:
                    # Try a wider search if we haven't exhausted
                    for x in range(start, max_val + 1):
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                            found = True
                            break
                
                if not found:
                    return None
            rows.append(row)
        
        return rows

    def greedy_standard(max_val, randomize_order=False):
        """
        Standard greedy: build each row from scratch.
        If randomize_order, try values in randomized order within a window.
        """
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                start = row[-1] + 1
                
                if randomize_order:
                    # Try a window of candidates in random order
                    window_end = min(max_val, start + 100)
                    candidates = list(range(start, window_end + 1))
                    random.shuffle(candidates)
                    for x in candidates:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                            found = True
                            break
                    if not found:
                        for x in range(window_end + 1, max_val + 1):
                            new_diffs = {x - a for a in row}
                            if not new_diffs & used:
                                row.append(x)
                                used |= new_diffs
                                found = True
                                break
                else:
                    for x in range(start, max_val + 1):
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                            found = True
                            break
                
                if not found:
                    return None
            rows.append(row)
        
        return rows

    def get_max_entry(rows):
        return max(max(row) for row in rows)

    def is_valid(rows):
        """Check that all rows are valid and all differences are distinct."""
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    # First, try a standard greedy with a reasonable upper bound
    # Start with an upper bound slightly above the counting bound
    # The counting bound is total_diffs, but in practice we need more
    
    best_rows = None
    best_max = float('inf')
    
    # Try standard greedy first with a generous bound
    # Upper bound estimate: counting bound + some slack
    # For (4,7): counting bound = 4*28 = 112, best known is 134
    initial_bound = counting_bound + 50
    
    # Try standard greedy
    rows = greedy_standard(initial_bound, randomize_order=False)
    if rows and is_valid(rows):
        best_rows = rows
        best_max = get_max_entry(rows)
    
    # Now try randomized restarts to improve
    # Use the best found so far as a reference
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_rows if best_rows else greedy_standard(initial_bound + 100)
    
    # Try multiple randomized attempts with tighter bounds
    # Work backwards from best_max to find a better solution
    attempts = 0
    max_attempts = 200
    
    # Try to improve by using randomized greedy with the current best bound
    current_bound = best_max if best_max < float('inf') else initial_bound
    
    while time.time() - start_time < deadline - 5:
        attempts += 1
        if attempts > max_attempts:
            break
        
        # Randomize the seed: start with a few random initial differences
        # But keep it simple: just use randomized greedy
        rows = greedy_standard(current_bound, randomize_order=True)
        if rows and is_valid(rows):
            m = get_max_entry(rows)
            if m < best_max:
                best_max = m
                best_rows = rows
                current_bound = m  # Try to go even lower
                # Reset attempts to allow more exploration at new bound
                attempts = 0
                max_attempts = 100
    
    # Final fallback: if we still don't have a valid result, use a very generous bound
    if best_rows is None:
        for bound in [initial_bound + 100, initial_bound + 200, initial_bound + 500]:
            rows = greedy_standard(bound, randomize_order=False)
            if rows and is_valid(rows):
                best_rows = rows
                break
    
    if best_rows is None:
        # Last resort: very generous bound
        rows = greedy_standard(initial_bound + 1000, randomize_order=False)
        best_rows = rows
    
    return best_rows
# EVOLVE-BLOCK-END
