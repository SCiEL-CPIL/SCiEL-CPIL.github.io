import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    timeout = 145.0  # Safety margin against 150s kill
    
    k_plus_1 = k + 1
    
    def time_check():
        return time.time() - start_time > timeout

    def get_score(rows):
        if not rows:
            return float('inf')
        return max(max(row) for row in rows)

    def compute_diffs_mask(row):
        """Compute a bitmask of all differences in a row."""
        mask = 0
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                mask |= (1 << d)
        return mask

    def greedy_build_row(forbidden_mask, seed, max_val_limit=None):
        """
        Greedily build a single row of length k+1 starting at 0.
        Avoids differences already present in forbidden_mask.
        Returns the row if successful, else None.
        """
        random.seed(seed)
        row = [0]
        current_mask = 0  # Differences used within this row so far
        
        # We need to add k more elements
        for _ in range(k):
            found = False
            # Start searching from the smallest possible value
            # To ensure strict increase, start from row[-1] + 1
            start_x = row[-1] + 1
            
            # Heuristic: Don't search infinitely. If max_val_limit is set, respect it.
            # Otherwise, search up to a reasonable bound or until found.
            # For minimization, we want the smallest valid x.
            
            # We can randomize the search order slightly to break ties or explore?
            # No, for greedy minimization of max, we strictly want the smallest valid x.
            # Randomness comes from the initial state or previous choices.
            
            # However, if we always pick the smallest, we might get stuck in a bad local structure
            # for the *whole set*. But since we are rebuilding one row against fixed others,
            # picking the smallest valid x for this row is generally good for minimizing its max.
            
            x = start_x
            # Set a hard limit to prevent infinite loops if no solution exists in this branch
            # A safe upper bound for a single row given forbidden mask is tricky, 
            # but usually small numbers work. Let's try up to 500 or so for k=7.
            # Actually, if we are minimizing max, and we are stuck, it means this specific
            # forbidden set makes it hard to keep numbers small.
            
            search_limit = 1000 if max_val_limit is None else max_val_limit
            
            while x <= search_limit:
                # Check if x creates any forbidden differences
                valid = True
                new_diffs = 0
                for a in row:
                    d = x - a
                    # Check against global forbidden
                    if (forbidden_mask >> d) & 1:
                        valid = False
                        break
                    # Check against internal differences
                    if (new_diffs >> d) & 1:
                        valid = False
                        break
                    new_diffs |= (1 << d)
                
                if valid:
                    row.append(x)
                    current_mask |= new_diffs
                    found = True
                    break
                x += 1
            
            if not found:
                return None
        
        return row

    def generate_initial_solution(seed):
        """
        Generate a full set of rows using randomized greedy.
        """
        random.seed(seed)
        used_mask = 0
        rows = []
        
        for _ in range(n_rows):
            # Build row greedily
            # We don't have a fixed "forbidden" from a previous partial set in a simple way
            # because we build sequentially. But we can use the 'used_mask' of previous rows.
            
            # To add diversity, we can shuffle the order of rows or use random seeds for internal choices?
            # The greedy_build_row above is deterministic for a given forbidden mask.
            # To make the initial solution diverse, we can randomize the *order* in which we
            # consider candidates? Or just rely on the fact that different seeds for the
            # *entire process* might not change the deterministic greedy if it's strictly smallest-first.
            
            # Let's modify greedy_build_row to accept a "randomness" parameter or just
            # rely on the fact that we will do local search which fixes things.
            # Actually, a better initial solution:
            # Just build them one by one.
            
            row = [0]
            for _ in range(k):
                found = False
                x = row[-1] + 1
                # Search for valid x
                # To add randomness to the initial generation, we can pick a random valid x
                # from a small window of valid candidates, rather than always the smallest.
                
                candidates = []
                # Look ahead a bit
                for y in range(x, x + 50):
                    valid = True
                    new_diffs = 0
                    for a in row:
                        d = y - a
                        if (used_mask >> d) & 1:
                            valid = False
                            break
                        if (new_diffs >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    if valid:
                        candidates.append(y)
                        if len(candidates) > 5:
                            break
                
                if candidates:
                    # Pick randomly from the small set of valid small candidates
                    # This introduces diversity
                    chosen = random.choice(candidates)
                else:
                    # Fallback: find the first valid one
                    while True:
                        valid = True
                        new_diffs = 0
                        for a in row:
                            d = x - a
                            if (used_mask >> d) & 1:
                                valid = False
                                break
                            if (new_diffs >> d) & 1:
                                valid = False
                                break
                            new_diffs |= (1 << d)
                        if valid:
                            chosen = x
                            break
                        x += 1
                        if x > 500:
                            return None # Failed to build
                
                row.append(chosen)
                # Update used_mask with diffs from this new element
                for a in row[:-1]:
                    used_mask |= (1 << (chosen - a))
            
            rows.append(row)
            
        return rows

    def local_search(rows, seed):
        """
        Perform row-rebuild local search to minimize the maximum entry.
        """
        current_rows = [list(r) for r in rows]
        current_max = get_score(current_rows)
        improved = True
        
        while improved and not time_check():
            improved = False
            # Shuffle the order of rows to process to break symmetry
            order = list(range(n_rows))
            random.seed(seed + int(time.time() * 1000) % 10000)
            random.shuffle(order)
            
            for idx in order:
                if time_check():
                    break
                
                # Calculate forbidden mask from all OTHER rows
                forbidden_mask = 0
                for j in range(n_rows):
                    if j != idx:
                        forbidden_mask |= compute_diffs_mask(current_rows[j])
                
                # Attempt to rebuild row idx
                # We want to minimize the max of this row.
                # The greedy_build_row finds the smallest valid x at each step,
                # which minimizes the final max for that specific row given the forbidden mask.
                
                new_row = greedy_build_row(forbidden_mask, seed=idx, max_val_limit=current_max)
                
                if new_row is not None:
                    new_row_max = max(new_row)
                    # Check if this improves the global max
                    # The global max is the max of all row maxes.
                    # If we replace row idx with new_row, the new global max is:
                    # max( max(max(r) for r in current_rows if r is not current_rows[idx]), new_row_max )
                    
                    other_max = 0
                    for j in range(n_rows):
                        if j != idx:
                            other_max = max(other_max, max(current_rows[j]))
                    
                    new_global_max = max(other_max, new_row_max)
                    
                    if new_global_max < current_max:
                        current_rows[idx] = new_row
                        current_max = new_global_max
                        improved = True
                        # If we improved, we can continue searching
                        # But we should probably restart the loop to re-evaluate other rows?
                        # Or just continue. Continuing is usually fine.
                    elif new_row_max < max(current_rows[idx]):
                        # Even if it doesn't improve global max, if it reduces this specific row's max,
                        # it might help later? 
                        # Actually, if it doesn't improve global max, it doesn't change the score.
                        # But it might free up space? No, the forbidden mask for OTHER rows changes.
                        # Wait, if we change row idx, the forbidden mask for row j (j != idx) changes.
                        # So it might allow row j to be smaller.
                        # So we should accept it if it reduces the max of the row itself,
                        # even if the global max doesn't drop immediately, as it might enable
                        # further reductions in other rows.
                        
                        # However, we must ensure we don't oscillate.
                        # Simple rule: Only accept if it strictly reduces the max of the row being replaced.
                        if new_row_max < max(current_rows[idx]):
                            current_rows[idx] = new_row
                            # Global max might stay same or drop
                            new_global_max = max(other_max, new_row_max)
                            if new_global_max < current_max:
                                current_max = new_global_max
                            improved = True
        
        return current_rows, current_max

    # Main Loop: Multiple Restarts
    best_rows = None
    best_max = float('inf')
    
    # Try a few initial seeds
    # The number of restarts depends on time.
    # Let's do at least 10-20 quick restarts.
    
    for restart in range(100):
        if time_check():
            break
        
        seed = 42 + restart
        
        # 1. Generate Initial Solution
        init_rows = generate_initial_solution(seed)
        if init_rows is None:
            continue
            
        init_max = get_score(init_rows)
        
        # 2. Local Search
        ls_rows, ls_max = local_search(init_rows, seed)
        
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
            
        # Early termination if we hit the theoretical lower bound?
        # Counting bound is 112. If we hit 112, we can stop?
        # But we don't know if 112 is achievable for this specific n_rows, k.
        # The problem says "best known ... 134". So 112 is a lower bound.
        # If we find 112, it's optimal.
        if best_max <= 112:
            break

    if best_rows is None:
        # Fallback: Return a valid solution even if not optimized?
        # generate_initial_solution usually works.
        best_rows = generate_initial_solution(0)
        if best_rows is None:
             # Last resort: very simple greedy
             best_rows = []
             used = 0
             for _ in range(n_rows):
                 row = [0]
                 x = 1
                 while len(row) < k+1:
                     valid = True
                     new_diffs = 0
                     for a in row:
                         d = x - a
                         if (used >> d) & 1 or (new_diffs >> d) & 1:
                             valid = False
                             break
                         new_diffs |= (1 << d)
                     if valid:
                         row.append(x)
                         used |= new_diffs
                     else:
                         x += 1
                 best_rows.append(row)
    
    return best_rows
# EVOLVE-BLOCK-END
