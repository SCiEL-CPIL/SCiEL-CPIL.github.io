import time
import random
import math
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin (150s limit, 120s stated, be safe)
    
    # Total number of differences needed
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    # Lower bound on scope M
    M_lower = int(math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2))
    
    # Strategy:
    # 1. Try a few values of M starting from M_lower.
    # 2. Use a hybrid approach:
    #    - Generate random candidate rows using a greedy heuristic with randomization.
    #    - Check for conflicts (duplicate differences).
    #    - Use Local Search (Simulated Annealing or Hill Climbing) to fix conflicts.
    
    # State representation:
    # A solution is a list of `n_rows` rows, each a list of `k+1` ints.
    # Rows are strictly increasing, start with 0.
    # Max element <= M.
    
    def count_conflicts(rows):
        """Returns number of duplicate differences."""
        seen = set()
        conflicts = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        conflicts += 1
                    else:
                        seen.add(d)
        return conflicts
    
    def get_scope(rows):
        return max(max(row) for row in rows)
    
    # Heuristic to generate a valid single row (no internal conflicts, but doesn't check global)
    # Actually, internal conflicts in a single row are impossible if the row is strictly increasing?
    # No, internal differences within a single row are distinct by construction?
    # Wait. The task says: "all differences between two entries of the same row are distinct over all rows".
    # This means the set of ALL differences across ALL rows must be unique.
    # Within a single row, differences are a_j - a_i. Are they distinct?
    # Example: row [0, 1, 3]. Diffs: 1, 3, 2. Distinct.
    # Example: row [0, 2, 3]. Diffs: 2, 3, 1. Distinct.
    # In a strictly increasing sequence, are differences always distinct?
    # No. Consider [0, 1, 2, 4]. Diffs: 1, 2, 4, 1, 3, 2.
    # 1 appears twice (2-1=1, 1-0=1). 2 appears twice (3-1=2, 2-0=2).
    # So even within a single row, differences can repeat.
    # The condition is that the GLOBAL set of differences (across all rows, all pairs) must be distinct.
    
    def generate_random_row(M, k, used_diffs_global):
        """
        Generates a random row of length k+1 starting at 0, max <= M.
        Tries to avoid diffs already in used_diffs_global.
        """
        row = [0]
        # We need to pick k more elements.
        # Greedy with randomization.
        # To make it more diverse, we can randomize the step sizes.
        # But we must ensure strictly increasing.
        
        # Let's pick elements one by one.
        # For each next element, pick a random value > last, <= M.
        # If it creates a conflict with used_diffs_global or internal diffs, retry or pick next.
        
        # A better way: 
        # Pick a random subset of {1..M} of size k, sort it, prepend 0.
        # But this might lead to high density and more conflicts.
        # The "best" rows are often sparse.
        
        # Let's try a random walk.
        current = 0
        for _ in range(k):
            # Pick a random step
            # Max remaining space: M - current
            # We need to leave enough space for remaining elements.
            # Remaining elements: k - len(row) + 1 (including the one we are about to pick? No, len(row) is current count)
            # We need `k - len(row) + 1` more elements? No.
            # Current len(row). Target len k+1.
            # Need `k + 1 - len(row)` more elements.
            # Let r = k + 1 - len(row).
            # We need to pick r elements.
            # The smallest possible next element is current + 1.
            # The largest possible next element is M - (r - 1) (leaving 1 for each subsequent).
            
            r = k + 1 - len(row)
            min_next = current + 1
            max_next = M - (r - 1)
            
            if min_next > max_next:
                return None # Impossible to fit
            
            # Randomly choose next
            # Bias towards smaller steps to keep M low? Or uniform?
            # Uniform might be better for diversity.
            # But to find valid solutions quickly, maybe bias towards values that don't conflict?
            
            # Let's try a few random candidates
            for _ in range(20):
                candidate = random.randint(min_next, max_next)
                # Check internal conflicts with existing row
                # and global conflicts
                valid = True
                for a in row:
                    d = candidate - a
                    if d in used_diffs_global:
                        valid = False
                        break
                if valid:
                    row.append(candidate)
                    current = candidate
                    break
            else:
                # If we failed 20 times, just pick the smallest possible?
                # Or fail?
                # Let's just pick a random one and accept the conflict, hoping Local Search fixes it.
                candidate = random.randint(min_next, max_next)
                row.append(candidate)
                current = candidate
        
        return row

    def local_search(rows, M, time_limit=5.0):
        """
        Improves the rows to reduce conflicts.
        Moves: Change a single element in a row.
        """
        start_ls = time.time()
        current_conflicts = count_conflicts(rows)
        if current_conflicts == 0:
            return rows, 0
        
        # Temperature for SA? Or just Hill Climbing with restarts?
        # Let's try Hill Climbing with random restarts (Perturbation).
        
        best_rows = [row[:] for row in rows]
        best_conflicts = current_conflicts
        
        while time.time() < start_ls + time_limit:
            # Pick a random row and a random position to change
            r_idx = random.randint(0, n_rows - 1)
            # Position 0 is fixed (0).
            pos = random.randint(1, k)
            
            row = rows[r_idx]
            old_val = row[pos]
            
            # Pick a new value
            # Must maintain strictly increasing order.
            # min_val = row[pos-1] + 1
            # max_val = row[pos+1] - 1 if pos < k else M
            min_val = row[pos - 1] + 1
            max_val = row[pos + 1] - 1 if pos < k else M
            
            if min_val > max_val:
                continue
            
            # Random choice
            new_val = random.randint(min_val, max_val)
            if new_val == old_val:
                continue
            
            # Apply move
            row[pos] = new_val
            
            # Evaluate
            new_conflicts = count_conflicts(rows)
            
            # Accept if better or (with probability p) if worse?
            # For simplicity, accept if better. If stuck, randomize more.
            if new_conflicts < best_conflicts:
                best_conflicts = new_conflicts
                best_rows = [r[:] for r in rows]
                if best_conflicts == 0:
                    break
            else:
                # Revert if not improving? 
                # Or keep it if it's a "kick"?
                # Let's do simple hill climbing: revert if not better.
                # But this can get stuck.
                # Let's add a small probability of accepting worse moves (SA-like).
                # Or just accept it to allow exploration, but track the best seen.
                # Let's accept it for now to allow wandering.
                pass
        
        return best_rows, best_conflicts

    # Main Loop:
    # Try increasing M.
    best_sol = None
    best_M = None
    
    # We want to find the smallest M.
    # Let's try M from M_lower to M_lower + 50.
    
    # To be efficient, we'll run a limited amount of search for each M.
    
    for offset in range(0, 60):
        M = M_lower + offset
        
        if time.time() > deadline - 10:
            break
            
        # Time budget for this M: 5 seconds?
        # If we find a solution, we stop.
        # If not, we try next M.
        
        # Initialization:
        # Generate n_rows random rows.
        
        found = False
        rows = []
        
        # Strategy: 
        # 1. Generate all rows randomly.
        # 2. Check conflicts.
        # 3. Local search.
        
        # To ensure diversity, we can generate the rows one by one, 
        # using the diffs from previous rows to guide the generation?
        # Or just generate all at once and fix.
        
        # Let's generate all at once.
        
        for _ in range(n_rows):
            r = generate_random_row(M, k, set()) # Empty set for now, we'll handle global conflicts in LS
            if r is None:
                r = [0] + list(range(1, k+1)) # Fallback: dense row
                # Scale it?
                # If M is small, dense row might exceed M.
                # Ensure it fits M.
                # [0, 1, ..., k] has max k. If M >= k, ok.
            rows.append(r)
        
        # Ensure rows fit M
        for r in rows:
            if max(r) > M:
                # Scale down?
                # Just truncate and re-pad?
                # Simpler: regenerate.
                pass
        
        # Local Search
        # We have 5 seconds per M.
        # Let's use 3 seconds for LS.
        best_rows, best_conf = local_search(rows, M, time_limit=3.0)
        
        if best_conf == 0:
            best_sol = best_rows
            best_M = M
            found = True
            break
            
        # If not found, try another initialization?
        # Maybe 1 restart with new random rows?
        
        # Let's do a second attempt with a different initialization strategy:
        # Greedy construction of first row, then others.
        
        # Actually, let's just try another random init if time permits.
        if time.time() < deadline - 15:
            rows2 = []
            for _ in range(n_rows):
                r = generate_random_row(M, k, set())
                if r is None:
                    r = [0] + list(range(1, k+1))
                rows2.append(r)
            
            best_rows2, best_conf2 = local_search(rows2, M, time_limit=2.0)
            if best_conf2 == 0:
                best_sol = best_rows2
                best_M = M
                found = True
                break
            elif best_conf2 < best_conf:
                best_rows = best_rows2
                best_conf = best_conf2
    
    if best_sol is None:
        # Fallback: Greedy construction with large M
        # This ensures validity.
        M_fallback = 500
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    valid = True
                    for a in row:
                        if x - a in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used_diffs.add(x - a)
                        break
                    x += 1
            rows.append(row)
        
        best_sol = rows
    
    return best_sol
