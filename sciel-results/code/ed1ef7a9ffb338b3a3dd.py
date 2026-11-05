# EVOLVE-BLOCK-START
import sys
import time
import random
import math

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses local search (simulated annealing) with gap-based representation.
    Each row is represented by its k gaps (differences between consecutive elements).
    A solution is valid if all positive differences (sums of consecutive gap subsequences)
    are distinct across all rows.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
    # Helper: compute all positive differences from a row (list of k+1 increasing ints starting at 0)
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Helper: compute all positive differences from gaps
    def gaps_to_diffs(gaps):
        # gaps has length k, row has k+1 elements
        # row[0] = 0, row[i] = sum(gaps[0:i])
        diffs = set()
        for i in range(k):
            s = 0
            for j in range(i, k):
                s += gaps[j]
                diffs.add(s)
        return diffs
    
    # Helper: check if a full solution (list of gap lists) is valid
    def is_valid(gap_lists):
        all_diffs = set()
        for gaps in gap_lists:
            for d in gaps_to_diffs(gaps):
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return True
    
    # Helper: compute the largest entry (sum of all gaps) for a gap list
    def max_entry(gaps):
        return sum(gaps)
    
    # Helper: compute total cost (sum of all max entries, or just the global max)
    # We want to minimize the global max entry
    def global_max(gap_lists):
        return max(sum(gaps) for gaps in gap_lists)
    
    # Greedy construction to get an initial feasible solution
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
        # Convert to gaps
        gap_lists = []
        for row in rows:
            gaps = [row[i+1] - row[i] for i in range(k)]
            gap_lists.append(gaps)
        return gap_lists, rows
    
    # Get initial solution
    start_time = time.time()
    time_limit = 115.0
    
    initial_gaps, initial_rows = greedy_construct()
    best_gaps = [list(g) for g in initial_gaps]
    best_max = global_max(best_gaps)
    
    # Simulated annealing
    # State: list of gap lists, each of length k
    # Move: pick a random row, pick a random position in that row, change the gap
    #   - We can increase or decrease a gap by 1 (or more)
    #   - Must keep all gaps >= 1
    #   - Must maintain validity (all diffs distinct)
    
    # For efficiency, we track the set of all used differences
    def build_used_set(gap_lists):
        used = set()
        for gaps in gap_lists:
            for d in gaps_to_diffs(gaps):
                used.add(d)
        return used
    
    # Precompute for initial solution
    used_set = build_used_set(best_gaps)
    
    # Simulated annealing parameters
    T_start = 50.0
    T_end = 0.1
    cooling_rate = 0.995
    max_iterations = 500000
    
    rng = random.Random(42)
    
    current_gaps = [list(g) for g in best_gaps]
    current_used = build_used_set(current_gaps)
    current_max = global_max(current_gaps)
    
    T = T_start
    
    iteration = 0
    while iteration < max_iterations:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5.0:
            break
        
        # Pick a random row and position
        row_idx = rng.randrange(n_rows)
        pos = rng.randrange(k)
        
        old_gap = current_gaps[row_idx][pos]
        
        # Choose a new gap value
        # Proposal: change by -1, 0, or +1 with some probability, or larger jumps
        delta = rng.choice([-1, 1, -2, 2, -3, 3, -5, 5, -10, 10])
        new_gap = old_gap + delta
        
        if new_gap < 1:
            new_gap = 1
        # Cap new_gap to avoid absurdly large values
        if new_gap > 500:
            new_gap = 500
        
        if new_gap == old_gap:
            iteration += 1
            continue
        
        # Compute the differences that will be removed and added
        # For the row being modified, we need to recompute all differences
        # This is O(k^2) per move, which is acceptable for k=8
        
        old_row_gaps = current_gaps[row_idx]
        old_diffs = gaps_to_diffs(old_row_gaps)
        
        new_row_gaps = list(old_row_gaps)
        new_row_gaps[pos] = new_gap
        new_diffs = gaps_to_diffs(new_row_gaps)
        
        # Differences that are removed (in old but not in new)
        removed = old_diffs - new_diffs
        # Differences that are added (in new but not in old)
        added = new_diffs - old_diffs
        
        # Check if any added difference conflicts with:
        # 1. Other differences in the same row (already handled by set)
        # 2. Differences in other rows
        conflict = False
        for d in added:
            # Check if d is used by other rows
            # We need to check if d is in current_used but not in old_diffs
            # Actually, current_used contains all diffs from all rows
            # d is in added means it's in new_diffs but not in old_diffs
            # If d is in current_used, it could be from this row (but then it'd be in old_diffs, contradiction)
            # or from another row
            if d in current_used and d not in old_diffs:
                conflict = True
                break
        
        if conflict:
            # Reject the move
            iteration += 1
            # Cool down
            T *= cooling_rate
            continue
        
        # Compute the change in objective
        old_global_max = current_max
        new_row_max = sum(new_row_gaps)
        
        # Recompute global max
        other_maxes = [sum(current_gaps[i]) for i in range(n_rows) if i != row_idx]
        new_global_max = max([new_row_max] + other_maxes) if other_maxes else new_row_max
        
        delta_obj = new_global_max - old_global_max
        
        # Acceptance criterion
        if delta_obj <= 0:
            # Better or equal, always accept
            accept = True
        else:
            # Worse, accept with probability exp(-delta_obj / T)
            if T > 0:
                prob = math.exp(-delta_obj / T)
                accept = rng.random() < prob
            else:
                accept = False
        
        if accept:
            # Update state
            current_gaps[row_idx] = new_row_gaps
            # Update used set
            for d in removed:
                current_used.discard(d)
            for d in added:
                current_used.add(d)
            current_max = new_global_max
            
            # Update best
            if current_max < best_max:
                best_max = current_max
                best_gaps = [list(g) for g in current_gaps]
        
        # Cool down
        T *= cooling_rate
        iteration += 1
    
    # Convert best_gaps back to rows
    best_rows = []
    for gaps in best_gaps:
        row = [0]
        s = 0
        for g in gaps:
            s += g
            row.append(s)
        best_rows.append(row)
    
    # Final validation
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    if not validate(best_rows, n_rows, k):
        # Fallback to greedy
        best_rows = initial_rows
    
    return best_rows

# EVOLVE-BLOCK-END
