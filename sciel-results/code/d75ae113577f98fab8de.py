"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import random
from collections import deque

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a difference-first greedy approach: assign differences to rows,
    then reconstruct positions. Multiple randomized restarts for robustness.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound: m*(m+1)/2 >= total_diffs_needed
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs_needed:
        lb += 1
    
    # Greedy baseline (position-based)
    def greedy_construct(n_rows, k):
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
    
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    start_time = time.time()
    time_limit = 115.0
    
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
    
    # Difference-first construction:
    # Each row needs exactly k*(k+1)/2 distinct positive differences.
    # We partition the difference set {1, ..., M} into n_rows subsets of size k*(k+1)/2,
    # then for each subset, find positions that realize exactly those differences.
    
    def reconstruct_row(diff_set, max_entry):
        """
        Given a set of required differences, find k+1 positions starting at 0
        that realize exactly these differences.
        Returns list of positions or None if impossible.
        """
        n_pos = k + 1
        if len(diff_set) != n_pos * (n_pos - 1) // 2:
            return None
        
        # Try to build the row by selecting positions
        # Use backtracking with pruning
        positions = [0]
        
        def backtrack(pos_list, remaining_diffs):
            if time.time() > deadline:
                return False
            if len(pos_list) == n_pos:
                # Verify all differences match
                actual_diffs = set()
                for i in range(len(pos_list)):
                    for j in range(i + 1, len(pos_list)):
                        actual_diffs.add(pos_list[j] - pos_list[i])
                return actual_diffs == diff_set
            
            last = pos_list[-1]
            remaining_count = n_pos - len(pos_list)
            
            # Candidate next positions
            candidates = []
            for x in range(last + 1, max_entry + 1):
                # Prune: need room for remaining_count-1 more
                if max_entry - x < remaining_count - 1:
                    break
                
                # Compute new differences
                new_diffs = {x - a for a in pos_list}
                # All new diffs must be in remaining_diffs
                if not new_diffs.issubset(remaining_diffs):
                    continue
                
                # Score: prefer positions that use "smaller" remaining diffs
                # to leave larger diffs for later (they're harder to place)
                score = sum(new_diffs)
                candidates.append((score, x, new_diffs))
            
            # Try candidates in order of score (ascending = use small diffs first)
            candidates.sort()
            for score, x, new_diffs in candidates:
                if time.time() > deadline:
                    return False
                pos_list.append(x)
                new_remaining = remaining_diffs - new_diffs
                if backtrack(pos_list, new_remaining):
                    return True
                pos_list.pop()
            
            return False
        
        if backtrack(positions, set(diff_set)):
            return positions
        return None
    
    def difference_first_greedy(n_rows, k, max_entry, seed=0, deadline=None):
        """
        Assign differences to rows greedily, then reconstruct each row.
        """
        rng = random.Random(seed)
        if deadline is None:
            deadline = time.time() + 10.0
        
        total_per_row = k * (k + 1) // 2
        all_diffs = list(range(1, max_entry + 1))
        
        # Shuffle differences to break symmetry
        rng.shuffle(all_diffs)
        
        # Try to assign differences to rows
        row_diff_sets = []
        idx = 0
        for r in range(n_rows):
            row_diffs = set()
            for _ in range(total_per_row):
                if idx >= len(all_diffs):
                    return None
                row_diffs.add(all_diffs[idx])
                idx += 1
            row_diff_sets.append(row_diffs)
        
        # Reconstruct each row
        rows = []
        for diff_set in row_diff_sets:
            row = reconstruct_row(diff_set, max_entry)
            if row is None:
                return None
            rows.append(row)
        
        # Final validation
        if not validate(rows, n_rows, k):
            return None
        
        return rows
    
    # Smarter: use a priority-based difference assignment
    def priority_difference_greedy(n_rows, k, max_entry, seed=0, deadline=None):
        """
        Assign differences to rows using a priority queue approach.
        Larger differences are harder to place, so assign them first.
        """
        rng = random.Random(seed)
        if deadline is None:
            deadline = time.time() + 10.0
        
        total_per_row = k * (k + 1) // 2
        all_diffs = list(range(1, max_entry + 1))
        
        # Sort by descending value (hard to place first)
        all_diffs.sort(reverse=True)
        # Add some randomization
        for i in range(len(all_diffs) - 1, 0, -1):
            j = rng.randint(0, i)
            all_diffs[i], all_diffs[j] = all_diffs[j], all_diffs[i]
        
        # Assign round-robin: give one diff to each row in turn
        row_diff_lists = [[] for _ in range(n_rows)]
        for d in all_diffs:
            # Assign to the row with fewest diffs so far
            min_idx = min(range(n_rows), key=lambda i: len(row_diff_lists[i]))
            row_diff_lists[min_idx].append(d)
        
        # Check each row has exactly total_per_row diffs
        for i in range(n_rows):
            if len(row_diff_lists[i]) != total_per_row:
                return None
        
        # Reconstruct each row
        rows = []
        for i in range(n_rows):
            if time.time() > deadline:
                return None
            diff_set = set(row_diff_lists[i])
            row = reconstruct_row(diff_set, max_entry)
            if row is None:
                return None
            rows.append(row)
        
        if not validate(rows, n_rows, k):
            return None
        
        return rows
    
    # Hybrid: try both strategies with multiple seeds
    def hybrid_search(n_rows, k, max_entry, num_seeds=5, deadline=None):
        if deadline is None:
            deadline = time.time() + 10.0
        
        best = None
        for seed in range(num_seeds):
            if time.time() > deadline:
                break
            # Try priority-based
            result = priority_difference_greedy(n_rows, k, max_entry, seed=seed, deadline=deadline)
            if result is not None:
                return result
            # Try shuffled
            result = difference_first_greedy(n_rows, k, max_entry, seed=seed, deadline=deadline)
            if result is not None:
                return result
        
        return None
    
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Try to improve from lb upward
    max_search_range = min(greedy_max, lb + 25)
    
    for max_entry in range(lb, max_search_range + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 5
        
        if remaining_time < 2.0:
            break
        
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(20.0, remaining_time)
            num_seeds = 10
        elif distance_from_lb <= 5:
            budget = min(10.0, remaining_time)
            num_seeds = 5
        elif distance_from_lb <= 15:
            budget = min(5.0, remaining_time)
            num_seeds = 3
        else:
            budget = min(2.0, remaining_time)
            num_seeds = 2
        
        if budget < 1.0:
            break
        
        deadline = time.time() + budget
        result = hybrid_search(n_rows, k, max_entry, num_seeds=num_seeds, deadline=deadline)
        if result is not None:
            best_rows = result
            best_max = max_entry
            break
    
    # If no improvement found, try a few more with the greedy max
    if best_max == greedy_max:
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 5
        if remaining_time > 5.0:
            for max_entry in range(max(lb, greedy_max - 10), greedy_max):
                if time.time() - start_time > time_limit - 5:
                    break
                budget = min(3.0, time_limit - (time.time() - start_time) - 5)
                if budget < 1.0:
                    break
                deadline = time.time() + budget
                result = hybrid_search(n_rows, k, max_entry, num_seeds=3, deadline=deadline)
                if result is not None:
                    best_rows = result
                    best_max = max_entry
                    break
    
    # Final validation
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
