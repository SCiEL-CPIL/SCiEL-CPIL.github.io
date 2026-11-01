"""Difference triangle set (7,5) - improved search with local optimization."""
import time
import random
import itertools

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct across all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    
    def get_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            row_diffs = get_diffs(row)
            if len(row_diffs) != len(row) * (len(row) - 1) // 2:
                return False  # duplicate within row
            if all_diffs & row_diffs:
                return False  # conflict across rows
            all_diffs |= row_diffs
        return True
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    def greedy_construct(n_rows, k):
        """Original greedy approach as baseline."""
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
    
    # Start with greedy
    best_rows = greedy_construct(n_rows, k)
    best_score = max_entry(best_rows)
    
    # Local search: try to reduce the largest entry by modifying rows
    def try_improve(rows, deadline):
        """Try to improve the solution by local moves."""
        rows = [list(r) for r in rows]
        score = max_entry(rows)
        
        # Collect all used differences
        all_diffs = set()
        for row in rows:
            all_diffs |= get_diffs(row)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to reduce each row's maximum
            for idx in range(len(rows)):
                row = rows[idx]
                if max(row) < score:
                    continue
                # Try replacing the last element with something smaller
                last = row[-1]
                # Find current max
                current_max = max(row)
                if current_max < score:
                    continue
                
                # Try to find a new value for the last position that's smaller
                # and doesn't create conflicts
                row_without_last = row[:-1]
                diffs_without_last = get_diffs(row_without_last)
                
                # The new value must be > row_without_last[-1]
                # and its differences with all other elements must not be in all_diffs - diffs_without_last
                forbidden = all_diffs - diffs_without_last
                
                # Try values from row_without_last[-1]+1 to last-1
                for new_val in range(row_without_last[-1] + 1, last):
                    new_diffs = {new_val - a for a in row_without_last}
                    if new_diffs & forbidden:
                        continue
                    if len(new_diffs) != len(new_diffs):
                        continue
                    # Check for internal duplicates
                    if len(new_diffs) != len(row_without_last):
                        continue
                    
                    # This move is valid
                    # Update all_diffs
                    all_diffs -= diffs_without_last | {last - a for a in row_without_last}
                    all_diffs |= new_diffs
                    row[-1] = new_val
                    rows[idx] = row
                    
                    new_score = max_entry(rows)
                    if new_score < score:
                        score = new_score
                        improved = True
                        break
        
        return rows, score
    
    # Run local improvement
    best_rows, best_score = try_improve(best_rows, deadline)
    
    # Try randomized restarts if time permits
    if time.time() < deadline - 20:
        for attempt in range(50):
            if time.time() >= deadline - 10:
                break
            
            # Randomized greedy
            used = set()
            rows = []
            valid = True
            for _ in range(n_rows):
                row = [0]
                x = 0
                # Add some randomness
                random_offset = random.randint(0, 2)
                while len(row) < k + 1:
                    x += 1
                    if random_offset and x % random_offset == 0:
                        x += random.randint(0, 3)
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                rows.append(row)
            
            if is_valid(rows):
                cur_score = max_entry(rows)
                if cur_score < best_score:
                    best_rows = [list(r) for r in rows]
                    best_score = cur_score
            
            # Try to improve this random solution
            if time.time() < deadline - 5:
                improved_rows, improved_score = try_improve(rows, deadline)
                if improved_score < best_score:
                    best_rows = improved_rows
                    best_score = improved_score
    
    return [list(r) for r in best_rows]
