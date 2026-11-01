"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimizes the largest entry (scope).
    """
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Each row has k+1 entries, so k(k+1)/2 differences per row
    # Total differences needed: n_rows * k(k+1)/2
    # We need to find n_rows rows with distinct differences
    
    # Strategy: Use simulated annealing / local search to find compact difference sets
    # First, let's understand the structure:
    # A row [a_0, a_1, ..., a_k] with a_0=0 generates differences a_j - a_i for i<j
    # We need all these differences to be distinct across all rows
    
    # Key insight: if we can find n_rows "difference sets" of size k(k+1)/2 
    # that are pairwise disjoint, and each difference set can be realized by a valid ruler,
    # then we're done.
    
    # Let's use a different approach: directly search for the rows using backtracking
    # with pruning, and use simulated annealing for refinement.
    
    def get_all_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        seen = set()
        for row in rows:
            for d in get_all_diffs(row):
                if d in seen:
                    return False
                seen.add(d)
        return True
    
    def get_scope(rows):
        """Get the largest entry across all rows."""
        return max(max(row) for row in rows)
    
    # Greedy construction as baseline
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
    
    # Start with greedy
    rows = greedy_construct(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(best_rows)
    
    # Simulated annealing to improve
    def random_row(used_diffs, max_val):
        """Generate a random valid row given used differences."""
        row = [0]
        x = 0
        attempts = 0
        while len(row) < k + 1 and attempts < 1000:
            # Try to find a valid next element
            x += 1
            if x > max_val:
                break
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
            attempts += 1
        if len(row) == k + 1:
            return row, new_diffs
        return None, set()
    
    def get_used_diffs(rows):
        """Get all used differences from rows."""
        used = set()
        for row in rows:
            used |= get_all_diffs(row)
        return used
    
    # Simulated annealing
    current_rows = [r[:] for r in best_rows]
    current_scope = best_scope
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        # Pick a random row to modify
        row_idx = random.randint(0, n_rows - 1)
        
        # Remove this row's differences from used set
        old_diffs = get_all_diffs(current_rows[row_idx])
        other_diffs = get_used_diffs([r for i, r in enumerate(current_rows) if i != row_idx])
        
        # Try to build a new row
        new_row, new_diffs = random_row(other_diffs, current_scope)
        
        if new_row is not None:
            candidate_rows = [r[:] for r in current_rows]
            candidate_rows[row_idx] = new_row
            candidate_scope = get_scope(candidate_rows)
            
            # Accept if better or with some probability
            delta = candidate_scope - current_scope
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_rows = candidate_rows
                current_scope = candidate_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
        
        T *= cooling
    
    return best_rows
# EVOLVE-BLOCK-END
