import random
import time
import math

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows of k+1 integers.
    Minimizes the largest entry using Simulated Annealing for global optimization.
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Pre-compute max possible differences? No, range is dynamic.
    
    def get_scope(rows):
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        all_diffs = set()
        for row in rows:
            row_diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in row_diffs or d in all_diffs:
                        return False
                    row_diffs.add(d)
            all_diffs.update(row_diffs)
        return True

    def greedy_init():
        """Generate an initial valid set using greedy method."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        break
                    x += 1
                row.append(x)
                used_diffs.update({x - a for a in row[:-1]})
            rows.append(row)
        return rows

    def random_move(rows, used_diffs):
        """
        Perform a random mutation on the rows.
        Returns (new_rows, new_used_diffs, success)
        """
        # Strategy: Pick a random row and try to modify one of its elements
        # to be smaller, or swap elements within the row to allow smaller values.
        # Simpler: Pick a random row, remove it, and re-generate it greedily 
        # but with a bias towards smaller numbers, or just re-generate randomly 
        # within the current scope.
        
        # Better for SA: Pick a random row, pick a random element (not 0), 
        # and try to replace it with a smaller valid number.
        
        idx = random.randint(0, n_rows - 1)
        row = rows[idx]
        
        # We can only change elements at position > 0
        if len(row) <= 1:
            return rows, used_diffs, False
            
        pos = random.randint(1, len(row) - 1)
        old_val = row[pos]
        
        # Determine range for new value
        # Must be > row[pos-1] and < row[pos+1] (if exists)
        lower = row[pos-1] + 1
        upper = old_val - 1
        
        if lower > upper:
            return rows, used_diffs, False
            
        # Try a few random candidates in [lower, upper]
        # To make it efficient, try the smallest valid one first? 
        # Or random? Random is better for SA to explore.
        
        # First, remove old diffs from used_diffs
        old_diffs = set()
        for j in range(len(row)):
            if j != pos:
                d = abs(row[j] - old_val)
                old_diffs.add(d)
        
        # Check if old_diffs are actually in used_diffs? 
        # We maintain used_diffs as the global set.
        # We need to temporarily remove them to check new ones.
        
        # This is getting complex to manage state efficiently in SA.
        # Let's use a simpler "Re-generate one row" move.
        # Remove row idx from used_diffs.
        
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                row_diffs.add(row[j] - row[i])
        
        # Remove from global
        new_used_diffs = used_diffs.copy()
        for d in row_diffs:
            new_used_diffs.discard(d)
            
        # Generate new row
        new_row = [0]
        # Limit the search to slightly below the current global max to encourage compression
        current_max = max(max(r) for r in rows)
        # Allow going up to current_max + 10 to allow escape
        limit = current_max + 10
        
        valid = True
        for _ in range(k):
            candidates = []
            x = new_row[-1] + 1
            # Scan for a few candidates
            while x < limit and len(candidates) < 5:
                new_diffs = {x - a for a in new_row}
                if not (new_diffs & new_used_diffs):
                    candidates.append(x)
                x += 1
            
            if not candidates:
                # If we can't find a candidate within limit, this move fails
                valid = False
                break
                
            # Pick randomly from candidates to introduce diversity
            chosen = random.choice(candidates)
            new_row.append(chosen)
            new_used_diffs.update({chosen - a for a in new_row[:-1]})
            
        if not valid:
            return rows, used_diffs, False
            
        # Check if the new row is strictly increasing (it is by construction)
        # Check if the new scope is acceptable (SA handles this)
        
        new_rows = rows.copy()
        new_rows[idx] = new_row
        
        return new_rows, new_used_diffs, True

    def solve():
        # Initialize
        rows = greedy_init()
        # Compute initial used_diffs
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used_diffs.add(row[j] - row[i])
                    
        best_rows = rows[:]
        best_scope = get_scope(rows)
        
        # Simulated Annealing Parameters
        T = 10.0
        T_min = 0.1
        cooling = 0.995
        max_iters = 200000 # Adjust based on time
        
        current_scope = best_scope
        
        for i in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Generate move
            new_rows, new_used_diffs, success = random_move(rows, used_diffs)
            
            if not success:
                continue
                
            new_scope = get_scope(new_rows)
            
            # Calculate delta
            delta = new_scope - current_scope
            
            # Metropolis criterion
            if delta < 0 or (T > T_min and random.random() < math.exp(-delta / T)):
                # Accept
                rows = new_rows
                used_diffs = new_used_diffs
                current_scope = new_scope
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = rows[:]
                    
            # Cool down
            T *= cooling
            
            # Early stop if we hit the bound
            if best_scope <= 105:
                break
                
        return best_rows

    return solve()
