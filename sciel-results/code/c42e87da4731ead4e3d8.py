"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    """
    Constructs a set of difference triangles (rulers) with minimized largest entry.
    Uses Simulated Annealing to improve upon a greedy initialization.
    """
    start_time = time.time()
    # Leave 10 seconds buffer for safety
    deadline = start_time + 110.0
    
    num_marks = k + 1
    
    def get_cost(rows):
        """
        Calculates the cost of the configuration.
        Cost = max_value + penalty * number_of_duplicate_differences.
        """
        diffs = []
        max_val = 0
        for row in rows:
            # Row is assumed to be sorted
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    diffs.append(d)
                    if row[j] > max_val:
                        max_val = row[j]
        
        # Count duplicates
        # If all differences are distinct, len(set(diffs)) == len(diffs)
        unique_count = len(set(diffs))
        duplicates = len(diffs) - unique_count
        
        # Large penalty to prioritize valid solutions
        return max_val + duplicates * 10000

    def greedy_init():
        """
        Generates a valid initial solution using a greedy approach.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_marks:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Initialize with greedy solution
    current_rows = greedy_init()
    best_rows = [r[:] for r in current_rows]
    
    # Calculate initial scores
    current_score = get_cost(current_rows)
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 100.0
    T_min = 0.01
    cooling = 0.999
    
    # Main SA Loop
    while time.time() < deadline:
        # 1. Generate Move
        i = random.randint(0, n_rows - 1)
        # Do not change the first element (0)
        j = random.randint(1, num_marks - 1)
        
        # Determine new value
        # We sample from the current maximum plus a small noise to allow expansion
        # but bias towards the current range to encourage shrinking
        curr_max = max(max(r) for r in current_rows)
        # If curr_max is small, we might want to expand a bit
        # If curr_max is large, we stay in range
        v = random.randint(1, curr_max + 20)
        
        # Apply Move
        old_val = current_rows[i][j]
        current_rows[i][j] = v
        current_rows[i].sort()
        
        # 2. Evaluate New State
        new_score = get_cost(current_rows)
        
        # 3. Acceptance Criterion
        delta = new_score - current_score
        
        accept = False
        if delta < 0:
            accept = True
        else:
            # Accept with probability exp(-delta / T)
            # Use a small epsilon to avoid division by zero or overflow
            if T > 1e-6:
                # If delta is very large, exp is 0
                # If delta is small, exp is close to 1
                prob = pow(2.71828, -delta / T)
                accept = random.random() < prob
        
        if accept:
            current_score = new_score
            # Update best if better
            if new_score < best_score:
                best_score = new_score
                best_rows = [r[:] for r in current_rows]
        else:
            # Reject Move: Restore state
            current_rows[i][j] = old_val
            current_rows[i].sort()
            
        # 4. Cooling
        T *= cooling
        if T < T_min:
            T = T_min

    # Return the best valid solution found
    # The problem requires a valid solution. 
    # Our best_score includes a penalty for invalidity.
    # If best_score has no penalty (i.e., duplicates=0), it is valid.
    # If for some reason we didn't find a valid one (unlikely given greedy init), 
    # we should return the greedy one or the best found.
    # Since greedy_init returns a valid solution, best_rows is at least as good as that.
    
    return best_rows

# EVOLVE-BLOCK-END
