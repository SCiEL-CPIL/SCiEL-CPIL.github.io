import time
import random
import math

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences are distinct across all rows.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Helper to compute all differences for a row
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper to check validity and get scope
    def evaluate(rows):
        all_diffs = set()
        max_val = 0
        for row in rows:
            max_val = max(max_val, row[-1])
            row_diffs = get_row_diffs(row)
            if all_diffs & row_diffs:
                # Collision found
                # Count collisions for penalty
                collisions = len(all_diffs & row_diffs)
                return max_val + 1000 * collisions, False
            all_diffs |= row_diffs
        return max_val, True

    # Greedy initialization with some randomness
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Add some randomness to the step
                step = random.randint(1, 3)
                x += step
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                else:
                    # If collision, try incrementing by 1
                    x -= step
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
            rows.append(row)
        return rows

    # Initial solution
    best_rows = greedy_init()
    best_score, _ = evaluate(best_rows)
    current_rows = [r[:] for r in best_rows]
    current_score, _ = evaluate(current_rows)

    # Simulated Annealing
    T = 50.0  # Initial temperature
    cooling_rate = 0.9999

    while time.time() < deadline:
        # Choose a random row and a random mark to move
        r_idx = random.randint(0, n_rows - 1)
        m_idx = random.randint(1, k)  # Don't move the first mark (0)
        
        row = current_rows[r_idx]
        old_val = row[m_idx]
        
        # Determine valid range for new value
        min_val = row[m_idx - 1] + 1
        max_val = row[m_idx + 1] - 1 if m_idx < k else old_val + 10  # Allow some increase
        
        if max_val < min_val:
            continue
            
        # Choose a new value
        # Bias towards smaller values to reduce scope
        new_val = random.randint(min_val, max_val)
        
        if new_val == old_val:
            continue

        # Tentatively move
        row[m_idx] = new_val
        
        new_score, new_valid = evaluate(current_rows)
        
        # Accept or reject
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_rows = [r[:] for r in current_rows]
        else:
            # Accept with probability exp(-(delta)/T)
            delta = new_score - current_score
            if delta > 0 and random.random() < math.exp(-delta / T):
                current_score = new_score
            else:
                # Revert
                row[m_idx] = old_val
        
        # Cool down
        T *= cooling_rate
        if T < 1.0:
            T = 1.0

    return best_rows
