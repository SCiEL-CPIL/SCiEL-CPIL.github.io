import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    # Leave 5 seconds margin for safety
    deadline = start_time + 115.0
    
    row_len = k + 1
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    # Number of differences per row
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row

    def greedy_construct(randomize=False):
        """
        Greedy construction. If randomize, it picks from a small set of valid candidates 
        to create diverse initial states.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                candidates = []
                # Look ahead a bit to find valid candidates
                # Standard greedy just takes the first one.
                # Randomized greedy takes the first one but with some probability 
                # picks a slightly larger one if valid.
                found = False
                while not found:
                    x += 1
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                    if valid:
                        found = True
                        if randomize:
                            # Occasionally pick a larger valid number to spread out
                            # This helps break symmetry and potentially lower the max later
                            if random.random() < 0.1:
                                # Try to find a valid number a bit further out
                                # But keep it small to not blow up the scope
                                for _ in range(5):
                                    x += random.randint(0, 2)
                                    valid2 = True
                                    for a in row:
                                        d = x - a
                                        if d in used:
                                            valid2 = False
                                            break
                                    if valid2:
                                        break
            row.append(x)
            for a in row[:-1]:
                used.add(x - a)
            rows.append(row)
        return rows

    # Generate a few diverse baselines
    baselines = []
    baselines.append(greedy_construct(randomize=False))
    baselines.append(greedy_construct(randomize=True))
    baselines.append(greedy_construct(randomize=True))
    
    # Pick the one with smallest max val as the primary start
    best_baseline = min(baselines, key=lambda r: max(max(row) for row in r))
    
    class SAState:
        def __init__(self, rows):
            self.rows = [r[:] for r in rows]
            self.n_rows = len(rows)
            self.row_len = len(rows[0])
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            self._rebuild()
        
        def _rebuild(self):
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            for r in self.rows:
                if r[-1] > self.max_val:
                    self.max_val = r[-1]
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d = r[j] - r[i]
                        if d in self.diff_count:
                            self.diff_count[d] += 1
                            if self.diff_count[d] == 2:
                                self.collisions += 1
                        else:
                            self.diff_count[d] = 1
        
        def get_score(self):
            # We want valid solutions (collisions == 0) with minimum max_val.
            # If collisions > 0, we want to minimize collisions first.
            # Penalty should be large enough to prioritize validity.
            # Max possible max_val is roughly n_rows * k * something. 
            # Let's use a penalty of 10000 * n_rows * k to be safe.
            return self.collisions * 100000 + self.max_val

        def try_move(self, r_idx, i_idx, new_val):
            old_val = self.rows[r_idx][i_idx]
            if old_val == new_val:
                return False, old_val, 0
            
            # Check bounds
            if i_idx > 0 and new_val <= self.rows[r_idx][i_idx-1]:
                return False, old_val, 0
            if i_idx < self.row_len - 1 and new_val >= self.rows[r_idx][i_idx+1]:
                return False, old_val, 0
            
            old_row = self.rows[r_idx]
            
            # Remove old differences
            old_diffs = []
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(old_val - old_row[j])
                    old_diffs.append(d)
            
            old_score = self.get_score()
            
            for d in old_diffs:
                self.diff_count[d] -= 1
                if self.diff_count[d] == 1:
                    self.collisions -= 1
                if self.diff_count[d] == 0:
                    del self.diff_count[d]
            
            # Apply change
            self.rows[r_idx][i_idx] = new_val
            
            # Add new differences
            new_row = self.rows[r_idx]
            new_diffs = []
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(new_val - new_row[j])
                    new_diffs.append(d)
            
            for d in new_diffs:
                if d in self.diff_count:
                    self.diff_count[d] += 1
                    if self.diff_count[d] == 2:
                        self.collisions += 1
                else:
                    self.diff_count[d] = 1
            
            # Update max_val
            if new_val > self.max_val:
                self.max_val = new_val
            elif old_val == self.max_val and new_val < old_val:
                # Need to recompute max
                self.max_val = max(max(r) for r in self.rows)
            
            new_score = self.get_score()
            delta_score = new_score - old_score
            
            return True, old_val, delta_score

        def undo_move(self, r_idx, i_idx, old_val):
            new_val = self.rows[r_idx][i_idx]
            if new_val == old_val:
                return
            
            # Remove new differences
            new_row = self.rows[r_idx]
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(new_val - new_row[j])
                    self.diff_count[d] -= 1
                    if self.diff_count[d] == 1:
                        self.collisions -= 1
                    if self.diff_count[d] == 0:
                        del self.diff_count[d]
            
            # Restore old value
            self.rows[r_idx][i_idx] = old_val
            
            # Add old differences
            old_row = self.rows[r_idx]
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(old_val - old_row[j])
                    if d in self.diff_count:
                        self.diff_count[d] += 1
                        if self.diff_count[d] == 2:
                            self.collisions += 1
                    else:
                        self.diff_count[d] = 1
            
            # Update max_val
            if old_val > self.max_val:
                self.max_val = old_val
            elif new_val == self.max_val and old_val < new_val:
                self.max_val = max(max(r) for r in self.rows)

    def simulate_annealing(initial_rows, time_limit):
        state = SAState(initial_rows)
        current_score = state.get_score()
        best_score = current_score
        best_rows = [r[:] for r in state.rows]
        
        T = 100.0
        cooling = 0.9995
        start = time.time()
        last_improvement = start
        
        # Track best valid solution separately to ensure we return a valid one
        best_valid_score = float('inf')
        best_valid_rows = None
        
        if state.collisions == 0:
            best_valid_score = state.max_val
            best_valid_rows = [r[:] for r in state.rows]

        while time.time() - start < time_limit:
            # Check time
            if time.time() > deadline:
                break
                
            # Choose move type:
            # 1. Random element change (50%)
            # 2. Collision resolution (50% if collisions > 0)
            
            move_type = random.random()
            r_idx = -1
            i_idx = -1
            new_val = -1
            accepted = False
            old_val = 0
            delta = 0
            
            if state.collisions > 0 and move_type < 0.5:
                # Try to resolve a collision
                # Pick a random colliding difference
                colliding_diffs = [d for d, c in state.diff_count.items() if c > 1]
                if colliding_diffs:
                    d = random.choice(colliding_diffs)
                    # Find one of the rows using this difference
                    # This is expensive to do naively. 
                    # Approximation: Just pick a random row and element and hope it helps?
                    # Better: We can't easily map d back to (r, i, j) without storing it.
                    # For now, fall back to random move if collision resolution is too complex to implement fast.
                    # Let's stick to random moves but with a wider range.
                    pass
            
            # Default: Random move
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)
            
            lower = state.rows[r_idx][i_idx - 1] + 1
            if i_idx < row_len - 1:
                upper = state.rows[r_idx][i_idx + 1] - 1
            else:
                # Allow expansion for the last element
                upper = state.max_val + 50 + random.randint(0, 50)
                
            if lower <= upper:
                # Bias towards smaller values to minimize scope
                # But allow some randomness
                # new_val = random.randint(lower, upper)
                # Weighted random
                if random.random() < 0.7:
                    new_val = random.randint(lower, min(upper, lower + 10))
                else:
                    new_val = random.randint(lower, upper)
                
                accepted, old_val, delta = state.try_move(r_idx, i_idx, new_val)
            
            if not accepted:
                continue
            
            # Metropolis criterion
            if delta <= 0:
                # Accept
                if state.collisions == 0:
                    if state.max_val < best_valid_score:
                        best_valid_score = state.max_val
                        best_valid_rows = [r[:] for r in state.rows]
                        last_improvement = time.time()
                if state.get_score() < best_score:
                    best_score = state.get_score()
                    best_rows = [r[:] for r in state.rows]
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < math.exp(-delta / T):
                    # Accept
                    if state.collisions == 0:
                        if state.max_val < best_valid_score:
                            best_valid_score = state.max_val
                            best_valid_rows = [r[:] for r in state.rows]
                            last_improvement = time.time()
                else:
                    # Reject
                    state.undo_move(r_idx, i_idx, old_val)
            
            T *= cooling
            if T < 0.1:
                # Restart or cool down faster?
                # If stuck, restart T
                if time.time() - last_improvement > 5.0:
                    T = 50.0
                    # Maybe perturb the state?
                    # For now, just reset T.

        # Return the best valid solution found. 
        # If no valid solution was found during SA, return the best_rows if it's valid, 
        # otherwise return the initial baseline (which is valid).
        if best_valid_rows is not None:
            return best_valid_rows
        else:
            # Check if best_rows is valid
            def is_valid(rows):
                used = set()
                for r in rows:
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            d = r[j] - r[i]
                            if d in used:
                                return False
                            used.add(d)
                return True
            
            if is_valid(best_rows):
                return best_rows
            else:
                return initial_rows

    # Run SA on the best baseline
    time_left = deadline - time.time()
    if time_left > 1.0:
        sa_rows = simulate_annealing(best_baseline, time_left)
    else:
        sa_rows = best_baseline
    
    return sa_rows
