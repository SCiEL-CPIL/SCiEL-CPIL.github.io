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

    def greedy_construct(randomize=False, seed=None):
        """
        Fast greedy construction.
        """
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                found = False
                # Small random jump to avoid getting stuck in local minima of greedy
                if randomize:
                    x += random.randint(0, 2)
                else:
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
                        if random.random() < 0.2:
                            for _ in range(3):
                                x += random.randint(1, 3)
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

    # Generate a few diverse baselines quickly
    baselines = []
    for i in range(3):
        baselines.append(greedy_construct(randomize=True, seed=i))
    
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
            # Prioritize validity, then minimize max_val
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
                # Recompute max only if the global max was changed
                self.max_val = max(max(r) for r in self.rows)
            
            return True, old_val, 0 # Delta calculation is done outside or simplified

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
        
        T = 50.0
        cooling = 0.999
        start = time.time()
        last_improvement = start
        
        best_valid_score = float('inf')
        best_valid_rows = None
        
        if state.collisions == 0:
            best_valid_score = state.max_val
            best_valid_rows = [r[:] for r in state.rows]

        while time.time() - start < time_limit:
            if time.time() > deadline:
                break
                
            # Choose move
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)
            
            lower = state.rows[r_idx][i_idx - 1] + 1
            if i_idx < row_len - 1:
                upper = state.rows[r_idx][i_idx + 1] - 1
            else:
                # Allow expansion for the last element, but keep it tight
                upper = state.max_val + 10 + random.randint(0, 10)
                
            if lower <= upper:
                # Bias towards smaller values
                if random.random() < 0.8:
                    new_val = random.randint(lower, min(upper, lower + 5))
                else:
                    new_val = random.randint(lower, upper)
                
                old_val = state.rows[r_idx][i_idx]
                
                # Quick check before full try_move
                if new_val == old_val:
                    continue
                
                # Perform move
                state.rows[r_idx][i_idx] = new_val
                
                # Recalculate score components efficiently
                # We need to update diff_count and collisions
                # For simplicity and speed, we will use a "delta" approach in a real prod,
                # but here we rely on the fact that row_len is small (6).
                
                # Actually, let's stick to the try_move/undo pattern for correctness
                # but optimize the loop.
                
                accepted, old_val_ret, _ = state.try_move(r_idx, i_idx, new_val)
                if not accepted:
                    continue
                
                new_score = state.get_score()
                delta = new_score - current_score
                
                if delta <= 0:
                    current_score = new_score
                    if state.collisions == 0:
                        if state.max_val < best_valid_score:
                            best_valid_score = state.max_val
                            best_valid_rows = [r[:] for r in state.rows]
                            last_improvement = time.time()
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [r[:] for r in state.rows]
                else:
                    # Metropolis
                    if random.random() < math.exp(-delta / T):
                        current_score = new_score
                        if state.collisions == 0:
                            if state.max_val < best_valid_score:
                                best_valid_score = state.max_val
                                best_valid_rows = [r[:] for r in state.rows]
                                last_improvement = time.time()
                    else:
                        state.undo_move(r_idx, i_idx, old_val_ret)
            
            T *= cooling
            if T < 1.0:
                if time.time() - last_improvement > 2.0:
                    T = 20.0

        if best_valid_rows is not None:
            return best_valid_rows
        else:
            # Return best found, hoping it's valid or close
            return best_rows

    # Run SA on the best baseline
    time_left = deadline - time.time()
    if time_left > 1.0:
        sa_rows = simulate_annealing(best_baseline, time_left)
    else:
        sa_rows = best_baseline
    
    return sa_rows
