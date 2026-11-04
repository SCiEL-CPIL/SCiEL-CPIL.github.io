import time
import random
import sys
from collections import Counter

def construct(n_rows, k):
    deadline = time.time() + 145.0
    
    # Helper functions
    def get_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

    def max_entry(rows):
        return max(max(row) for row in rows)

    # Phase 1: Greedy Initialization
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy_init()
    best_max = max_entry(best_rows)
    
    # Phase 2: Local Search with Bitset for Fast Conflict Checking
    # Represent used differences as a bitset (integer) for O(1) conflict checks
    
    def rows_to_bitset(rows):
        bitset = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    bitset |= (1 << d)
        return bitset

    def is_valid(rows):
        # Check if all differences are distinct across all rows
        bitset = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if bitset & (1 << d):
                        return False
                    bitset |= (1 << d)
        return True

    # Local search: try to move individual elements to reduce max entry
    current_rows = [r[:] for r in best_rows]
    current_bitset = rows_to_bitset(current_rows)
    current_max = max_entry(current_rows)
    
    # Precompute all positions
    positions = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(positions)
        
        for r, i in positions:
            old_val = current_rows[r][i]
            row = current_rows[r]
            
            # Determine valid range
            lower = row[i-1] + 1
            upper = row[i+1] - 1 if i < k else old_val
            
            # Try values from lower to upper (prefer smaller values)
            for new_val in range(lower, upper + 1):
                if new_val == old_val:
                    continue
                
                # Check if new_val creates conflicts
                # Remove old differences involving position i
                old_diffs = []
                for j in range(len(row)):
                    if j == i:
                        continue
                    if j < i:
                        old_diffs.append(old_val - row[j])
                    else:
                        old_diffs.append(row[j] - old_val)
                
                # Build bitset without these differences
                temp_bitset = current_bitset
                for d in old_diffs:
                    temp_bitset &= ~(1 << d)
                
                # Check if new differences conflict
                conflict = False
                new_diffs = []
                for j in range(len(row)):
                    if j == i:
                        continue
                    if j < i:
                        new_d = new_val - row[j]
                    else:
                        new_d = row[j] - new_val
                    
                    if temp_bitset & (1 << new_d):
                        conflict = True
                        break
                    new_diffs.append(new_d)
                
                if conflict:
                    continue
                
                # Check if new differences are distinct from each other
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                
                # Check if new_max is better
                new_max = max(max(row) for row in current_rows)
                # Actually, we need to compute new_max properly
                new_row = row[:]
                new_row[i] = new_val
                new_max = max(max(row) if row_idx != r else max(new_row) for row_idx, row in enumerate(current_rows))
                
                if new_max < current_max:
                    # Accept this move
                    current_bitset = temp_bitset
                    for d in new_diffs:
                        current_bitset |= (1 << d)
                    current_rows[r][i] = new_val
                    current_max = new_max
                    improved = True
                    break
            else:
                continue
            break
    
    if is_valid(current_rows):
        best_rows = [r[:] for r in current_rows]
        best_max = current_max
    
    # Phase 3: Simulated Annealing to polish
    current_rows = [r[:] for r in best_rows]
    
    # Precompute diff counts
    diff_counts = Counter()
    for row in current_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diff_counts[row[j] - row[i]] += 1
                
    def get_violations():
        return sum(c - 1 for c in diff_counts.values() if c > 1)
        
    current_violations = get_violations()
    current_max = max_entry(current_rows)
    current_cost = current_violations * 10000 + current_max
    
    T = 50.0
    T_min = 0.01
    cooling = 0.99995
    
    indices = [(r, i) for r in range(n_rows) for i in range(1, k + 1)]
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        r, i = random.choice(indices)
        old_val = current_rows[r][i]
        
        lower = current_rows[r][i - 1] + 1
        upper = current_rows[r][i + 1] - 1 if i < k else current_rows[r][k] + 50
        
        if random.random() < 0.3:
            span = upper - lower + 1
            new_val = lower + int(random.random() * span * 0.5)
        else:
            new_val = random.randint(lower, upper)
            
        if new_val == old_val:
            continue
            
        if new_val <= lower or (i < k and new_val >= upper):
            continue
            
        row = current_rows[r]
        removed_diffs = []
        added_diffs = []
        
        for j in range(len(row)):
            if j == i:
                continue
            if j < i:
                old_d = old_val - row[j]
                new_d = new_val - row[j]
                removed_diffs.append(old_d)
                added_diffs.append(new_d)
            else:
                old_d = row[j] - old_val
                new_d = row[j] - new_val
                removed_diffs.append(old_d)
                added_diffs.append(new_d)
                
        for d in removed_diffs:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
        for d in added_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
            
        new_violations = get_violations()
        new_max = max(max(row) for row in current_rows)
        new_cost = new_violations * 10000 + new_max
        
        delta = new_cost - current_cost
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            current_rows[r][i] = new_val
            current_violations = new_violations
            current_max = new_max
            current_cost = new_cost
            
            if new_max < best_max:
                best_max = new_max
                best_rows = [rr[:] for rr in current_rows]
        else:
            for d in added_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            for d in removed_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            current_rows[r][i] = old_val
            
        T *= cooling
        
    return best_rows
