import random
import copy
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    if n_rows == 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    # Total number of differences needed: n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on max value: need at least total_diffs distinct positive differences
    # The smallest possible max m must satisfy m*(m+1)/2 >= total_diffs
    # So m >= ceil((-1 + sqrt(1 + 8*total_diffs))/2)
    lb = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)
    
    # Try to construct solution with max value starting from a reasonable estimate
    # Start with greedy to get an initial solution, then improve
    
    def greedy_construct(n_rows, k, max_bound=None):
        """Greedy construction trying to stay within max_bound if given"""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if max_bound is not None and x > max_bound:
                    return None
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
        return rows
    
    # Get initial greedy solution
    rows = greedy_construct(n_rows, k)
    if rows is None:
        # Shouldn't happen, but fallback
        rows = [[0] + list(range(1, k+1)) for _ in range(n_rows)]
    
    # Compute current max
    current_max = max(max(r) for r in rows)
    
    # Build used_diffs set
    def compute_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    used_diffs = compute_used_diffs(rows)
    best_rows = copy.deepcopy(rows)
    best_max = current_max
    
    # Local search with simulated annealing
    # Moves: 
    # 1. Swap two elements in different rows (if they're in the same position index)
    # 2. Move an element to a different position within its row
    # 3. Shift all elements in a row by a constant (maintaining order)
    
    random.seed(42)
    T = 100.0
    cooling = 0.9995
    iterations = 200000
    no_improve = 0
    patience = 20000
    
    for it in range(iterations):
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.4:
            # Move: change value of a single element (with constraint that row stays increasing)
            ri = random.randint(0, n_rows - 1)
            ei = random.randint(1, k)  # Don't move the 0
            
            row = rows[ri]
            old_val = row[ei]
            
            # Compute old diffs
            old_diffs = set()
            for j in range(len(row)):
                if j != ei:
                    old_diffs.add(abs(old_val - row[j]))
            
            # Determine valid range for new value
            lower = row[ei-1] + 1
            upper = row[ei+1] - 1 if ei < k else best_max
            
            if old_val < lower or old_val > upper:
                continue
            
            # Try to find a new value
            # Bias towards values that create smaller max
            if random.random() < 0.7:
                # Try smaller values
                if old_val <= lower:
                    continue
                new_val = random.randint(lower, old_val - 1)
            else:
                # Try any value in range
                if upper <= lower:
                    continue
                new_val = random.randint(lower, upper)
            
            if new_val == old_val:
                continue
            
            # Check validity
            new_diffs = set()
            valid = True
            for j in range(len(row)):
                if j != ei:
                    d = abs(new_val - row[j])
                    if d in used_diffs and d not in old_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
            
            if not valid:
                continue
            
            # Check if new_diffs has internal conflicts (shouldn't, but verify)
            if len(new_diffs) != len([abs(new_val - row[j]) for j in range(len(row)) if j != ei]):
                continue
            
            # Apply move
            row[ei] = new_val
            used_diffs -= old_diffs
            used_diffs |= new_diffs
            
            new_max = max(max(r) for r in rows)
            
            # Accept/reject with SA
            if new_max <= current_max:
                # Always accept improvements
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_rows = copy.deepcopy(rows)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Accept with probability exp(-(new_max - current_max)/T)
                delta = new_max - current_max
                if random.random() < math.exp(-delta / T):
                    current_max = new_max
                    no_improve += 1
                else:
                    # Reject: undo
                    row[ei] = old_val
                    used_diffs -= new_diffs
                    used_diffs |= old_diffs
                    no_improve += 1
        
        elif move_type < 0.7:
            # Move: swap values at same index in two different rows
            ri1, ri2 = random.sample(range(n_rows), 2)
            ei = random.randint(1, k)
            
            row1, row2 = rows[ri1], rows[ri2]
            v1, v2 = row1[ei], row2[ei]
            
            if v1 == v2:
                continue
            
            # Check if swapping maintains increasing order in both rows
            # Row 1: need row1[ei-1] < v2 < row1[ei+1] (if ei < k)
            if ei > 0 and v2 <= row1[ei-1]:
                continue
            if ei < k and v2 >= row1[ei+1]:
                continue
            # Row 2: need row2[ei-1] < v1 < row2[ei+1] (if ei < k)
            if ei > 0 and v1 <= row2[ei-1]:
                continue
            if ei < k and v1 >= row2[ei+1]:
                continue
            
            # Compute old diffs for both rows at index ei
            old_diffs1 = set(abs(v1 - row1[j]) for j in range(len(row1)) if j != ei)
            old_diffs2 = set(abs(v2 - row2[j]) for j in range(len(row2)) if j != ei)
            
            # Compute new diffs
            new_diffs1 = set(abs(v2 - row1[j]) for j in range(len(row1)) if j != ei)
            new_diffs2 = set(abs(v1 - row2[j]) for j in range(len(row2)) if j != ei)
            
            # Check validity: new diffs must not conflict with used_diffs minus old_diffs
            other_diffs = used_diffs - old_diffs1 - old_diffs2
            if new_diffs1 & other_diffs or new_diffs2 & other_diffs:
                continue
            # Also check new_diffs1 and new_diffs2 don't conflict with each other
            if new_diffs1 & new_diffs2:
                continue
            
            # Apply swap
            row1[ei] = v2
            row2[ei] = v1
            used_diffs -= old_diffs1
            used_diffs -= old_diffs2
            used_diffs |= new_diffs1
            used_diffs |= new_diffs2
            
            new_max = max(max(r) for r in rows)
            
            if new_max <= current_max:
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_rows = copy.deepcopy(rows)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                delta = new_max - current_max
                if random.random() < math.exp(-delta / T):
                    current_max = new_max
                    no_improve += 1
                else:
                    # Undo
                    row1[ei] = v1
                    row2[ei] = v2
                    used_diffs -= new_diffs1
                    used_diffs -= new_diffs2
                    used_diffs |= old_diffs1
                    used_diffs |= old_diffs2
                    no_improve += 1
        
        else:
            # Move: shift an entire row by a constant (if it maintains increasing order and doesn't go negative)
            ri = random.randint(0, n_rows - 1)
            row = rows[ri]
            
            # Shift all elements except the first (which is 0) by delta
            # Actually, we can only shift if we keep 0 at position 0
            # So this move doesn't work directly. Instead, let's try to compress the row
            # by reducing gaps between consecutive elements
            
            # Find a pair of consecutive elements and try to reduce the gap
            if k < 1:
                continue
            ei = random.randint(1, k)
            
            old_val = row[ei]
            prev_val = row[ei-1]
            
            if old_val <= prev_val + 1:
                continue
            
            # Try to reduce the gap
            new_val = random.randint(prev_val + 1, old_val - 1)
            
            # Compute old diffs for this element
            old_diffs = set(abs(old_val - row[j]) for j in range(len(row)) if j != ei)
            
            # Compute new diffs
            new_diffs = set(abs(new_val - row[j]) for j in range(len(row)) if j != ei)
            
            # Check validity
            other_diffs = used_diffs - old_diffs
            if new_diffs & other_diffs:
                continue
            
            # Apply
            row[ei] = new_val
            used_diffs -= old_diffs
            used_diffs |= new_diffs
            
            new_max = max(max(r) for r in rows)
            
            if new_max <= current_max:
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_rows = copy.deepcopy(rows)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                delta = new_max - current_max
                if random.random() < math.exp(-delta / T):
                    current_max = new_max
                    no_improve += 1
                else:
                    # Undo
                    row[ei] = old_val
                    used_diffs -= new_diffs
                    used_diffs |= old_diffs
                    no_improve += 1
        
        # Cool down
        T *= cooling
        
        # Early stopping if no improvement for a long time
        if no_improve > patience and it > 50000:
            break
    
    return best_rows
