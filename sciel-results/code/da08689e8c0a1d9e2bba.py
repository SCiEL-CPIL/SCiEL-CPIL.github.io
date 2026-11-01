import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows and k differences (k+1 elements per row).
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit

    # Fallback: Greedy solution (always valid)
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

    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

    # Lower bound: total number of distinct differences needed
    # Each row has k*(k+1)/2 differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed  # scope must be at least this since differences are 1..scope
    
    # For (7,5): total_diffs = 7 * 15 = 105, so lb = 105
    # Known best is 112

    # We'll use a randomized local search / simulated annealing approach
    # combined with a greedy initialization and local improvement.
    
    # First, try to find a solution with scope close to lb using a smarter search.
    # We'll use a randomized greedy with backtracking and restarts.
    
    def try_find_solution(target_scope, num_restarts=50):
        """Try to find a valid DTS with all entries <= target_scope."""
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
            
            random.seed(restart * 1000 + int(time.time() * 1000) % 10000)
            
            # Randomized greedy construction
            used = set()
            rows = []
            success = True
            
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    if x > target_scope:
                        success = False
                        break
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                if not success:
                    break
                rows.append(row)
            
            if success and len(rows) == n_rows:
                # Validate
                all_diffs = set()
                valid = True
                for row in rows:
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                    if not valid:
                        break
                if valid:
                    return rows
            
            # If greedy failed, try a more careful backtracking for this restart
            # Use a simple backtracking with randomized ordering
            if time.time() > deadline:
                return None
            
            def backtrack_row(row_idx, current_rows, used_set):
                if row_idx == n_rows:
                    return list(current_rows)
                
                if time.time() > deadline:
                    return None
                
                # Try to build row row_idx
                # Use randomized candidate ordering
                def build_single_row(row_pos, current_row, used_local):
                    if row_pos == k + 1:
                        # Check if we can add this row
                        new_diffs = set()
                        for i in range(len(current_row)):
                            for j in range(i + 1, len(current_row)):
                                d = current_row[j] - current_row[i]
                                if d in used_local:
                                    return None
                                new_diffs.add(d)
                        
                        new_used = used_local | new_diffs
                        result = backtrack_row(row_idx + 1, current_rows + [current_row], new_used)
                        return result
                    
                    last_val = current_row[-1] if current_row else -1
                    remaining = k + 1 - row_pos
                    
                    # Candidate values
                    candidates = []
                    for v in range(last_val + 1, target_scope + 1):
                        # Check if v can be added
                        new_diffs = {v - a for a in current_row}
                        if new_diffs & used_local:
                            continue
                        # Check if we can still fill remaining slots
                        if target_scope - v < remaining - 1:
                            break
                        candidates.append(v)
                    
                    # Randomize order
                    random.shuffle(candidates)
                    
                    for v in candidates:
                        new_diffs = {v - a for a in current_row}
                        current_row.append(v)
                        result = build_single_row(row_pos + 1, current_row, used_local | new_diffs)
                        current_row.pop()
                        if result is not None:
                            return result
                    
                    return None
                
                result = build_single_row(1, [0], used_set)
                return result
            
            result = backtrack_row(0, [], set())
            if result is not None:
                return result
        
        return None

    # Try scopes from lower bound upwards
    for target_scope in range(lb, lb + 30):
        if time.time() > deadline:
            break
        
        # First try the simple greedy with a few restarts
        solution = try_find_solution(target_scope, num_restarts=20)
        
        if solution is not None:
            # Validate
            all_diffs = set()
            valid = True
            for row in solution:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
            
            if valid:
                actual_scope = max(max(row) for row in solution)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_solution = solution
                break
    
    # If we still have time, try to improve the best solution found
    # using local search / simulated annealing
    if time.time() < deadline - 10 and best_scope > lb:
        # Try to reduce the scope further using local search
        # This is a simplified approach: try to modify the solution to reduce max entry
        
        # We'll use a simple approach: try to find a better solution with scope best_scope - 1
        for target_scope in range(best_scope - 1, lb - 1, -1):
            if time.time() > deadline - 5:
                break
            
            solution = try_find_solution(target_scope, num_restarts=10)
            if solution is not None:
                # Validate
                all_diffs = set()
                valid = True
                for row in solution:
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                    if not valid:
                        break
                
                if valid:
                    actual_scope = max(max(row) for row in solution)
                    if actual_scope < best_scope:
                        best_scope = actual_scope
                        best_solution = solution
                    break
    
    return best_solution
# EVOLVE-BLOCK-END
