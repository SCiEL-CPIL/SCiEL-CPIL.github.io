import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    
    Strategy:
    1. Calculate a theoretical lower bound M_min.
    2. Initialize a solution with M = M_min (or slightly higher).
       Each row is a set of k+1 integers.
    3. Use Simulated Annealing to minimize the "scope" (max entry) 
       while maintaining validity (all differences distinct).
       Moves: 
       - Change a single element in a row (if it maintains increasing order and distinct diffs).
       - Swap two elements between different rows (if valid).
       - Add/Remove small perturbations.
       
    Actually, a simpler and more effective approach for this specific "difference triangle" 
    structure is to treat the rows as independent but coupled by the global difference set.
    
    Let's use a Randomized Local Search (Hill Climbing with Restart) on the rows themselves.
    Objective: Minimize max(row)
    Constraint: All pairwise differences across all rows are unique.
    
    Since the constraint is "global distinctness", we can define a penalty function:
    Penalty = sum of squares of duplicate differences + lambda * (max_entry - target_M)
    
    However, we need a VALID solution. 
    Better: 
    1. Start with a valid solution for a larger M (e.g., greedy construction with M=200).
    2. Use Local Search to reduce M.
       - Try to reduce the largest element in the row that has the maximum.
       - If we reduce an element, we must ensure no new difference conflicts arise.
       - If a conflict arises, we might need to swap values with other rows.
    
    Let's implement a "Conflict-Driven Local Search":
    1. Generate an initial valid solution using a greedy algorithm with a large M.
    2. Identify the row with the maximum element, let it be row_r with max val m.
    3. Try to reduce m by 1.
       - This changes differences involving m.
       - Check if new differences conflict with existing global diffs.
       - If conflict, try to fix by modifying other rows or swapping.
    
    A more robust method:
    Use Simulated Annealing on the vector of all row elements (excluding the 0s).
    State: List of n_rows lists of k elements (the non-zero parts).
    Validity: All differences distinct.
    Cost: Max element in state.
    
    Moves:
    - Pick a random row r and index i.
    - Change the value v to v' (v' != v, v' > 0, and row remains increasing).
    - If the move creates a difference conflict, reject it (or accept with low prob if we use penalty).
    - To strictly maintain validity, we can use a "repair" strategy or just reject invalid moves.
    - If we only accept valid moves, we might get stuck.
    
    Alternative: 
    Since the problem size is small (7 rows, 6 elements -> 42 elements, 21 diffs/row -> 147 total diffs),
    we can use a CSP solver with backtracking but with a much smarter heuristic: 
    "Minimum Remaining Values" on the differences themselves?
    
    Let's try a different angle: 
    The differences must be distinct. There are N = n_rows * k*(k+1)/2 differences.
    They must be chosen from {1, ..., M}.
    So M >= N.
    
    For (7, 5), N = 7 * 15 = 105.
    Lower bound M >= 105.
    
    We can model this as:
    Find a set of 147 distinct integers D = {d_1, ..., d_147} such that they can be partitioned into 7 groups of 15, 
    where each group corresponds to the differences of a row of 6 numbers.
    
    A row of 6 numbers (0, x1, x2, x3, x4, x5) has differences:
    x1, x2, x3, x4, x5,
    x2-x1, x3-x1, x4-x1, x5-x1,
    x3-x2, x4-x2, x5-x2,
    x4-x3, x5-x3,
    x5-x4.
    
    This is a complex dependency.
    
    Let's stick to the Row-based Local Search but improve the move set.
    1. Start with a valid solution for M=120 (easy to find with greedy).
    2. Iteratively try to reduce the max element.
    3. If we reduce an element, we create "holes" in the difference set and "conflicts" if the new diffs exist elsewhere.
    
    Actually, the most effective simple strategy for these small combinatorial problems is often 
    **Randomized Restarted Backtracking with a strong heuristic**.
    The previous program did this but maybe the heuristic was weak or the search space was explored poorly.
    
    Let's try **Simulated Annealing on the Rows** with a penalty for invalid differences.
    We allow invalid states during the search but penalize them heavily.
    Objective: Minimize (Max_Element + Penalty * Invalid_Count).
    If we reach Invalid_Count == 0, we have a valid solution.
    Then we continue to minimize Max_Element.
    
    Moves:
    - Pick a random row r.
    - Pick a random index i in row r.
    - Change row[r][i] by +/- delta (random small int).
    - Ensure row[r] remains increasing (sort after move? No, that changes identity. 
      Better: pick a new value for the whole row? Or just one element and repair sorting?).
      
    If we change one element, the row might not be sorted. 
    We can represent the row as a SET of k+1 numbers including 0.
    Move: Remove a number x from row r, add a number y to row r.
    Constraint: y not in row r, y > 0.
    After move, sort the row to maintain canonical form.
    
    This allows any permutation of elements within a row, which is fine since differences are invariant under permutation? 
    No, differences depend on the values, not the order. The set of differences is determined by the set of numbers.
    So representing rows as sets is correct.
    
    Algorithm:
    1. Initialize n_rows sets, each with 0 and k random numbers in [1, M_init].
    2. Calculate current cost: Max(union of all numbers) + Lambda * (number of duplicate differences).
    3. Simulated Annealing loop:
       a. Generate neighbor: Pick random row, remove random non-zero element, add random new non-zero element.
       b. Calculate delta cost.
       c. Accept if better, or with probability exp(-delta_cost/T).
       d. Cool T.
    4. Once we find a valid solution (0 duplicates), we can try to reduce M by restricting the domain of new elements to [1, M_current - 1] or similar.
    
    This is a structural change from backtracking to global optimization with soft constraints.
    """
    
    start_time = time.time()
    DEADLINE = 140.0
    
    num_elements = k + 1
    num_pairs_per_row = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs_per_row
    
    # Lower bound
    lower_bound = total_diffs_needed
    
    # Helper: Calculate all differences for a row (list of sorted ints)
    def get_row_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    # Helper: Count duplicates in a list of diffs
    def count_duplicates(diffs):
        if not diffs:
            return 0
        s = set(diffs)
        return len(diffs) - len(s)

    # Helper: Get max element in rows
    def get_max_element(rows):
        return max(max(row) for row in rows)

    # Initialize rows
    def init_rows(M_init, rng):
        rows = []
        used_diffs_global = set() # Not strictly needed for init if we allow duplicates, but good for heuristic init
        for _ in range(n_rows):
            row = [0]
            # Greedy-ish init to get close to valid
            for _ in range(k):
                # Pick random from 1..M_init
                while True:
                    x = rng.randint(1, M_init)
                    # Check if x > last
                    if x > row[-1]:
                        # Check local diffs
                        new_diffs = [x - y for y in row]
                        # We don't enforce global uniqueness in init to allow SA to find valid states
                        row.append(x)
                        break
            rows.append(row)
        return rows

    # Simulated Annealing
    def simulated_annealing(M_target, rng):
        # Start with a random valid-ish configuration
        # To help SA, start with a greedy solution for M_target + 10
        M_start = M_target + 20
        rows = init_rows(M_start, rng)
        
        # Precompute initial diffs
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_row_diffs(r))
        current_dups = count_duplicates(all_diffs)
        current_max = get_max_element(rows)
        current_cost = current_max + 100 * current_dups # Heavy penalty for duplicates
        
        best_rows = [r[:] for r in rows]
        best_cost = current_cost
        
        T = 10.0
        cooling = 0.9995
        min_T = 0.01
        
        steps = 0
        max_steps = 50000 # Adjust based on time
        
        while time.time() < DEADLINE - 5:
            # Generate neighbor
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]
            
            # Pick an element to change (not the 0)
            # We will replace one element with another
            if len(row) <= 1:
                continue
                
            elem_idx = rng.randint(1, len(row) - 1) # Index 0 is 0
            old_val = row[elem_idx]
            
            # Generate new value
            # Bias towards smaller values to reduce max
            # Range: 1 to M_target (or slightly more to allow exploration)
            # If we restrict to M_target, we might get stuck. 
            # Let's allow values up to M_target + 5 for flexibility, but penalize > M_target.
            
            # Simple move: change old_val to new_val
            # Ensure uniqueness in row
            new_val = rng.randint(1, M_target + 10)
            if new_val in row:
                continue
            
            # Apply move
            row[elem_idx] = new_val
            row.sort() # Maintain canonical form
            
            # Recalculate cost for this row and update global
            # This is expensive O(k^2) per move.
            # Optimization: Only recalculate diffs for the changed row and check against global set?
            # For simplicity and correctness first, recalculate all.
            # With 7 rows, it's fast enough.
            
            all_diffs_new = []
            for r in rows:
                all_diffs_new.extend(get_row_diffs(r))
            
            new_dups = count_duplicates(all_diffs_new)
            new_max = get_max_element(rows)
            
            # Cost function:
            # We want to minimize max. 
            # Penalty for duplicates.
            # Penalty for exceeding M_target.
            excess = max(0, new_max - M_target)
            new_cost = new_max + 100 * new_dups + 10 * excess
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or rng.random() < pow(2.71828, -delta_cost / T):
                # Accept
                current_cost = new_cost
                current_dups = new_dups
                current_max = new_max
                
                if new_dups == 0:
                    # Valid solution
                    if new_max < best_cost:
                        best_cost = new_max
                        best_rows = [r[:] for r in rows]
                        # If we found a valid solution, we can try to tighten M_target?
                        # Or just let SA continue to reduce max.
                        pass
            else:
                # Revert
                row[elem_idx] = old_val
                row.sort()
            
            T *= cooling
            steps += 1
            
            if steps > max_steps:
                break
        
        return best_rows, best_cost

    # Main logic
    best_solution = None
    best_M = float('inf')
    
    # Try M from lower_bound upwards
    # Since SA is stochastic, we might find a solution for M_lower later.
    # We will run SA for a fixed budget for each M.
    
    # To save time, we can run SA once with a high M, then try to reduce.
    # Or run multiple SAs with different M targets.
    
    # Strategy:
    # 1. Run SA with M_target = lower_bound.
    # 2. If valid solution found, record it.
    # 3. If not, try M_target = lower_bound + 1, etc.
    
    # But SA might be slow to converge.
    # Let's do a hybrid:
    # 1. Greedy construct for M = lower_bound + 10 (should be fast and likely valid).
    # 2. Use Local Search to reduce M.
    
    # Let's implement the Local Search on a valid solution.
    
    def greedy_construct(M, rng):
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                found = False
                # Try values from row[-1]+1 to M
                # Randomize order to avoid bias
                candidates = list(range(row[-1] + 1, M + 1))
                rng.shuffle(candidates)
                
                for x in candidates:
                    new_diffs = [x - y for y in row]
                    if not any(d in used_diffs for d in new_diffs):
                        row.append(x)
                        used_diffs.update(new_diffs)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows

    # Local Search to reduce max
    def local_search_reduce(rows, M_current, rng):
        # rows is a valid solution with max <= M_current
        current_max = get_max_element(rows)
        
        # Identify the row with the max element
        # Try to reduce it
        
        max_reductions = 1000
        for _ in range(max_reductions):
            if time.time() > DEADLINE - 5:
                break
                
            # Find row with max element
            max_val = current_max
            # Find all rows containing max_val
            # We want to reduce the global max.
            
            # Pick a random row that has an element equal to current_max
            candidate_rows = [i for i, r in enumerate(rows) if current_max in r]
            if not candidate_rows:
                # Max must have decreased?
                current_max = get_max_element(rows)
                continue
                
            r_idx = rng.choice(candidate_rows)
            row = rows[r_idx]
            
            # Find index of max_val in row
            idx = row.index(current_max)
            
            # Try to replace current_max with a smaller value
            # New value must be > row[idx-1] (if idx>0) and < row[idx+1] (if idx<len-1)
            # And must not create duplicate differences.
            
            lower_bound_val = row[idx-1] + 1 if idx > 0 else 1
            upper_bound_val = row[idx+1] - 1 if idx < len(row)-1 else current_max - 1
            
            if lower_bound_val > upper_bound_val:
                continue
                
            # Try random values in range
            tried = set()
            for _ in range(50):
                if lower_bound_val > upper_bound_val:
                    break
                new_val = rng.randint(lower_bound_val, upper_bound_val)
                if new_val in tried:
                    continue
                tried.add(new_val)
                
                # Apply
                row[idx] = new_val
                row.sort()
                
                # Check validity
                all_diffs = []
                for r in rows:
                    all_diffs.extend(get_row_diffs(r))
                
                if len(all_diffs) == len(set(all_diffs)):
                    # Valid!
                    current_max = get_max_element(rows)
                    # Continue loop to reduce further
                    break
                else:
                    # Revert
                    row[idx] = current_max
                    row.sort()
        
        return rows, current_max

    # Execution
    # 1. Get a valid solution with Greedy for M = lower_bound + 15
    M_start = lower_bound + 15
    rng = random.Random(42)
    rows = greedy_construct(M_start, rng)
    
    if rows is None:
        # Try larger M
        M_start += 10
        rows = greedy_construct(M_start, rng)
        
    if rows is not None:
        # 2. Local Search to reduce
        rows, M_current = local_search_reduce(rows, M_start, rng)
        
        # 3. Try to find better with SA targeting lower M
        # Run SA for a few seconds targeting lower_bound
        # But SA is stochastic. Let's just return the best from Local Search.
        
        # Verify
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_row_diffs(r))
        if len(all_diffs) == len(set(all_diffs)):
            best_solution = rows
            best_M = M_current
        else:
            # Fallback to greedy
            best_solution = rows
            best_M = M_start
    else:
        # Emergency fallback
        rows_fb = []
        used = set()
        M = lower_bound + 50
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        break
                    x += 1
                row.append(x)
                used.update(new_diffs)
            rows_fb.append(row)
        best_solution = rows_fb
        best_M = get_max_element(rows_fb)

    return best_solution

# EVOLVE-BLOCK-END
