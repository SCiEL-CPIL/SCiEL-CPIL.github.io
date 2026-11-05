"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy:
# The current program uses exact DP to determine the optimal set of items for the remaining capacity.
# It then backtracks to find *one* item from that optimal set to return.
# The issue with the simple backtrack in the current code is that it returns the *last* item 
# (in the DP order) that is part of the optimal solution. This might not be the best "first" 
# item to pick in terms of maintaining flexibility for subsequent steps, although since we re-solve
# DP at each step, any item from the optimal set should lead to the global optimum eventually.
#
# However, there is a subtle risk: if there are multiple optimal solutions, the backtrack might
# pick an item that, while part of *an* optimal solution, leaves a residual capacity that is
# awkward for the *next* greedy/DP step? No, if we solve DP exactly at every step, we are
# effectively solving the problem from scratch each time. 
#
# Wait, the evaluator calls `select_next_item` repeatedly.
# Step 1: We have items 0..99. We solve DP. We pick item `i`.
# Step 2: Evaluator removes `i`. We have items 0..99 \ {i}. We solve DP again. We pick item `j`.
# ...
# If we always pick an item that is part of *the* optimal solution for the current subproblem,
# we will construct the global optimal solution.
#
# Why did the current program score 1.0019? It seems to be working well.
# Why did the "2-for-1 swap" hint mention 1.001? Maybe the current program IS that one,
# or close to it.
#
# Let's look at Inspiration 1: Pure density. Score 1.0000.
# Inspiration 2: Density * Fill. Score 0.9968.
# Inspiration 3: Beam search. Score 0.9934.
#
# The current program (DP) is theoretically superior to all inspirations if it runs fast enough.
# The main risk is performance or floating point issues.
#
# Improvements to make:
# 1. **Optimization of DP**: The current DP allocates a full 2D table `(N+1) x (C_int+1)`.
#    This is memory-heavy and might be slow due to allocation overhead in Python loops,
#    even with vectorized row updates.
#    We can use a 1D DP array for the values, but we lose the ability to backtrack easily.
#    
#    Alternative: We don't need the full table if we just want to find *one* valid item.
#    We can compute the optimal value `V_opt`.
#    Then, for each item `i`, check if `value[i] + OptimalValue(remaining_capacity - weight[i], items \ {i}) == V_opt`.
#    This is expensive to check for all items.
#    
#    Let's stick to the 2D table but optimize the loop.
#    Actually, we can avoid storing the full table if we just track the *choice* during the forward pass?
#    No, standard 0/1 knapsack DP with 1D array updates in reverse order doesn't allow backtracking
#    without storing the full table or a `take` matrix.
#    
#    Let's try to reduce the scale.
#    Capacity is 12.5.
#    If weights are generated with some distribution, what is the precision?
#    If we scale by 100 instead of 1000, C_int = 1250.
#    Table size: 100 * 1250 = 125,000.
#    This is 10x smaller.
#    Is scaling by 100 sufficient?
#    If weights are like 0.123, rounding to 12 (scale 100) introduces error.
#    However, knapsack problems often have weights that are "nice" numbers or the evaluator
#    might use integers or 1-decimal floats.
#    If the weights are arbitrary floats, rounding might lead to suboptimal or infeasible choices.
#    
#    Let's check the error.
#    If we round weights, we might allow an item to fit when it shouldn't, or prevent it from fitting when it should.
#    To be safe, we should use a higher scale, or handle floating point carefully.
#    
#    Actually, there is a better way to backtrack with 1D DP:
#    We can store the `dp` table as a list of arrays, one per item.
#    `dp_tables[i]` is the state after processing item `i`.
#    This is exactly what the current code does (`dp_table[i]`).
#    
#    Let's try to speed up the current DP.
#    The loop `for i in range(1, f_n + 1)` runs `f_n` times.
#    Inside, we do vectorized ops.
#    `curr_row[:] = prev_row` is a copy of size `C_int+1`.
#    `curr_row[w:] = np.maximum(...)` is an op on size `C_int-w+1`.
#    
#    If `C_int` is 12500 (scale 1000), and `f_n` is 100.
#    Total elements copied: 100 * 12500 = 1.25M.
#    This is very fast in NumPy.
#    
#    The bottleneck might be the Python loop overhead itself, but 100 iterations is nothing.
#    
#    So the current code should be fast.
#    
#    What could go wrong?
#    1. **Floating Point Precision in Backtrack**:
#       `if dp_table[i][j] != dp_table[i-1][j]:`
#       Floating point equality is dangerous.
#       We should use a tolerance.
#       `if abs(dp_table[i][j] - dp_table[i-1][j]) > 1e-9:`
#       
#    2. **Scale Selection**:
#       If we use scale 1000, `C_int` is 12500.
#       If weights are very small, e.g., 0.0001, `w_int` becomes 0.
#       If `w_int` is 0, the DP update `curr_row[0:] = max(curr_row[0:], prev_row[0:] + v)`
#       effectively adds `v` to all capacities. This is correct for 0-weight items.
#       
#    3. **Better Item Selection**:
#       The backtrack finds the *last* item in the optimal set (in the order of the array).
#       Is it possible that picking a *different* item from the optimal set is better for the *next* step?
#       No, because the next step will also solve the DP exactly.
#       If `S` is the optimal set, and we pick `i in S`, the remaining optimal set is `S \ {i}`.
#       The DP will find `S \ {i}` (or an equivalent set with the same value).
#       So the order of picking doesn't matter for the final value, as long as we always pick from the optimal set.
#       
#       However, there is a catch: **Ties**.
#       If there are multiple optimal sets, the DP table might favor one due to floating point errors or iteration order.
#       If we pick an item from Set A (optimal), but the DP table for the next step is influenced by the specific
#       floating point values, we might drift?
#       Unlikely, as the values are deterministic.
#       
#       But what if `dp_table[i][j] == dp_table[i-1][j]` due to floating point noise, even though item `i` *should* be taken?
#       Or vice versa?
#       Using a tolerance helps.
#       
#    4. **Hybrid Approach**:
#       The "2-for-1 swap" hint suggests that local search improvements over greedy helped.
#       But DP is exact. Why would local search help?
#       Maybe the DP is failing due to discretization?
#       If we discretize weights, we solve a *different* problem (integer weights) than the original (float weights).
#       The solution to the integer problem might not be feasible or optimal for the float problem.
#       
#       **Crucial Insight**:
#       If we discretize weights, we might pick an item that fits in the discretized capacity but not in the real capacity.
#       Example:
#       Real Capacity: 1.0
#       Item A weight: 0.9999
#       Item B weight: 0.0001
#       Scale 1000:
#       C_int = 1000
#       w_A = 1000 (rounds to 1000)
#       w_B = 0 (rounds to 0)
#       
#       In DP:
#       Can we take A? `1000 <= 1000`. Yes.
#       Real: `0.9999 <= 1.0`. Yes.
#       
#       Can we take B? `0 <= 1000`. Yes.
#       Real: `0.0001 <= 1.0`. Yes.
#       
#       Can we take A and B?
#       DP: `1000 + 0 = 1000 <= 1000`. Yes.
#       Real: `0.9999 + 0.0001 = 1.0000 <= 1.0`. Yes.
#       
#       What if:
#       Real Capacity: 1.0
#       Item A weight: 0.9996
#       Item B weight: 0.0006
#       Scale 1000:
#       C_int = 1000
#       w_A = 1000 (rounds up)
#       w_B = 0 (rounds down)
#       
#       DP: Take A (1000) and B (0) -> Total 1000. OK.
#       Real: 0.9996 + 0.0006 = 1.0002 > 1.0. **Infeasible!**
#       
#       So discretization can lead to infeasible solutions.
#       The evaluator will likely reject infeasible items or cause an error.
#       The task says "invalid result ... scores 0".
#       
#       To avoid this, we must ensure the item we return is **feasible** in the real (float) domain.
#       The DP might suggest taking item `i` because it fits in the integer domain, but it doesn't fit in the float domain.
#       
#       **Fix**:
#       After identifying candidate items from the DP backtrack, we must verify they fit in the real capacity.
#       But the backtrack only gives us *one* item.
#       If that item is infeasible in the real domain, we have a problem.
#       
#       How to handle this?
#       1. Use a finer scale (e.g., 10000) to reduce error.
#       2. After DP, verify the chosen item. If it doesn't fit, pick the next best feasible item?
#          But "next best" might not be in the optimal set.
#       3. **Better**: When checking feasibility in the DP, use the *real* weights for the final check?
#          No, DP needs integer indices.
#          
#       Alternative:
#       Don't rely solely on DP for the final decision.
#       Use DP to estimate the optimal value.
#       Then, among all items that are **feasible** in the real domain, pick the one that maximizes:
#       `value[i] + DP_optimal(remaining_capacity - weight[i], items \ {i})`
#       This is the "best first move" heuristic.
#       But calculating `DP_optimal` for each item is expensive.
#       
#       Let's refine the DP approach:
#       1. Compute DP with integer weights (scale 1000).
#       2. Get the optimal value `V_int`.
#       3. Backtrack to find the set of items `S_int` that achieves `V_int`.
#       4. Filter `S_int` to only those items that are feasible in the real domain (weight <= remaining_capacity).
#          Actually, if an item is in `S_int`, it was considered feasible in the integer domain.
#          It might not be feasible in the real domain due to rounding.
#       5. If `S_int` is empty (all items in optimal integer set are infeasible in real domain), fallback to greedy.
#       6. If `S_int` is not empty, pick one item from it.
#       
#       But wait, if an item `i` is in `S_int`, it means `sum(w_int[k] for k in S_int) <= C_int`.
#       Does this imply `sum(w_float[k] for k in S_int) <= C_float`?
#       Not necessarily, due to rounding errors accumulating.
#       
#       However, we are only picking **one** item at a time.
#       The evaluator removes that item and calls us again.
#       So we don't need to ensure the *entire set* `S_int` is feasible in the real domain simultaneously.
#       We just need to ensure the **single item** we return is feasible in the real domain.
#       
#       So:
#       1. Compute DP.
#       2. Backtrack to find *any* item `i` in the optimal integer solution.
#       3. Check if `weights[i] <= remaining_capacity`.
#          - If yes, return `i`.
#          - If no, this item is infeasible in the real domain. We should not return it.
#            We need to find another item.
#            
#       How to find another item?
#       We can modify the backtrack to find *all* items that are part of *some* optimal solution?
#       Or, we can simply iterate through all feasible items (in real domain) and check if they are part of the optimal solution?
#       
#       Let's try this:
#       1. Compute DP table.
#       2. Get `V_opt = dp_table[N][C_int]`.
#       3. For each item `i` in the remaining set (that is feasible in real domain):
#          Check if `value[i] + dp_table_excluding_i(C_int - w_int[i]) == V_opt`.
#          This requires knowing `dp_table_excluding_i`.
#          This is hard.
#          
#       Simpler:
#       Just pick the item from the backtrack that is feasible.
#       If the backtrack returns an infeasible item, we can try to find the *next* item in the optimal set.
#       We can modify the backtrack to collect *all* items that are taken in the specific optimal path found by the greedy backtrack.
#       Then filter for feasibility.
#       
#       Let's implement:
#       1. Backtrack to get the list of items in the optimal solution (the specific one found by the standard backtrack).
#       2. Filter this list to items that are feasible in the real domain.
#       3. If the list is not empty, return the first one (or the one with highest value/density).
#       4. If the list is empty, fallback to greedy (highest density among feasible items).
#       
#       This seems robust.
#       
#    5. **Performance**:
#       Backtracking is O(N).
#       Filtering is O(N).
#       DP is O(N*C).
#       Total per call: O(N*C).
#       
#    6. **Scale**:
#       Let's use scale 1000.
#       If `C_int` is too large, we might hit memory limits.
#       12500 * 100 * 8 bytes = 10MB. Fine.
#       
#    7. **Edge Case: No feasible items in real domain**.
#       Return -1.
#       
#    8. **Edge Case: DP optimal value is 0**.
#       This means no items should be taken? Or all values are 0?
#       If all values are 0, any item is fine.
#       If no items fit in integer domain, but some fit in real domain?
#       Example: C=1.0, w=0.0001.
#       C_int=1000, w_int=0.
#       DP will take it.
#       If C=0.0001, C_int=0.
#       w_int=0.
#       DP will take it.
#       If C=0.0, C_int=0.
#       w_int=0.
#       DP will take it.
#       
#       What if `w_int > C_int` but `w_float <= C_float`?
#       Example: C=0.0004, w=0.0004.
#       Scale 1000: C_int=0, w_int=0. OK.
#       Example: C=0.0004, w=0.0005.
#       Scale 1000: C_int=0, w_int=1.
#       DP: w_int > C_int, so item is not taken.
#       Real: 0.0005 > 0.0004, so item is infeasible. Correct.
#       
#       Example: C=0.0005, w=0.0004.
#       Scale 1000: C_int=1 (round(0.5)=0? No, round(0.5) is 0 in Python? `round(0.5)` is 0. `round(1.5)` is 2. Banker's rounding?
#       Let's use `int(round(x * scale))`.
#       `round(0.0005 * 1000) = round(0.5) = 0`.
#       `round(0.0004 * 1000) = round(0.4) = 0`.
#       So C_int=0, w_int=0.
#       DP takes it.
#       Real: 0.0004 <= 0.0005. Feasible.
#       
#       Example: C=0.0005, w=0.0006.
#       C_int=0, w_int=1.
#       DP: w_int > C_int. Not taken.
#       Real: 0.0006 > 0.0005. Infeasible.
#       
#       It seems scale 1000 is reasonably safe for typical floating point inputs.
#       
#    Let's implement the refined DP with backtrack and feasibility filter.

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if len(remaining_weights) == 0:
        return -1
    
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    n = len(weights)
    
    # Filter feasible items in REAL domain
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    f_n = len(f_weights)
    
    if f_n == 0:
        return -1
        
    # Discretize capacity
    scale = 1000
    C_int = int(round(remaining_capacity * scale))
    if C_int < 0:
        return -1
        
    f_weights_int = np.round(f_weights * scale).astype(int)
    
    # Ensure weights are non-negative
    f_weights_int = np.maximum(f_weights_int, 0)
    
    # If capacity is 0 in integer domain, we can only take items with weight 0.
    # DP handles this naturally.
    
    # Time guard
    start_time = time.time()
    deadline = start_time + 5.0 # 5s per call is generous. 800 calls * 5s = 4000s >> 150s.
    # Actually, we want to be much faster. 150s / 800 = 0.187s.
    # DP should be < 10ms.
    
    # DP Table
    # dp[i][j] = max value using first i items with capacity j
    # Size: (f_n + 1) x (C_int + 1)
    
    # Optimization: If C_int is very large, we might run out of memory or time.
    # But C_int is ~12500.
    
    # Initialize
    # We use a list of arrays to allow backtracking.
    dp_table = []
    dp_table.append(np.zeros(C_int + 1, dtype=float))
    
    for i in range(1, f_n + 1):
        w = f_weights_int[i-1]
        v = f_values[i-1]
        prev_row = dp_table[i-1]
        
        # Create current row
        curr_row = prev_row.copy()
        
        if w <= C_int:
            # Update from w to C_int
            # curr_row[j] = max(curr_row[j], prev_row[j-w] + v)
            # Note: We use prev_row for the source, not curr_row, to ensure 0/1 (each item used once).
            candidates = prev_row[:C_int - w + 1] + v
            curr_row[w:] = np.maximum(curr_row[w:], candidates)
            
        dp_table.append(curr_row)
        
        # Time check
        if time.time() - start_time > deadline:
            # Fallback to greedy
            break
            
    if time.time() - start_time > deadline:
        densities = f_values / f_weights
        best_idx_in_f = int(np.argmax(densities))
        return int(feasible_indices[best_idx_in_f])

    # Backtrack to find items in the optimal solution
    # Start from i=f_n, j=C_int
    j = C_int
    optimal_items = []
    
    for i in range(f_n, 0, -1):
        # Check if item i-1 was taken
        # Condition: dp[i][j] > dp[i-1][j]
        # Using tolerance for float comparison
        if dp_table[i][j] > dp_table[i-1][j] + 1e-9:
            # Item i-1 is taken
            optimal_items.append(i-1)
            # Reduce capacity by weight of item i-1
            w = f_weights_int[i-1]
            j -= w
            # Ensure j is not negative (shouldn't happen if DP is correct)
            if j < 0:
                break
        else:
            # Item i-1 not taken, j remains same
            pass
            
    # Filter optimal_items to those that are feasible in the real domain
    # The indices in optimal_items are relative to f_weights/f_values (0 to f_n-1)
    # We need to map them back to the original feasible_indices
    
    valid_candidates = []
    for idx_in_f in optimal_items:
        # idx_in_f is index in f_weights
        orig_idx = feasible_indices[idx_in_f]
        # Check real feasibility
        if f_weights[idx_in_f] <= remaining_capacity:
            valid_candidates.append(orig_idx)
            
    if valid_candidates:
        # Pick the one with highest value? Or just the first one?
        # Since they are all part of the optimal set, any is fine.
        # Let's pick the one with highest value to be safe?
        # Or highest density?
        # It shouldn't matter for the final score if we re-solve DP next time.
        # Let's just pick the first one.
        return int(valid_candidates[0])
        
    # Fallback: Greedy
    # If no item from the DP optimal set is feasible (due to rounding issues),
    # or if DP failed, use greedy.
    densities = f_values / f_weights
    # Avoid division by zero
    densities[f_weights == 0] = np.inf
    best_idx_in_f = int(np.argmax(densities))
    return int(feasible_indices[best_idx_in_f])

# EVOLVE-BLOCK-END
