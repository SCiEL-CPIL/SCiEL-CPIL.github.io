import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Precompute: for a row of k+1 elements starting at 0, the differences are
    # a[j] - a[i] for 0 <= i < j <= k. There are C(k+1, 2) differences.
    # For k=6, that's C(7,2) = 21 differences per row.
    # Total differences needed: n_rows * 21.
    
    # Strategy: Use a constraint-based approach with backtracking + local search.
    # Key insight: We want to minimize the maximum entry. 
    # 
    # New approach: Model as a packing problem.
    # Each row needs k+1 numbers: 0 = x_0 < x_1 < ... < x_k
    # The 21 differences must all be distinct across all rows.
    # 
    # We'll use a hybrid:
    # 1. Generate initial solution via greedy with randomization
    # 2. Use a more sophisticated local search that considers row-level moves
    #    (replacing entire rows at once) rather than single-element moves
    # 3. Add a constraint propagation layer to prune infeasible moves quickly
    
    n_diffs_per_row = (k + 1) * k // 2  # C(k+1, 2)
    
    def compute_row_diffs(row):
        """Compute all differences for a row."""
        diffs = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(row[j] - row[i])
        return diffs
    
    def row_diffs_set(row):
        return set(compute_row_diffs(row))
    
    def greedy_init(seed=None):
        """Greedy initialization with optional randomization."""
        if seed is not None:
            random.seed(seed)
        
        rows = []
        used = set()
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Add some randomization
                if random.random() < 0.1:
                    x += random.randint(0, 5)
                new_diffs = set(x - a for a in row)
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
        return rows
    
    def row_max(row):
        return row[-1]  # Since rows are sorted
    
    def all_row_maxes(rows):
        return max(row_max(r) for r in rows)
    
    def validate_full(rows):
        """Check if all differences are distinct across all rows."""
        all_diffs = set()
        for row in rows:
            rd = row_diffs_set(row)
            if rd & all_diffs:
                return False
            all_diffs |= rd
        return True
    
    def try_row_replacement(current_rows, ri, candidate_row, all_used_except_ri):
        """
        Try to replace row ri with candidate_row.
        all_used_except_ri is the set of all differences used by other rows.
        Returns (new_all_used, is_valid) or (None, False).
        """
        new_diffs = row_diffs_set(candidate_row)
        # Check internal consistency (should always be true for valid row)
        if len(new_diffs) != n_diffs_per_row:
            return None, False
        # Check no conflict with other rows
        if new_diffs & all_used_except_ri:
            return None, False
        # Compute new all_used
        new_all_used = all_used_except_ri | new_diffs
        return new_all_used, True
    
    def generate_candidate_row(upper_bound, used_diffs, seed_offset=0):
        """
        Generate a valid row (k+1 elements starting at 0) with max <= upper_bound
        that doesn't use any difference in used_diffs.
        """
        # Try greedy construction with the constraint
        row = [0]
        x = 0
        attempts = 0
        max_attempts = 5000
        while len(row) < k + 1 and attempts < max_attempts:
            attempts += 1
            x += 1
            if x > upper_bound:
                return None
            new_diffs = set(x - a for a in row)
            if not (new_diffs & used_diffs):
                row.append(x)
        if len(row) < k + 1:
            return None
        return row
    
    def local_search_row_level(rows, time_limit):
        """
        Local search that replaces entire rows at once.
        This avoids getting stuck in local minima from single-element moves.
        """
        current_rows = [r[:] for r in rows]
        current_val = all_row_maxes(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_val = current_val
        
        # Precompute all_used for each row
        row_diffs_list = [row_diffs_set(r) for r in current_rows]
        
        def compute_all_used_except(idx):
            result = set()
            for i in range(n_rows):
                if i != idx:
                    result |= row_diffs_list[i]
            return result
        
        T = 100.0
        T_min = 0.5
        cool = 0.99999
        no_improve = 0
        max_no_improve = 10000
        
        while time.time() < time_limit and no_improve < max_no_improve:
            # Pick a row to modify
            ri = random.randrange(n_rows)
            
            # Determine target max for this row
            # We want to reduce the global max, so try to make this row smaller
            target_max = min(current_val - 1, current_rows[ri][-1] - 1)
            if target_max < k:  # Minimum possible max for k+1 elements is k
                target_max = k
            
            # Get all differences used by other rows
            all_used_except_ri = compute_all_used_except(ri)
            
            # Generate candidate rows
            candidate = None
            for attempt in range(10):
                # Try with slightly different upper bounds
                ub = target_max + random.randint(0, 5)
                candidate = generate_candidate_row(ub, all_used_except_ri)
                if candidate is not None:
                    break
            
            if candidate is None:
                # Try a larger bound
                ub = current_val + random.randint(0, 10)
                candidate = generate_candidate_row(ub, all_used_except_ri)
                if candidate is None:
                    no_improve += 1
                    T = max(T_min, T * cool)
                    continue
            
            # Validate
            new_all_used, valid = try_row_replacement(
                current_rows, ri, candidate, all_used_except_ri
            )
            if not valid:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            # Compute new global max
            new_row_max = row_max(candidate)
            # Need to recompute global max considering all rows
            new_global_max = max(
                row_max(current_rows[i]) if i != ri else new_row_max
                for i in range(n_rows)
            )
            
            delta = new_global_max - current_val
            
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.1:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                # Update state
                current_rows[ri] = candidate
                row_diffs_list[ri] = row_diffs_set(candidate)
                current_val = new_global_max
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            T = max(T_min, T * cool)
        
        return best_rows, best_val
    
    def element_level_search(rows, time_limit):
        """
        Fine-grained local search on individual elements.
        """
        current_rows = [r[:] for r in rows]
        current_val = all_row_maxes(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_val = current_val
        
        row_diffs_list = [row_diffs_set(r) for r in current_rows]
        row_maxes = [row_max(r) for r in current_rows]
        
        def compute_all_used():
            result = set()
            for rd in row_diffs_list:
                result |= rd
            return result
        
        all_used = compute_all_used()
        
        T = 50.0
        T_min = 0.1
        cool = 0.999995
        no_improve = 0
        max_no_improve = 50000
        
        while time.time() < time_limit and no_improve < max_no_improve:
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            
            old_val = current_rows[ri][pi]
            
            # Generate candidates (mostly decreasing)
            candidates = []
            
            # Decreasing
            min_allowed = current_rows[ri][pi - 1] + 1
            max_dec = min(50, old_val - min_allowed)
            for d in range(1, max_dec + 1):
                candidates.append(old_val - d)
            
            # Small increasing
            if pi < k:
                max_inc = min(20, current_rows[ri][pi + 1] - 1 - old_val)
            else:
                max_inc = min(20, current_val - old_val)
            for d in range(1, max_inc + 1):
                candidates.append(old_val + d)
            
            # Random jump
            if random.random() < 0.2:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
                if low <= high:
                    candidates.append(random.randint(low, high))
            
            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            
            random.shuffle(candidates)
            
            moved = False
            for new_val in candidates:
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue
                
                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val
                
                new_diffs = row_diffs_set(cand_row)
                if len(new_diffs) != n_diffs_per_row:
                    continue
                
                # Check conflicts
                old_diffs = row_diffs_list[ri]
                conflict = False
                for d in new_diffs:
                    if d in all_used and d not in old_diffs:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Compute new global max
                cand_row_max = row_max(cand_row)
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                
                delta = new_global_max - current_val
                
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.05:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.05))
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    # Update
                    all_used -= old_diffs
                    all_used |= new_diffs
                    current_rows[ri] = cand_row
                    row_diffs_list[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = new_global_max
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    moved = True
                    break
            
            if not moved:
                no_improve += 1
            
            T = max(T_min, T * cool)
        
        return best_rows, best_val
    
    # Phase 1: Multiple diverse initializations
    best_rows = None
    best_val = float('inf')
    
    init_deadline = start_time + 5.0
    for trial in range(20):
        if time.time() > init_deadline:
            break
        rows = greedy_init(seed=trial * 12345)
        val = all_row_maxes(rows)
        if val < best_val:
            best_val = val
            best_rows = [r[:] for r in rows]
    
    # Phase 2: Row-level local search (coarse)
    if time.time() < deadline - 30:
        best_rows, best_val = local_search_row_level(
            best_rows, min(deadline - 30, start_time + 60)
        )
    
    # Phase 3: Element-level local search (fine)
    if time.time() < deadline - 5:
        best_rows, best_val = element_level_search(
            best_rows, deadline - 5
        )
    
    # Final validation
    if not validate_full(best_rows):
        # Fallback: regenerate
        best_rows = greedy_init()
    
    return best_rows

# EVOLVE-BLOCK-END
