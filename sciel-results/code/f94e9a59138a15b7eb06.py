"""Difference triangle set (4,7): Optimized CSP solver using bitsets for difference tracking."""
import random
import sys
import time
sys.setrecursionlimit(100000)

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows, each of length k+1.
    Uses a bitset-based backtracking solver with strong pruning to minimize the maximum element (scope).
    """
    # Greedy baseline
    def greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        d = x - a
                        used |= (1 << d)
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = max(max(r) for r in best_rows)
    
    # Precompute total differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def popcount(x):
        return bin(x).count('1')

    def solve_scope(scope, seed, max_nodes=2000000):
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        nodes = [0]
        # Mask for valid differences (1 to scope)
        valid_mask = (1 << (scope + 1)) - 2 # bits 1..scope set
        
        def build_row(idx):
            nonlocal used_bits, nodes
            if nodes[0] > max_nodes:
                return False
            nodes[0] += 1
            
            if idx == n_rows:
                return True
            
            row = [0]
            row_diff_bits = 0
            
            # Symmetry breaking: 
            # We assume rows are sorted by their second element (index 1).
            # This is a valid symmetry breaking for DTS.
            if idx == 0:
                min_start = 1
            else:
                min_start = rows[idx - 1][1] + 1
            
            def rec(row, row_diff_bits, min_x):
                nonlocal used_bits, nodes
                if nodes[0] > max_nodes:
                    return False
                nodes[0] += 1
                
                if len(row) == k + 1:
                    # Check conflict
                    if used_bits & row_diff_bits:
                        return False
                    
                    used_bits |= row_diff_bits
                    rows.append(row)
                    if build_row(idx + 1):
                        return True
                    else:
                        rows.pop()
                        used_bits ^= row_diff_bits
                        return False

                remaining_elems = (k + 1) - len(row)
                # We need to pick 'remaining_elems' more elements.
                # Each new element adds 'len(row)' new differences.
                # Total new diffs needed: sum_{i=len(row)}^{k} i ? No.
                # If we have m elements, next element adds m diffs.
                # Current len is m. Next adds m. Then m+1 adds m+1.
                # Total remaining diffs = m + (m+1) + ... + k
                # where m = len(row)
                m = len(row)
                rem_diffs_needed = sum(range(m, k + 1))
                
                # Pruning: Check if enough distinct differences are available
                # Available bits: valid_mask & ~used_bits & ~row_diff_bits
                # Note: row_diff_bits are "reserved" for this row, so they are not available for other rows,
                # but they are also not "used" globally yet. However, for the *current* row,
                # we cannot reuse row_diff_bits.
                # So available for current row = valid_mask & ~(used_bits | row_diff_bits)
                
                available = valid_mask & ~(used_bits | row_diff_bits)
                avail_count = popcount(available)
                
                if avail_count < rem_diffs_needed:
                    return False

                max_x = scope
                # Heuristic: The largest element in the row is bounded by scope.
                # Also, the differences from the last element a_k are a_k - a_i.
                # The largest difference is a_k - a_0 = a_k.
                # So a_k <= scope.
                
                # Generate candidates for the next element x
                # x must be > min_x
                # x - a_i must be in 'available' for all a_i in row
                
                candidates = []
                # Optimization: Iterate x from min_x to scope
                # But we can bound x.
                # The smallest possible value for the next element is min_x.
                # The largest possible value is scope.
                
                for x in range(min_x, scope + 1):
                    ok = True
                    # Check if x - a_i is available
                    # We can do this with bit operations if we shift?
                    # d_i = x - a_i.
                    # We need (available >> d_i) & 1 for all i.
                    # This is fast enough for small k.
                    for a in row:
                        d = x - a
                        if d > scope: break # x is too big? No, d < x <= scope.
                        if (available >> d) & 1:
                            continue
                        else:
                            ok = False
                            break
                    if ok:
                        candidates.append(x)
                
                if not candidates:
                    return False
                
                # Heuristic: Choose candidate with "least impact" or "fail-first"
                # Fail-first: Pick the candidate that leaves the fewest options for the next step.
                # This is expensive to compute.
                # Alternative: Pick the candidate that is "most constrained" (smallest x? largest x?)
                # Usually, smaller x keeps the row "tighter" and might fill up small differences,
                # which are more scarce?
                # Or larger x uses large differences, which are more abundant?
                # Actually, small differences are "scarce" in the sense that they appear in more potential conflicts?
                # Let's try sorting candidates by x (ascending) or by some metric.
                
                # Let's use a mix:
                # 1. Try the smallest valid x (greedy)
                # 2. Try a random valid x
                # 3. Try the largest valid x
                
                # To improve on the seed, we can vary the strategy based on seed.
                if len(candidates) > 1:
                    if seed % 3 == 0:
                        x = candidates[0]
                    elif seed % 3 == 1:
                        x = candidates[-1]
                    else:
                        x = rng.choice(candidates)
                else:
                    x = candidates[0]
                
                new_diffs = 0
                for a in row:
                    d = x - a
                    new_diffs |= (1 << d)
                
                if rec(row + [x], row_diff_bits | new_diffs, x + 1):
                    return True
                return False
            
            return rec(row, row_diff_bits, min_start)
        
        success = build_row(0)
        if success:
            return rows
        return None

    t0 = time.time()
    time_limit = 115
    best_found = best_rows
    best_scope_found = best_scope
    
    # Search downwards from best_scope - 1
    for scope in range(best_scope - 1, 99, -1):
        if time.time() - t0 > time_limit:
            break
            
        found = None
        # Try several seeds
        for seed in range(15):
            if time.time() - t0 > time_limit:
                break
            res = solve_scope(scope, seed, max_nodes=2000000)
            if res:
                found = res
                break
        
        if found:
            best_found = found
            best_scope_found = scope
        else:
            pass

    return best_found
