"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from functools import reduce
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    def try_assign_in_window(L, W):
        """
        Given a window [L, L+W], check if we can assign distinct multiples.
        Returns the assignment list if successful, else None.
        """
        avail = []
        for d in divisors:
            k_min = (L + d - 1) // d
            k_max = (L + W) // d
            if k_max < k_min:
                return None
            # Collect available multiples
            multiples = [k * d for k in range(k_min, k_max + 1)]
            avail.append(multiples)
        
        # Backtracking to find distinct assignment
        # Sort indices by number of options to prune faster
        indices = list(range(n))
        indices.sort(key=lambda i: len(avail[i]))
        
        used = set()
        assignment = [None] * n
        
        def backtrack(idx_in_sorted):
            if idx_in_sorted == n:
                return True
            i = indices[idx_in_sorted]
            for m in avail[i]:
                if m not in used:
                    used.add(m)
                    assignment[i] = m
                    if backtrack(idx_in_sorted + 1):
                        return True
                    used.remove(m)
            return False
        
        if backtrack(0):
            return assignment
        return None

    # Strategy 1: Deterministic scan for small W
    # We scan L from 1 to a limit. If a solution exists with small W, it's likely found early.
    # If not found, we increase W.
    max_L_scan = 50000
    for W in range(1, 200):
        # Optimization: If W is very small, we might need to scan more L.
        # But 50k * 200 * 10 checks is ~100M ops, might be slow in Python.
        # Let's limit W to 50 for the full scan, then use random/heuristic for larger W.
        if W <= 50:
            for L in range(1, max_L_scan + 1):
                # Quick check: does each divisor have a multiple in [L, L+W]?
                possible = True
                for d in divisors:
                    k_min = (L + d - 1) // d
                    k_max = (L + W) // d
                    if k_max < k_min:
                        possible = False
                        break
                if not possible:
                    continue
                
                # Try to assign
                res = try_assign_in_window(L, W)
                if res is not None:
                    # Verify constraints
                    if all(isinstance(x, int) for x in res) and len(set(res)) == n:
                        if all(res[i] % divisors[i] == 0 for i in range(n)):
                            if all(1 <= x <= max_val for x in res):
                                return res
        else:
            # For larger W, use random sampling of L
            for _ in range(1000):
                L = random.randint(1, max_val - W)
                possible = True
                for d in divisors:
                    k_min = (L + d - 1) // d
                    k_max = (L + W) // d
                    if k_max < k_min:
                        possible = False
                        break
                if not possible:
                    continue
                
                res = try_assign_in_window(L, W)
                if res is not None:
                    if all(isinstance(x, int) for x in res) and len(set(res)) == n:
                        if all(res[i] % divisors[i] == 0 for i in range(n)):
                            if all(1 <= x <= max_val for x in res):
                                return res
            # If random fails for this W, try next W (larger)
            # But we should probably break and try a different strategy if W gets too big
            if W > 100:
                break

    # Strategy 2: Randomized Local Search
    # Start with a random valid assignment and try to minimize the range.
    def random_initial_assignment():
        res = []
        used = set()
        for d in divisors:
            # Pick a random multiple
            while True:
                k = random.randint(1, max_val // d)
                val = k * d
                if val not in used:
                    used.add(val)
                    res.append(val)
                    break
        return res

    best_res = None
    best_range = float('inf')
    
    for _ in range(500):
        res = random_initial_assignment()
        current_range = max(res) - min(res)
        if current_range < best_range:
            best_range = current_range
            best_res = res[:]
        
        # Local search: try to move one element to reduce range
        # Identify min and max
        min_idx = res.index(min(res))
        max_idx = res.index(max(res))
        
        # Try to shrink max: find next smaller multiple of d[max_idx] that is not used
        d_max = divisors[max_idx]
        current_max = res[max_idx]
        # Try decreasing k
        k_curr = current_max // d_max
        found_new = False
        for k in range(k_curr - 1, 0, -1):
            new_val = k * d_max
            if new_val in used:
                continue
            # Check if this reduces range
            new_max = max(max(res[:max_idx] + res[max_idx+1:]), new_val)
            new_min = min(res)
            new_range = new_max - new_min
            if new_range < current_range:
                used.remove(current_max)
                used.add(new_val)
                res[max_idx] = new_val
                current_range = new_range
                found_new = True
                break
        
        # Try to grow min: find next larger multiple of d[min_idx] that is not used
        if not found_new:
            d_min = divisors[min_idx]
            current_min = res[min_idx]
            k_curr = current_min // d_min
            for k in range(k_curr + 1, max_val // d_min + 1):
                new_val = k * d_min
                if new_val in used:
                    continue
                new_min = min(min(res[:min_idx] + res[min_idx+1:]), new_val)
                new_max = max(res)
                new_range = new_max - new_min
                if new_range < current_range:
                    used.remove(current_min)
                    used.add(new_val)
                    res[min_idx] = new_val
                    current_range = new_range
                    found_new = True
                    break

    if best_res:
        # Final verification
        if all(isinstance(x, int) for x in best_res) and len(set(best_res)) == n:
            if all(best_res[i] % divisors[i] == 0 for i in range(n)):
                if all(1 <= x <= max_val for x in best_res):
                    return best_res

    # Fallback: Return divisors themselves (valid but likely large range)
    return divisors[:]
# EVOLVE-BLOCK-END
