import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 80.0

    # LCM of all divisors
    lcm_all = reduce(_lcm, divisors)  # 30030

    best_val = float('inf')
    best_assign = None

    def solve_window(L, W):
        """Try to assign distinct multiples to each divisor within [L, L+W].
        Returns assignment list or None. Uses iterative bitmask backtracking."""
        # For each divisor, find all multiples in [L, L+W]
        candidates = []
        for d in divisors:
            k = (L + d - 1) // d
            m = k * d
            opts = []
            while m <= L + W:
                if m <= MAX_V:
                    opts.append(m)
                m += d
            if not opts:
                return None
            candidates.append(opts)

        # Order divisors by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        assigned = [None] * n
        used_mask = 0

        # Map each candidate value to a global index for bitmask tracking
        # Since values can be large, use a dict
        val_to_idx = {}
        all_vals = []
        for i in range(n):
            for v in candidates[i]:
                if v not in val_to_idx:
                    val_to_idx[v] = len(all_vals)
                    all_vals.append(v)
        
        # For each divisor in order, map its candidates to bit indices
        ordered_candidates = []
        for i in order:
            ordered_candidates.append([val_to_idx[v] for v in candidates[i]])
        
        num_vals = len(all_vals)
        full_mask = (1 << num_vals) - 1

        # Iterative backtracking using explicit stack
        # Stack entries: (div_index, candidate_index, used_mask)
        stack = [(0, 0, 0)]
        assigned = [None] * n

        while stack:
            div_idx, cand_idx, mask = stack.pop()
            
            if div_idx == n:
                # Found a complete assignment
                for i in range(n):
                    assigned[i] = all_vals[assigned[i]] if assigned[i] is not None else None
                # Actually, let's track assignment differently
                break
            
            i = order[div_idx]
            cands = ordered_candidates[div_idx]
            
            # Try remaining candidates
            while cand_idx < len(cands):
                bit = cands[cand_idx]
                if not (mask & (1 << bit)):
                    # This value is available
                    assigned[i] = bit
                    # Push next state
                    stack.append((div_idx + 1, 0, mask | (1 << bit)))
                    break
                cand_idx += 1
            else:
                # No candidates worked, backtrack
                if div_idx > 0:
                    prev_i = order[div_idx - 1]
                    prev_bit = assigned[prev_i]
                    if prev_bit is not None:
                        assigned[prev_i] = None
                        # Continue with next candidate for previous divisor
                        # This is tricky with iterative approach; let's use recursion with sys.setrecursionlimit
                        pass
                break
        
        # The iterative approach is getting complex; let's use recursion with proper limits
        # Reset and use recursive backtracking
        assigned = [None] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for bit in ordered_candidates[idx]:
                if not (used_mask_ref[0] & (1 << bit)):
                    used_mask_ref[0] |= (1 << bit)
                    assigned[i] = bit
                    if backtrack(idx + 1):
                        return True
                    used_mask_ref[0] &= ~(1 << bit)
                    assigned[i] = None
            return False
        
        used_mask_ref = [0]
        if not backtrack(0):
            return None
        
        # Convert bit indices back to values
        result = [None] * n
        for i in range(n):
            result[i] = all_vals[assigned[i]]
        return result

    # Precompute gaps for each residue
    gaps = [[0]*n for _ in range(lcm_all)]
    for r in range(lcm_all):
        for i, d in enumerate(divisors):
            rem = r % d
            if rem == 0:
                gaps[r][i] = 0
            else:
                gaps[r][i] = d - rem

    # For each W, find residues where max gap <= W
    # Precompute max_gap for each residue
    max_gaps = [max(gaps[r]) for r in range(lcm_all)]

    # Search W from minimum upward
    # For each W, check all valid residues to find the best (smallest span) assignment
    for W in range(9, 500):
        if time.time() > deadline:
            break

        # Find valid residues for this W
        valid_residues = [r for r in range(lcm_all) if max_gaps[r] <= W]

        if not valid_residues:
            continue

        found_any = False
        for r in valid_residues:
            if time.time() > deadline:
                break

            # L values with L % lcm_all == r, L >= 1, L + W <= MAX_V
            if r == 0:
                L_start = lcm_all
            else:
                L_start = r

            if L_start + W > MAX_V:
                continue

            L = L_start
            if L < 1:
                L = L_start + lcm_all
            if L + W > MAX_V:
                continue

            result = solve_window(L, W)
            if result is not None:
                span = max(result) - min(result)
                if span < best_val:
                    best_val = span
                    best_assign = result[:]
                    found_any = True
                # Don't break - check all residues for this W to find minimum span

        # If we found a solution with span == W, that's optimal for this W
        # and since W is increasing, we can stop
        if found_any and best_val == W:
            break
        # If we found any solution, we could also stop if we're confident
        # but let's continue to find potentially smaller spans at same W

    if best_assign is None:
        # Fallback: use the divisors themselves
        return divisors[:]

    return best_assign
