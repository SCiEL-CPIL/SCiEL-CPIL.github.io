import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030  # 2*3*5*7*11*13

    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s wall-clock kill

    # Minimum possible span is 9 (10 distinct integers need at least 10 slots)
    for W in range(9, 200):
        if time.time() > deadline:
            break

        # For each L in [0, LCM), check if all divisors have a multiple in [L, L+W]
        # Then use backtracking to find distinct assignments

        for L in range(LCM):
            if time.time() > deadline:
                break

            # Quick check: for each divisor d, is there a multiple in [L, L+W]?
            # A multiple exists iff (-L) mod d <= W
            candidates = []
            valid = True
            for d in divisors:
                r0 = (-L) % d
                if r0 > W:
                    valid = False
                    break
                # Build candidate offsets
                cands = []
                r = r0
                while r <= W:
                    cands.append(r)
                    r += d
                candidates.append(cands)

            if not valid:
                continue

            # MRV ordering: most constrained (fewest candidates) first
            indices = list(range(n))
            indices.sort(key=lambda i: len(candidates[i]))

            # Pruning: check that the union of candidates for the k most constrained
            # has size >= k for all k (Hall's condition, simplified)
            # For speed, just check the most constrained one
            if len(candidates[indices[0]]) == 0:
                continue

            # Use bitmask for used slots (W <= 199, so at most 200 bits)
            # Represent used as a Python int (arbitrary precision)
            used_mask = 0
            choices = [-1] * n

            def backtrack(idx):
                nonlocal used_mask
                if idx == n:
                    return True

                # Pruning: remaining variables need (n - idx) distinct slots
                remaining_slots = (W + 1) - bin(used_mask).count('1')
                if remaining_slots < (n - idx):
                    return False

                i = indices[idx]
                for r in candidates[i]:
                    bit = 1 << r
                    if not (used_mask & bit):
                        used_mask |= bit
                        choices[i] = r
                        if backtrack(idx + 1):
                            return True
                        used_mask &= ~bit
                        choices[i] = -1
                return False

            if backtrack(0):
                result = []
                for i in range(n):
                    val = L + choices[i]
                    if val <= 0:
                        # Shouldn't happen if L >= 1, but guard just in case
                        break
                    result.append(int(val))
                else:
                    # All values are positive
                    return result

    # Fallback: return the divisors themselves
    return [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
