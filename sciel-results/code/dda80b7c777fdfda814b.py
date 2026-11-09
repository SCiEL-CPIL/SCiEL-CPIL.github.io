"""Combined approach: bitset constraint propagation with LCM-periodic window search."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)  # 30030

    # Initial solution
    naive = divisors[:]
    best_val = max(naive) - min(naive)
    best_assign = naive[:]

    def check_window(W, L, deadline):
        """Check if window [L, L+W] has a valid distinct assignment using bitset propagation."""
        if L < 1 or L + W > MAX_V:
            return None

        candidate_bitsets = []
        for d in divisors:
            p = L % d
            first_offset = (d - p) % d
            if first_offset > W:
                return None
            bits = 0
            for offset in range(first_offset, W + 1, d):
                bits |= (1 << offset)
            candidate_bitsets.append(bits)

        # Build position -> possible divisors mapping
        pos_possible = [0] * (W + 1)
        for i in range(n):
            bits = candidate_bitsets[i]
            j = 0
            while bits:
                if bits & 1:
                    pos_possible[j] |= (1 << i)
                bits >>= 1
                j += 1

        # Check each divisor has at least one available position
        for i in range(n):
            if candidate_bitsets[i] == 0:
                return None

        # Backtracking with MRV and conflict-aware ordering
        assignment = [None] * n
        used_positions = 0

        # Precompute conflict counts for each position: how many divisors can use it
        pos_conflict_count = [bin(pos_possible[j]).count('1') for j in range(W + 1)]

        # Order divisors by number of available positions (MRV)
        order = sorted(range(n), key=lambda i: bin(candidate_bitsets[i]).count('1'))

        def backtrack(idx):
            nonlocal used_positions
            if time.time() > deadline:
                return False
            if idx == n:
                return True

            i = order[idx]
            avail = candidate_bitsets[i] & ~used_positions

            if avail == 0:
                return False

            # Collect available positions with their conflict counts
            positions = []
            bits = avail
            j = 0
            while bits:
                if bits & 1:
                    # Count how many other unassigned divisors also want this position
                    conflicts = 0
                    for k in range(idx + 1, n):
                        if candidate_bitsets[k] & (1 << j):
                            conflicts += 1
                    positions.append((conflicts, j))
                bits >>= 1
                j += 1

            # Sort by conflicts (ascending) - prefer positions with fewer conflicts
            positions.sort()

            for _, j in positions:
                bit = 1 << j
                if avail & bit:
                    used_positions |= bit
                    assignment[i] = L + j
                    if backtrack(idx + 1):
                        return True
                    used_positions &= ~bit
                    assignment[i] = None

            return False

        if backtrack(0):
            return assignment[:]
        return None

    # Systematic search over W from theoretical minimum
    # Minimum possible span for 10 distinct integers is 9
    for W in range(9, 100):
        if time.time() > deadline:
            break

        # Due to LCM periodicity, the pattern repeats every LCM
        # So we only need to try L in [1, LCM]
        max_L = min(LCM, MAX_V - W)

        found = False
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                break

            result = check_window(W, L, deadline)
            if result is not None:
                span = max(result) - min(result)
                if span < best_val:
                    best_val = span
                    best_assign = result[:]
                    found = True
                    break  # Found solution with span W, move to next W

        if found and best_val == W:
            break

    # If we found a good solution, do local search to polish
    if best_assign is not None and best_val > 9:
        for iteration in range(200):
            if time.time() > deadline:
                break

            candidate = best_assign[:]
            improved = False

            idx = random.randint(0, n - 1)
            d = divisors[idx]
            current = candidate[idx]

            for delta in [-2, -1, 1, 2]:
                new_val = current + delta * d
                if new_val < 1 or new_val > MAX_V:
                    continue
                # Check distinctness
                if any(candidate[j] == new_val for j in range(n) if j != idx):
                    continue

                candidate[idx] = new_val
                span = max(candidate) - min(candidate)
                if span < best_val:
                    best_val = span
                    best_assign = candidate[:]
                    improved = True
                    break
                candidate[idx] = current

            if improved:
                iteration = 0

    # Final validation
    if best_assign is None:
        best_assign = divisors[:]

    result = []
    for i, d in enumerate(divisors):
        v = int(best_assign[i])
        result.append(v)

    # Verify all constraints
    for i, d in enumerate(divisors):
        assert isinstance(result[i], int) and not isinstance(result[i], bool)
        assert 1 <= result[i] <= MAX_V
        assert result[i] % d == 0
    assert len(set(result)) == n

    return result

# EVOLVE-BLOCK-END
