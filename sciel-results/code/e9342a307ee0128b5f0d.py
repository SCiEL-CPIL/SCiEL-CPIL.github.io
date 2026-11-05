"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

# Precompute divisors of 5040
_N = 5040
_TARGET = 4979

def _get_divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs, reverse=True)

_DIVS = _get_divisors(_N)
# Filter to only usable divisors (<= target)
_USABLE = [d for d in _DIVS if d <= _TARGET]
_USABLE.sort(reverse=True)
_NUM_USABLE = len(_USABLE)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 second safety margin before 60s kill
    
    # We want to find the minimum number of distinct divisors that sum to TARGET.
    # This is a subset sum problem with an objective to minimize the count.
    # Since the number of divisors is small (60), we can use dynamic programming.
    # dp[s] = minimum number of divisors needed to sum to s
    # We also need to track which divisors were used to reconstruct the solution.
    
    # dp_count[s] = minimum number of divisors needed to sum to s
    # dp_parent[s] = index of the divisor used to reach sum s
    INF = float('inf')
    dp_count = [INF] * (_TARGET + 1)
    dp_parent = [-1] * (_TARGET + 1)
    dp_count[0] = 0

    # For each divisor, update the DP table
    # We process divisors in any order, but we need to ensure distinctness.
    # Standard 0/1 knapsack style: iterate divisors, then sums in reverse.
    
    for i, d in enumerate(_USABLE):
        # Update in reverse to avoid using the same divisor twice
        # Only update sums from _TARGET down to d
        for s in range(_TARGET, d - 1, -1):
            if dp_count[s - d] + 1 < dp_count[s]:
                dp_count[s] = dp_count[s - d] + 1
                dp_parent[s] = i  # Store the index of the divisor used to reach s

        # Check deadline
        if time.time() > deadline:
            break

    # Check if solution exists
    if dp_count[_TARGET] == INF:
        # Fallback: use the greedy approach
        remainder = _TARGET
        chosen = []
        for d in _USABLE:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        assert remainder == 0, "Greedy fallback failed"
        return chosen

    # Reconstruct the solution
    chosen = []
    s = _TARGET
    while s > 0:
        idx = dp_parent[s]
        if idx == -1:
            break
        d = _USABLE[idx]
        chosen.append(d)
        s -= d
        
    # Verify the solution
    if sum(chosen) != _TARGET or len(chosen) != len(set(chosen)):
        # Fallback to greedy if reconstruction failed
        remainder = _TARGET
        chosen = []
        for d in _USABLE:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        assert remainder == 0, "Greedy fallback failed"
        return chosen

    # Ensure distinctness (should be guaranteed by DP, but double check)
    assert len(chosen) == len(set(chosen)), "Non-distinct divisors found"
    assert sum(chosen) == _TARGET, "Sum mismatch"

    return chosen

# EVOLVE-BLOCK-END
