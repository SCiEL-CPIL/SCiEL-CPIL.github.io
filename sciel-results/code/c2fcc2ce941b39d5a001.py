# EVOLVE-BLOCK-START
def construct():
    import time
    from math import isqrt

    deadline = time.time() + 55.0

    N = 10000
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start::step] = [False] * (((N - start) // step) + 1)
    primes = [i for i, v in enumerate(is_prime) if v]
    num_primes = len(primes)

    # dp[i] = list of states (gap, len, pj, psi)
    # We use lists of tuples for speed? Or dicts?
    # Tuples are faster.
    # State: (gap, length, parent_j, parent_state_idx)
    
    dp = [[] for _ in range(num_primes)]

    for i in range(num_primes):
        if time.time() > deadline:
            break
        # Base case
        dp[i].append((0, 1, -1, -1))

        for j in range(i):
            gap = primes[i] - primes[j]
            # Extend from j
            # We need to iterate over dp[j]
            # dp[j] is a list of tuples
            for state in dp[j]:
                if time.time() > deadline:
                    break
                if state[0] <= gap:
                    # New state: store reference to parent state object
                    dp[i].append((gap, state[1] + 1, j, state))
        # Prune
        if dp[i]:
            # Group by gap, keep max len
            # And keep track of the state that gave it
            best_for_gap = {}
            for idx, s in enumerate(dp[i]):
                g = s[0]
                l = s[1]
                if g not in best_for_gap or l > best_for_gap[g][1]:
                    best_for_gap[g] = s
            
            items = sorted(best_for_gap.values(), key=lambda x: x[0])
            frontier = []
            max_len = 0
            for s in items:
                if s[1] > max_len:
                    frontier.append(s)
                    max_len = s[1]
            dp[i] = frontier

    # Find best
    best_len = 0
    best_i = -1
    best_state_idx = -1
    for i in range(num_primes):
        for si, s in enumerate(dp[i]):
            if s[1] > best_len:
                best_len = s[1]
                best_i = i
                best_state_idx = si

    if best_i == -1:
        return [2] if primes else []

    # Reconstruct
    chain = []
    i = best_i
    s = dp[i][best_state_idx]
    while s is not None:
        chain.append(primes[i])
        if s[2] == -1:
            break
        i = s[2]
        s = s[3]

    chain.reverse()

    # Validate
    if len(chain) >= 2:
        gaps = [chain[k+1] - chain[k] for k in range(len(chain)-1)]
        valid = True
        for k in range(len(gaps)-1):
            if gaps[k+1] < gaps[k]:
                valid = False
                break
        if valid:
            return chain

    # Fallback
    values = []
    for p in primes:
        if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
            values.append(p)
    return values
# EVOLVE-BLOCK-END
