import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 80.0

    # Sieve for largest prime factor
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        if lpf[i] == 0:
            for j in range(i, N + 1, i):
                lpf[j] = i

    # Binary search on the answer t: is there a split a+b=N with max(P(a),P(b)) <= t?
    # For a given t, "smooth" numbers are those whose largest prime factor <= t.
    # We need: exists a in [1, N-1] such that a is t-smooth AND N-a is t-smooth.

    best_val = float('inf')
    best_a = 1

    # First, do a linear scan to get a good initial upper bound
    # Actually, let's think about what the answer likely is.
    # N = 123457. We want both a and N-a to have small largest prime factors.
    # Let's try t from small to large.

    # Build sorted list of primes up to N for reference
    primes = []
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)

    # Precompute for each t-smoothness threshold which numbers are t-smooth
    # Strategy: iterate t over prime values. For each t, mark numbers with lpf <= t.
    # Then check if any a has both a and N-a marked.

    # To be efficient, iterate t over primes in increasing order.
    # For each prime t, we can incrementally add numbers whose lpf == t to the "smooth" set.

    # Initialize: t=1, only number 1 is 1-smooth (P(1)=1)
    smooth = [False] * (N + 1)
    smooth[1] = True

    # Group numbers by their largest prime factor
    # lpf_groups[lpf] = list of numbers with that lpf
    from collections import defaultdict
    lpf_groups = defaultdict(list)
    for x in range(2, N + 1):
        lpf_groups[lpf[x]].append(x)

    # Try t=1: only a=1, b=N-1. Is b 1-smooth? b=123456, lpf[123456]>1. No.
    # Iterate over primes in order
    for t in primes:
        # Add all numbers with lpf == t to smooth set
        for x in lpf_groups[t]:
            smooth[x] = True

        # Check if there exists a in [1, N-1] such that smooth[a] and smooth[N-a]
        # We can check this by iterating, but that's O(N) per prime. 
        # Better: maintain a set or bitset of smooth numbers.
        # Actually, let's just check: for a from 1 to N//2, if smooth[a] and smooth[N-a].
        # But we can be smarter: we only need to check once when we first find a valid split.
        # Since we iterate t in increasing order, the first t where a valid split exists is optimal.

        # Optimization: only check a values that became newly smooth or whose complement is smooth.
        # For now, do a linear check but with early exit.
        found = False
        # We can limit the search: a and N-a must both be in [1, N-1]
        # Check symmetric range to avoid double work
        for a in range(1, N // 2 + 1):
            if smooth[a] and smooth[N - a]:
                found = True
                best_a = a
                best_val = t
                break

        if found:
            break

        if time.time() > deadline:
            break

    # If we didn't find via the incremental approach (shouldn't happen for small N),
    # fall back to linear scan
    if best_val == float('inf'):
        for a in range(1, N):
            val = lpf[a] if lpf[a] > lpf[N - a] else lpf[N - a]
            if val < best_val:
                best_val = val
                best_a = a
            if time.time() > deadline:
                break

    return [best_a, N - best_a]
