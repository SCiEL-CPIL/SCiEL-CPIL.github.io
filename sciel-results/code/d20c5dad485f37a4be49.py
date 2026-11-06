"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import isqrt

    deadline = time.time() + 55.0  # safety margin before 60s hard kill

    # ---- Sieve of primes up to 10000 ----
    N = 10000
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start::step] = [False] * (((N - start) // step) + 1)
    primes = [i for i, v in enumerate(is_prime) if v]
    P = len(primes)

    # For each prime j, maintain a Pareto frontier of states.
    # A state is (length, last_gap, prev_prime_idx, prev_state_idx)
    # where prev_state_idx is the index into the frontier of the prev prime.
    # Dominance: state A dominates B if A.length >= B.length and A.last_gap <= B.last_gap.
    # We keep only non-dominated states, sorted by last_gap ascending.
    # For a given gap g, the best state with last_gap <= g is the one with max length among those with gap <= g.
    # Since frontier is sorted by gap ascending, we can binary search.

    # frontier[j] = list of (length, last_gap, prev_idx, prev_state_idx)
    # sorted by last_gap ascending. Within same gap, keep only max length.
    # Additionally, we maintain prefix_max_len[j] = prefix max of lengths in frontier[j]
    # so that for a query gap g, we find the largest index with last_gap <= g,
    # then the best length is prefix_max_len at that index.

    import bisect

    frontiers = [[] for _ in range(P)]  # each is list of (length, last_gap, prev_idx, prev_state_idx)
    prefix_lens = [[] for _ in range(P)]  # prefix max of lengths
    gaps_lists = [[] for _ in range(P)]   # list of gaps for bisect
    prefix_argmax = [[] for _ in range(P)] # index in frontier that achieves prefix_lens[k]

    # Initialize: each prime can be a sequence of length 1 with last_gap = 0
    for j in range(P):
        frontiers[j] = [(1, 0, -1, -1)]
        prefix_lens[j] = [1]
        gaps_lists[j] = [0]
        prefix_argmax[j] = [0]

    # Process primes in order
    for j in range(1, P):
        if time.time() > deadline:
            break
        p_j = primes[j]
        new_states = []

        for i in range(j):
            if time.time() > deadline:
                break
            p_i = primes[i]
            g = p_j - p_i

            # Find best state in frontiers[i] with last_gap <= g
            fr = frontiers[i]
            if not fr:
                continue

            # Use pre-computed gaps list for bisect
            gaps_i = gaps_lists[i]
            idx = bisect.bisect_right(gaps_i, g) - 1
            if idx < 0:
                continue

            # Get best length and the specific state index directly
            best_len = prefix_lens[i][idx]
            best_state_idx = prefix_argmax[i][idx]

            new_len = best_len + 1
            new_states.append((new_len, g, i, best_state_idx))

        # Add the "start new sequence" state
        new_states.append((1, 0, -1, -1))

        # Build Pareto frontier for j
        # Sort by last_gap ascending, then length descending
        new_states.sort(key=lambda x: (x[1], -x[0]))

        # Remove dominated states
        filtered = []
        max_len_so_far = 0
        for s in new_states:
            length, gap, prev_idx, prev_si = s
            # A state is dominated if there's a kept state with length >= this length and gap <= this gap
            # Since we process in order of increasing gap, we just check if length > max_len_so_far
            if length > max_len_so_far:
                filtered.append(s)
                max_len_so_far = length

        frontiers[j] = filtered

        # Build prefix_lens, gaps_lists, and prefix_argmax for j
        pl = []
        gl = []
        pa = []
        cur_max = 0
        cur_argmax = -1
        for k, s in enumerate(filtered):
            if s[0] > cur_max:
                cur_max = s[0]
                cur_argmax = k
            pl.append(cur_max)
            gl.append(s[1])
            pa.append(cur_argmax)
        
        prefix_lens[j] = pl
        gaps_lists[j] = gl
        prefix_argmax[j] = pa

    # Find the best sequence
    best_len = 0
    best_j = -1
    best_state_idx = -1

    for j in range(P):
        if frontiers[j]:
            # The max length for prime j is the max over all states
            # Since prefix_lens[j] is non-decreasing, the last element is the max
            max_len_j = prefix_lens[j][-1] if prefix_lens[j] else 0
            if max_len_j > best_len:
                best_len = max_len_j
                best_j = j
                # Find the state index that achieves this max
                # It's the last state in filtered with length == max_len_j
                for k in range(len(frontiers[j]) - 1, -1, -1):
                    if frontiers[j][k][0] == max_len_j:
                        best_state_idx = k
                        break

    # Backtrack to reconstruct the sequence
    path = []
    curr_j = best_j
    curr_state_idx = best_state_idx

    while curr_j != -1:
        length, gap, prev_idx, prev_si = frontiers[curr_j][curr_state_idx]
        path.append(primes[curr_j])
        curr_j = prev_idx
        curr_state_idx = prev_si

    path.reverse()

    return path
# EVOLVE-BLOCK-END
