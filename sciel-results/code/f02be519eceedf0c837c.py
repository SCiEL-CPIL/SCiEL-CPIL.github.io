"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def construct():
    deadline = time.time() + 55.0  # safety margin before 90s hard kill

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
    n = len(primes)
    prime_index = {p: i for i, p in enumerate(primes)}
    primes_arr = np.array(primes, dtype=np.int32)

    best = []

    def validate(chain):
        if not chain:
            return False
        if chain[-1] > N:
            return False
        for p in chain:
            if not is_prime[p]:
                return False
        if len(chain) < 2:
            return True
        for a, b in zip(chain, chain[1:]):
            if b <= a:
                return False
        for i in range(1, len(chain) - 1):
            if chain[i + 1] - chain[i] < chain[i] - chain[i - 1]:
                return False
        return True

    # ---- Phase 1: Greedy from each prime with multiple strategies ----
    def greedy_from(start_idx, strategy='min'):
        if start_idx >= n:
            return [primes[start_idx]]
        chain = [primes[start_idx]]
        last_gap = 0
        for j in range(start_idx + 1, n):
            p = primes[j]
            gap = p - chain[-1]
            if gap >= last_gap:
                if strategy == 'min':
                    chain.append(p)
                    last_gap = gap
                    break  # Take the first valid (smallest gap)
                elif strategy == 'max':
                    chain.append(p)
                    last_gap = gap
                elif strategy == 'median':
                    # Collect all valid, take middle
                    pass
        if strategy == 'max':
            # Continue with max gap strategy
            last_gap = 0
            chain = [primes[start_idx]]
            for j in range(start_idx + 1, n):
                p = primes[j]
                gap = p - chain[-1]
                if gap >= last_gap:
                    chain.append(p)
                    last_gap = gap
        return chain

    for i in range(n):
        if time.time() > deadline:
            break
        for strat in ['min', 'max']:
            chain = greedy_from(i, strat)
            if validate(chain) and len(chain) > len(best):
                best = chain

    # ---- Phase 2: Sparse DP - only consider gaps where p_j - g is prime ----
    # For each prime j, precompute list of (gap, prev_prime_idx) where p_j - gap is prime
    # dp[j][g_idx] = max length ending at prime j with last gap = gaps_list[j][g_idx]
    # Use dict-based DP for sparse gaps

    # Precompute for each prime j, the list of (gap, prev_prime_idx) for all i < j
    prev_primes_for_j = [[] for _ in range(n)]
    for j in range(n):
        p_j = primes[j]
        for i in range(j):
            gap = p_j - primes[i]
            prev_primes_for_j[j].append((gap, i))
        if time.time() > deadline:
            break

    # dp_vals[j] = dict: gap -> max chain length ending at j with last gap = gap
    dp_vals = [dict() for _ in range(n)]
    # For fast suffix max queries, maintain sorted gap lists with prefix max
    # suffix_info[j] = (sorted_gaps, prefix_max) where prefix_max[k] = max of dp_vals[j] for gaps[0..k]
    suffix_info = [([], []) for _ in range(n)]  # (sorted_gaps_list, prefix_max_list)

    best_len = 1
    best_j = 0
    best_g = 0

    for j in range(n):
        if time.time() > deadline:
            break
        p_j = primes[j]

        # For each previous prime i, gap = p_j - p_i
        for gap, i in prev_primes_for_j[j]:
            # Query suffix max for prime i: max of dp_vals[i][h] for h <= gap
            sg, pm = suffix_info[i]
            if not sg:
                max_prev = 0
            else:
                # Binary search for rightmost index where sg[idx] <= gap
                lo, hi = 0, len(sg) - 1
                idx = -1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if sg[mid] <= gap:
                        idx = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if idx >= 0:
                    max_prev = pm[idx]
                else:
                    max_prev = 0

            new_val = 1 + max_prev
            old_val = dp_vals[j].get(gap, 0)
            if new_val > old_val:
                dp_vals[j][gap] = new_val
                if new_val > best_len:
                    best_len = new_val
                    best_j = j
                    best_g = gap

        # After processing all incoming edges to j, build suffix info for j
        if dp_vals[j]:
            gaps_sorted = sorted(dp_vals[j].keys())
            vals = [dp_vals[j][g] for g in gaps_sorted]
            # Compute prefix max
            prefix_max = [vals[0]]
            for k in range(1, len(vals)):
                prefix_max.append(max(prefix_max[-1], vals[k]))
            suffix_info[j] = (gaps_sorted, prefix_max)

    # Reconstruct chain from best
    if best_len > len(best):
        chain = []
        j = best_j
        g = best_g
        while j is not None and g is not None:
            chain.append(primes[j])
            if g == 0:
                break
            p_prev = primes[j] - g
            i = prime_index.get(p_prev)
            if i is None:
                break
            # Find the h in dp_vals[i] with h <= g that maximizes dp_vals[i][h]
            sg, pm = suffix_info[i]
            if sg:
                # Binary search for rightmost index where sg[idx] <= g
                lo, hi = 0, len(sg) - 1
                idx = -1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if sg[mid] <= g:
                        idx = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if idx >= 0:
                    # The best h is the one that achieved pm[idx]
                    # We need to find which gap h gave value pm[idx]
                    # pm[idx] = max(dp_vals[i][sg[0..idx]])
                    # Find the last gap in sg[0..idx] that achieves this max
                    best_h = sg[0]
                    for k in range(idx + 1):
                        if dp_vals[i][sg[k]] == pm[idx]:
                            best_h = sg[k]
                            break
                    j = i
                    g = best_h
                    continue
            break
        chain.reverse()
        if validate(chain) and len(chain) > len(best):
            best = chain

    # ---- Phase 3: Beam search refinement (inspired by inspiration 1) ----
    # Use beam search to try to find longer chains, starting from the best found so far
    # and also from other promising starting points

    def beam_search_from(start_chain, max_depth=8, beam_width=50, gap_window=4):
        if time.time() > deadline:
            return start_chain

        # State: (chain, last_gap)
        states = [(list(start_chain), start_chain[-1] - start_chain[-2] if len(start_chain) >= 2 else 0)]
        best_local = list(start_chain)

        for depth in range(max_depth):
            if time.time() > deadline:
                break
            next_states = []
            for chain, lg in states:
                if len(chain) >= max_depth:
                    continue
                last = chain[-1]
                idx = prime_index.get(last)
                if idx is None:
                    continue
                # Try gaps from lg to lg + gap_window*2
                lo = lg
                hi = lg + gap_window * 2
                for j in range(idx + 1, n):
                    p = primes[j]
                    if p > N:
                        break
                    gap = p - last
                    if gap < lo:
                        continue
                    if gap > hi:
                        break
                    next_states.append((chain + [p], gap))
                    if len(next_states) > beam_width * 5:
                        break
                if len(next_states) > beam_width * 5:
                    break

            if not next_states:
                break

            # Prune: keep top beam_width states
            next_states.sort(key=lambda s: (-len(s[0]), s[1], s[0][-1]))
            seen = set()
            pruned = []
            for s in next_states:
                key = tuple(s[0])
                if key in seen:
                    continue
                seen.add(key)
                pruned.append(s)
                if len(pruned) >= beam_width:
                    break
            states = pruned

            # Update best
            for chain, _ in states:
                if len(chain) > len(best_local):
                    best_local = list(chain)

        return best_local

    # Try beam search from the current best chain
    if best and len(best) >= 2:
        bc = beam_search_from(best, max_depth=10, beam_width=100, gap_window=3)
        if validate(bc) and len(bc) > len(best):
            best = bc

    # Try beam search from a few other promising starting points
    for start_p in [2, 3, 5, 7, 11, 13]:
        if time.time() > deadline:
            break
        start_chain = [start_p]
        # Greedily extend a bit to get a decent starting chain
        last_gap = 0
        idx = prime_index.get(start_p)
        if idx is not None:
            for j in range(idx + 1, min(idx + 10, n)):
                p = primes[j]
                gap = p - start_chain[-1]
                if gap >= last_gap:
                    start_chain.append(p)
                    last_gap = gap
        if len(start_chain) >= 2:
            bc = beam_search_from(start_chain, max_depth=10, beam_width=50, gap_window=3)
            if validate(bc) and len(bc) > len(best):
                best = bc

    # ---- Phase 4: Local search refinement ----
    # Try to extend best chain by trying different next primes
    if best and len(best) >= 2:
        improved = True
        while improved:
            improved = False
            last = best[-1]
            last_gap = best[-1] - best[-2]
            start_idx = prime_index.get(last)
            if start_idx is None:
                break
            # Try all valid next primes, pick the one that leads to longest extension
            best_next = None
            best_next_len = 0
            for j in range(start_idx + 1, n):
                p = primes[j]
                gap = p - last
                if gap < last_gap:
                    continue
                # Greedily extend from p with gap
                test_chain = best + [p]
                lg = gap
                for k in range(j + 1, n):
                    pk = primes[k]
                    gk = pk - test_chain[-1]
                    if gk >= lg:
                        test_chain.append(pk)
                        lg = gk
                if len(test_chain) > best_next_len:
                    best_next_len = len(test_chain)
                    best_next = test_chain
            if best_next and best_next_len > len(best):
                best = best_next
                improved = True

    # Also try local search: replace last element with alternatives
    if best and len(best) >= 3:
        last_gap = best[-1] - best[-2]
        prev = best[-2]
        prev_idx = prime_index.get(prev)
        if prev_idx is not None:
            # Try different choices for the last prime
            for j in range(prev_idx + 1, n):
                p = primes[j]
                gap = p - prev
                if gap < last_gap:
                    continue
                # Extend from p
                test_chain = best[:-1] + [p]
                lg = gap
                for k in range(j + 1, n):
                    pk = primes[k]
                    gk = pk - test_chain[-1]
                    if gk >= lg:
                        test_chain.append(pk)
                        lg = gk
                if len(test_chain) > len(best):
                    best = test_chain
                    break

    if not validate(best):
        best = [2, 3, 5]

    return best
# EVOLVE-BLOCK-END
