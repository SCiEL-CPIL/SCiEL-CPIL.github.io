import time
import random

# Modular sieve primes for pre-filtering
_SIEVE_PRIMES = [3, 5, 7, 11, 13, 17, 19, 23]

def _modular_conflicts(a, x, n):
    """For element a and new x, count how many subset sizes k in [2, n+1]
    survive the modular sieve (i.e., NOT excluded by any prime).
    Returns (surviving_k_list, total_checked)."""
    surviving = []
    for k in range(2, n + 2):
        excluded = False
        for p in _SIEVE_PRIMES:
            if (k * a - x) % p == 0:
                # This k is excluded by prime p: no subset of size k-1 from others
                # can have sum ≡ k*a - x (mod p) unless... wait.
                # Actually the condition is: if NO subset of size k-1 from others
                # has sum ≡ (k*a - x) mod p, then k is excluded.
                # But we don't know that. The sieve only helps if we can prove
                # impossibility. For a necessary condition: if (k*a - x) mod p
                # is achievable by some subset sum, it might be valid.
                # The sieve should EXCLUDE k when (k*a - x) mod p is NOT achievable.
                # But we don't precompute that. So this simple congruence check
                # doesn't directly exclude.
                # Let me reconsider: the directive says "2x is not congruent to
                # the sum of any pair modulo p". That's for k=2.
                # For general k, we'd need to know achievable residues.
                # This simple approach won't work as a sound filter.
                pass
        surviving.append(k)
    return surviving

# The simple modular congruence check doesn't give a sound filter without
# knowing achievable residues. Let me use a different approach:
# Precompute for each prime p and each subset size j, the set of achievable
# residues mod p from the current set. But that's expensive to maintain.
#
# Alternative: use the modular sieve only for the k=2 case (pair sums),
# which is the most common conflict. For k=2: we need a+b = 2x, i.e., b = 2x - a.
# This is O(n) per candidate and doesn't need DP.
#
# Let me focus on a different structural improvement: better candidate ordering
# and more aggressive local search.

def _build_dp(elements):
    """Build DP bitsets for subset sums by cardinality."""
    n = len(elements)
    dp = [0] * (n + 1)
    dp[0] = 1
    for e in elements:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << e)
    return dp

def _is_valid_mean_free(A, x, dp_S, dp_minus):
    """
    Check if adding x to A keeps the set mean-free.
    A is a list of distinct integers.
    dp_S[k] = bitset of sums of k elements from A.
    dp_minus[a][k] = bitset of sums of k elements from A \ {a}.
    Returns True if A U {x} is valid.
    """
    n = len(A)

    # 1. Check if x is the mean of any subset of A of size k >= 2
    if n >= 2:
        for k in range(2, n + 1):
            target = k * x
            if target > 512 * 48:
                break
            if (dp_S[k] >> target) & 1:
                return False

    # 2. Check if any a in A is the mean of any subset of (A U {x}) \ {a} of size k >= 2
    # Such a subset must contain x.
    # S = T U {x}, T subset of A \ {a}.
    # |S| = k, |T| = k - 1 = j.
    # Sum(T) = k * a - x.
    for a in A:
        if a not in dp_minus:
            others = [v for v in A if v != a]
            dp_minus[a] = _build_dp(others)
        dp_a = dp_minus[a]
        m = n - 1
        for k in range(2, n + 1):
            target_sum = k * a - x
            if target_sum < 0:
                continue
            j = k - 1
            if j > m:
                break
            if j < 0:
                continue
            if (dp_a[j] >> target_sum) & 1:
                return False

    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    best_set = []
    best_len = 0

    def greedy_generate(seed_val):
        """Greedy construction with maintained DP."""
        random.seed(seed_val)
        candidates = list(range(1, 513))
        random.shuffle(candidates)

        S = []
        dp_S = [0] * 50
        dp_S[0] = 1
        dp_minus = {}

        for x in candidates:
            if time.time() > deadline:
                break
            if len(S) >= 48:
                break

            if _is_valid_mean_free(S, x, dp_S, dp_minus):
                S.append(x)
                n = len(S) - 1
                # Update dp_S
                for k in range(n, -1, -1):
                    if dp_S[k]:
                        dp_S[k + 1] |= (dp_S[k] << x)
                # Recompute dp_minus for all elements
                dp_minus = {}
                for a in S:
                    others = [v for v in S if v != a]
                    dp_minus[a] = _build_dp(others)

        return S

    def local_search(S, time_budget):
        """Local search: try additions and swaps to grow S."""
        ls_start = time.time()
        ls_deadline = min(start_time + time_budget, deadline)
        no_improvement = 0

        # Build initial DP
        dp_S = _build_dp(S)
        dp_minus = {}
        for a in S:
            others = [v for v in S if v != a]
            dp_minus[a] = _build_dp(others)

        S_set = set(S)

        while time.time() < ls_deadline:
            improved = False

            # Try additions
            for _ in range(15):
                if time.time() > ls_deadline:
                    break
                x = random.randint(1, 512)
                if x in S_set:
                    continue
                if _is_valid_mean_free(S, x, dp_S, dp_minus):
                    S.append(x)
                    S_set.add(x)
                    n = len(S) - 1
                    for k in range(n, -1, -1):
                        if dp_S[k]:
                            dp_S[k + 1] |= (dp_S[k] << x)
                    dp_minus = {}
                    for a in S:
                        others = [v for v in S if v != a]
                        dp_minus[a] = _build_dp(others)
                    improved = True
                    no_improvement = 0
                    break

            if improved:
                continue

            # Try swaps
            for _ in range(15):
                if time.time() > ls_deadline:
                    break
                a = random.choice(S)
                x = random.randint(1, 512)
                if x in S_set:
                    continue
                others = [v for v in S if v != a]
                if _is_valid_mean_free(others, x, _build_dp(others), {}):
                    S.remove(a)
                    S.append(x)
                    S_set.discard(a)
                    S_set.add(x)
                    # Rebuild DP
                    dp_S = _build_dp(S)
                    dp_minus = {}
                    for el in S:
                        others2 = [v for v in S if v != el]
                        dp_minus[el] = _build_dp(others2)
                    improved = True
                    no_improvement = 0
                    break

            if not improved:
                no_improvement += 1
                if no_improvement > 15:
                    break

        return S

    # Multiple restarts with greedy + local search
    for restart in range(30):
        if time.time() > deadline:
            break

        S = greedy_generate(1000 + restart * 7)

        if len(S) > best_len:
            best_len = len(S)
            best_set = S[:]

        if best_len >= 48:
            break

        # Local search to try to grow
        remaining = deadline - time.time()
        if remaining > 5.0:
            ls_budget = min(remaining * 0.4, 6.0)
            S = local_search(S, ls_budget)

            if len(S) > best_len:
                best_len = len(S)
                best_set = S[:]

            if best_len >= 48:
                break

    return best_set
