import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s

    # ---------- Bitset DP feasibility oracle ----------
    # dp[k] is a Python int bitset: bit s set iff sum s is achievable
    # by a subset of the current set S with cardinality k.
    # all_sums = OR over all dp[k].
    # Invariant: dp[a] & dp[b] == 0 for a != b.
    #
    # can_add(S_state, x) -> bool:
    #   For each k in 1..n+1, new_sums_k = dp[k-1] << x.
    #   Conflict if new_sums_k has a bit in all_sums that is not in dp[k].
    #   i.e., (new_sums_k & (all_sums & ~dp[k])) != 0.
    #
    # commit(S_state, x):
    #   Update dp in reverse order, update all_sums.

    class State:
        __slots__ = ('S', 'n', 'dp', 'all_sums')
        def __init__(self):
            self.S = []
            self.n = 0
            self.dp = [0] * 61
            self.dp[0] = 1
            self.all_sums = 1

        def can_add(self, x):
            n = self.n
            dp = self.dp
            all_sums = self.all_sums
            # For k in 1..n+1
            for k in range(1, n + 2):
                ns = dp[k - 1] << x
                if ns & (all_sums & ~dp[k]):
                    return False
            return True

        def commit(self, x):
            n = self.n
            dp = self.dp
            old_dp = dp[:]
            for k in range(n, 0, -1):
                dp[k] = old_dp[k] | (old_dp[k - 1] << x)
            # dp[0] stays 1
            self.all_sums |= (self.all_sums << x)
            self.S.append(x)
            self.n += 1

        def rollback(self):
            # Undo last commit
            x = self.S.pop()
            self.n -= 1
            n = self.n
            dp = self.dp
            # Recompute dp from scratch? Too slow.
            # Instead, we need to revert. Since we only add, we can't easily
            # revert a bitset. So we use a stack of (dp, all_sums) snapshots
            # when backtracking. For the greedy pass we don't need rollback.
            # For backtracking, we'll snapshot.
            pass

    def greedy_build(order, deadline):
        st = State()
        S_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if st.can_add(x):
                st.commit(x)
                S_set.add(x)
        return st.S

    def backtracking_search(candidates, max_size, deadline, depth=0):
        # Search for a set of size max_size from candidates.
        # Returns a list or None.
        st = State()
        S_set = set()

        def backtrack(idx, remaining):
            if time.time() > deadline:
                return None
            if remaining == 0:
                return st.S[:]
            if idx >= len(candidates):
                return None

            # Pruning: not enough candidates left
            if len(candidates) - idx < remaining:
                return None

            for i in range(idx, len(candidates)):
                if time.time() > deadline:
                    return None
                x = candidates[i]
                if x in S_set:
                    continue
                if not st.can_add(x):
                    continue
                # Commit
                st.commit(x)
                S_set.add(x)
                res = backtrack(i + 1, remaining - 1)
                if res is not None:
                    return res
                # Rollback: we need to revert st
                # Since rollback is hard with bitsets, we snapshot instead.
                # For simplicity in this search, we'll use a different approach:
                # We'll pass dp state as part of recursion. But that's heavy.
                # Alternative: just use greedy with multiple orders and a repair step.
                # Let's abandon backtracking and use a smarter greedy + repair.
                st.S.pop()
                st.n -= 1
                # We can't easily revert dp. So this backtracking is flawed.
                # Let's just break out and use a different strategy.
                return None

        return backtrack(0, max_size)

    # Since backtracking with bitset rollback is tricky, I'll use a different
    # structural approach: **iterative deepening greedy with adaptive reordering**.
    # Start with a greedy pass. Then, try to improve by re-ordering the
    # remaining candidates and re-running greedy from a partial state.
    # Also, try local search: swap an included element for an excluded one.

    def greedy_with_repair(order, deadline):
        st = State()
        S_set = set()
        included = []
        for x in order:
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if st.can_add(x):
                st.commit(x)
                S_set.add(x)
                included.append(x)

        # Repair: try to add more from the remaining candidates in a
        # different order (e.g., descending, or by "least conflict" heuristic).
        remaining = [x for x in order if x not in S_set]
        # Sort remaining by descending value (larger numbers create bigger gaps)
        remaining.sort(reverse=True)
        for x in remaining:
            if time.time() > deadline:
                break
            if st.can_add(x):
                st.commit(x)
                S_set.add(x)
                included.append(x)

        return st.S

    # Local improvement: try to replace one element with a better one.
    def local_improve(S, deadline):
        # For each element in S, try removing it and re-adding a different one.
        # This is expensive, so do it sparingly.
        if time.time() > deadline:
            return S
        st = State()
        for x in S:
            st.commit(x)
        # Try to swap: remove x_i, try to add y from 1..256 not in S
        improved = False
        S_list = S[:]
        for i in range(len(S_list)):
            if time.time() > deadline:
                break
            # Build state without S_list[i]
            st2 = State()
            for j, x in enumerate(S_list):
                if j == i:
                    continue
                if st2.can_add(x):
                    st2.commit(x)
            # Try to add a replacement
            for y in range(1, 257):
                if time.time() > deadline:
                    break
                if y in S_list:
                    continue
                if st2.can_add(y):
                    # Check if this replacement is "better" in some sense
                    # For now, just accept any replacement that keeps size
                    st2.commit(y)
                    new_S = st2.S[:]
                    # Replace S_list[i] with y and continue
                    S_list[i] = y
                    improved = True
                    break
        return S_list

    # Generate diverse orders
    rng = random.Random(12345)
    orders = []

    # 1. Descending (largest first) - cell finding says this helps
    orders.append(list(range(256, 0, -1)))

    # 2. Ascending
    orders.append(list(range(1, 257)))

    # 3. Alternating large/small
    large = list(range(256, 127, -1))
    small = list(range(1, 129))
    alt = []
    for i in range(128):
        alt.append(large[i])
        alt.append(small[i])
    orders.append(alt)

    # 4. Random shuffles with bias toward larger numbers
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        o = list(range(1, 257))
        # Weight larger numbers more heavily in early positions
        # Use a custom shuffle: assign weights
        weights = [1.0 / (257 - x) for x in o]  # larger x -> larger weight
        # Sample without replacement using weights
        remaining = o[:]
        shuffled = []
        for _ in range(256):
            if time.time() > deadline - 5:
                break
            total_w = sum(weights[remaining.index(x)] for x in remaining)
            r = rng.random() * total_w
            cum = 0
            for idx, x in enumerate(remaining):
                cum += weights[remaining.index(x)]
                if r <= cum:
                    shuffled.append(x)
                    remaining.remove(x)
                    break
        if len(shuffled) == 256:
            orders.append(shuffled)

    # 5. Spaced orders (multiples of step)
    for step in [2, 3, 4, 5, 8, 16]:
        o = []
        for i in range(256, 0, -step):
            o.append(i)
        for v in range(256, 0, -1):
            if v not in o:
                o.append(v)
        orders.append(o)

    # 6. Random with different seeds
    for seed in range(100):
        if time.time() > deadline - 5:
            break
        r = random.Random(seed)
        o = list(range(1, 257))
        r.shuffle(o)
        orders.append(o)

    best_set = []

    for o in orders:
        if time.time() > deadline:
            break
        S = greedy_with_repair(o, deadline)
        if len(S) > len(best_set):
            best_set = S
            # Try local improvement on the best found so far
            if len(best_set) >= 10 and time.time() < deadline - 10:
                improved = local_improve(best_set, deadline)
                if len(improved) > len(best_set):
                    best_set = improved

    return [int(x) for x in best_set]
