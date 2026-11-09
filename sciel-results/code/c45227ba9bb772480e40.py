import time
import random
import math
import sys


def construct():
    start = time.time()
    DEADLINE = start + 145.0  # 5s safety margin before 150s kill

    prescribed = [1, 2, 5, 11]
    pres_set = set(prescribed)
    N = 284
    random.seed(42)

    def remaining():
        return DEADLINE - time.time()

    def is_valid_set(s):
        if len(s) != len(set(s)):
            return False
        if not pres_set.issubset(set(s)):
            return False
        for v in s:
            if v < 1 or v > N:
                return False
        seen = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                v = s[i] + s[j]
                if v in seen:
                    return False
                seen.add(v)
        return True

    # ---------- Greedy baseline ----------
    def greedy_build():
        arr = list(prescribed)
        used = set(arr)
        sums = set()
        for a in arr:
            for b in arr:
                if a <= b:
                    sums.add(a + b)
        for x in range(1, N + 1):
            if x in used:
                continue
            ok = True
            new_sums = []
            for a in arr:
                s = a + x
                if s in sums:
                    ok = False
                    break
                new_sums.append(s)
            if ok and (x + x) in sums:
                ok = False
            if ok:
                sums.update(new_sums)
                sums.add(x + x)
                used.add(x)
                arr.append(x)
        return arr

    def greedy_build_reversed():
        arr = list(prescribed)
        used = set(arr)
        sums = set()
        for a in arr:
            for b in arr:
                if a <= b:
                    sums.add(a + b)
        for x in range(N, 0, -1):
            if x in used:
                continue
            ok = True
            new_sums = []
            for a in arr:
                s = a + x
                if s in sums:
                    ok = False
                    break
                new_sums.append(s)
            if ok and (x + x) in sums:
                ok = False
            if ok:
                sums.update(new_sums)
                sums.add(x + x)
                used.add(x)
                arr.append(x)
        return arr

    def greedy_build_random(seed_val):
        rng = random.Random(seed_val)
        arr = list(prescribed)
        used = set(arr)
        sums = set()
        for a in arr:
            for b in arr:
                if a <= b:
                    sums.add(a + b)
        pool = [x for x in range(1, N + 1) if x not in used]
        rng.shuffle(pool)
        for x in pool:
            ok = True
            new_sums = []
            for a in arr:
                s = a + x
                if s in sums:
                    ok = False
                    break
                new_sums.append(s)
            if ok and (x + x) in sums:
                ok = False
            if ok:
                sums.update(new_sums)
                sums.add(x + x)
                used.add(x)
                arr.append(x)
        return arr

    try:
        # Get multiple greedy baselines
        g1 = greedy_build()
        g2 = greedy_build_reversed()
        g3 = greedy_build_random(123)
        g4 = greedy_build_random(456)
        g5 = greedy_build_random(789)
        candidates = [g1, g2, g3, g4, g5]
        best = max(candidates, key=len)
        best_size = len(best)

        # ---------- Incremental state for SA ----------
        class State:
            __slots__ = ('arr', 'n', 'counts', 'conflict', 'val_counts', 'fixed')

            def __init__(self, arr, fixed):
                self.arr = arr
                self.n = len(arr)
                self.fixed = fixed
                self.counts = {}
                self.conflict = 0
                self.val_counts = {}
                for i in range(self.n):
                    vi = arr[i]
                    self.val_counts[vi] = self.val_counts.get(vi, 0) + 1
                    for j in range(i, self.n):
                        s = vi + arr[j]
                        c = self.counts.get(s, 0)
                        if c == 1:
                            self.conflict += 1
                        self.counts[s] = c + 1

            def _inc_sum(self, s):
                c = self.counts.get(s, 0)
                if c == 1:
                    self.conflict += 1
                self.counts[s] = c + 1

            def _dec_sum(self, s):
                c = self.counts[s]
                if c == 2:
                    self.conflict -= 1
                if c == 1:
                    del self.counts[s]
                else:
                    self.counts[s] = c - 1

            def propose(self, i, new_val):
                old = self.arr[i]
                if old == new_val:
                    return None
                if self.val_counts.get(new_val, 0) >= 1:
                    return None
                for j in range(self.n):
                    self._dec_sum(old + self.arr[j])
                self.val_counts[old] -= 1
                if self.val_counts[old] == 0:
                    del self.val_counts[old]
                self.arr[i] = new_val
                for j in range(self.n):
                    self._inc_sum(new_val + self.arr[j])
                self.val_counts[new_val] = self.val_counts.get(new_val, 0) + 1
                return self.conflict

            def revert(self, i, old_val, new_val):
                for j in range(self.n):
                    self._dec_sum(new_val + self.arr[j])
                self.val_counts[new_val] -= 1
                if self.val_counts[new_val] == 0:
                    del self.val_counts[new_val]
                self.arr[i] = old_val
                for j in range(self.n):
                    self._inc_sum(old_val + self.arr[j])
                self.val_counts[old_val] = self.val_counts.get(old_val, 0) + 1

            def swap(self, i, j):
                """Swap arr[i] and arr[j]. Returns conflict after swap."""
                if i == j:
                    return None
                old_i = self.arr[i]
                old_j = self.arr[j]
                if old_i == old_j:
                    return None
                # Remove sums involving old_i and old_j
                for k in range(self.n):
                    self._dec_sum(old_i + self.arr[k])
                    self._dec_sum(old_j + self.arr[k])
                # Swap
                self.arr[i] = old_j
                self.arr[j] = old_i
                # Add new sums
                for k in range(self.n):
                    self._inc_sum(old_j + self.arr[k])
                    self._inc_sum(old_i + self.arr[k])
                return self.conflict

            def unswap(self, i, j):
                """Undo a swap."""
                old_i = self.arr[j]
                old_j = self.arr[i]
                for k in range(self.n):
                    self._dec_sum(old_j + self.arr[k])
                    self._dec_sum(old_i + self.arr[k])
                self.arr[i] = old_i
                self.arr[j] = old_j
                for k in range(self.n):
                    self._inc_sum(old_i + self.arr[k])
                    self._inc_sum(old_j + self.arr[k])

        def run_sa(target, init, budget, T_start=None, T_end=None, cool=None):
            end = min(time.time() + budget, DEADLINE - 3.0)
            fixed = [v in pres_set for v in init]
            st = State(init, fixed)
            best_obj = st.conflict
            best_arr = list(init)

            if T_start is None:
                T_start = max(1.0, best_obj * 0.5)
            if T_end is None:
                T_end = 0.01
            if cool is None:
                cool = 0.9999

            T = T_start
            n = target
            no_improve_count = 0
            step_count = 0

            while time.time() < end and T > T_end:
                # Choose move type
                r = random.random()
                if r < 0.6:
                    # 1-opt: replace one non-fixed element
                    i = random.randrange(n)
                    if fixed[i]:
                        continue
                    old = st.arr[i]
                    cur = st.conflict

                    new = random.randint(1, N)
                    nc = st.propose(i, new)
                    if nc is None:
                        continue

                    step_count += 1

                    if nc <= cur:
                        if nc < best_obj:
                            best_obj = nc
                            best_arr = list(st.arr)
                            no_improve_count = 0
                            if nc == 0:
                                return best_arr
                        else:
                            no_improve_count += 1
                    else:
                        if random.random() < math.exp((cur - nc) / T):
                            no_improve_count += 1
                        else:
                            st.revert(i, old, new)
                            no_improve_count += 1
                else:
                    # 2-opt: swap two non-fixed elements
                    i = random.randrange(n)
                    j = random.randrange(n)
                    if fixed[i] or fixed[j] or i == j:
                        continue
                    cur = st.conflict
                    nc = st.swap(i, j)
                    if nc is None:
                        continue

                    step_count += 1

                    if nc <= cur:
                        if nc < best_obj:
                            best_obj = nc
                            best_arr = list(st.arr)
                            no_improve_count = 0
                            if nc == 0:
                                return best_arr
                        else:
                            no_improve_count += 1
                    else:
                        if random.random() < math.exp((cur - nc) / T):
                            no_improve_count += 1
                        else:
                            st.unswap(i, j)
                            no_improve_count += 1

                T *= cool

                # Early exit if stuck
                if no_improve_count > 2000 and best_obj > 0:
                    break

            if best_obj == 0:
                return best_arr
            return None

        def local_descent(arr, fixed, budget):
            """Aggressive local descent to reduce conflicts."""
            end = min(time.time() + budget, DEADLINE - 3.0)
            st = State(arr, fixed)
            n = st.n
            best_obj = st.conflict
            best_arr = list(arr)

            while time.time() < end and best_obj > 0:
                improved = False
                # Try all 1-opt moves
                for i in range(n):
                    if fixed[i]:
                        continue
                    old = st.arr[i]
                    cur = st.conflict
                    # Try a few random values
                    for _ in range(5):
                        new = random.randint(1, N)
                        nc = st.propose(i, new)
                        if nc is None:
                            continue
                        if nc < cur:
                            cur = nc
                            improved = True
                            break
                        else:
                            st.revert(i, old, new)
                    if improved:
                        break

                if not improved:
                    # Try swaps
                    for i in range(n):
                        if fixed[i]:
                            continue
                        for j in range(i + 1, n):
                            if fixed[j]:
                                continue
                            cur = st.conflict
                            nc = st.swap(i, j)
                            if nc is None:
                                continue
                            if nc < cur:
                                improved = True
                                break
                            else:
                                st.unswap(i, j)
                        if improved:
                            break

                if not improved:
                    break

                if st.conflict < best_obj:
                    best_obj = st.conflict
                    best_arr = list(st.arr)

            return best_arr

        def make_init(target, diversity=0.0):
            base = list(best)
            used = set(base)
            need = target - len(base)
            if need <= 0:
                return base[:target]
            pool = [x for x in range(1, N + 1) if x not in used]
            random.shuffle(pool)
            if diversity < 0.3:
                return base + pool[:need]
            elif diversity < 0.6:
                keep_count = max(4, int(len(base) * (1 - diversity)))
                keep = list(base[:keep_count])
                used2 = set(keep)
                need2 = target - len(keep)
                pool2 = [x for x in range(1, N + 1) if x not in used2]
                random.shuffle(pool2)
                return keep + pool2[:need2]
            else:
                keep = list(prescribed)
                used2 = set(keep)
                need2 = target - len(keep)
                pool2 = [x for x in range(1, N + 1) if x not in used2]
                random.shuffle(pool2)
                return keep + pool2[:need2]

        def sa_loop(target, budget):
            t0 = time.time()
            while time.time() - t0 < budget and remaining() > 5:
                sub = min(4.0, budget - (time.time() - t0), remaining() - 5)
                if sub <= 0.3:
                    break
                diversity = random.random()
                init = make_init(target, diversity)
                T_start = random.uniform(0.5, 2.0)
                T_end = random.uniform(0.005, 0.02)
                cool = random.uniform(0.9998, 0.99999)
                res = run_sa(target, init, sub, T_start, T_end, cool)
                if res is not None and is_valid_set(res):
                    return res
            return None

        # ---------- Incremental growth to maximize size ----------
        for target in range(best_size + 1, 21):
            if remaining() < 10:
                break
            budget = min(20.0, remaining() - 5)
            found = sa_loop(target, budget)
            if found is not None:
                best = found
                best_size = target
            else:
                # Try once more with more time
                if remaining() > 20:
                    budget2 = min(remaining() - 8, 60.0)
                    found = sa_loop(target, budget2)
                    if found is not None:
                        best = found
                        best_size = target
                break

        # Final validation
        if not is_valid_set(best):
            best = greedy_build()

        return best

    except Exception:
        try:
            return greedy_build()
        except Exception:
            return [1, 2, 5, 11]
