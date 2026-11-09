import time
import random
import math


def construct():
    start = time.time()
    DEADLINE = start + 115.0

    prescribed = [1, 2, 5, 11]
    pres_set = set(prescribed)
    N = 284
    random.seed(777)

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
        """Try greedy from high to low."""
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

    try:
        g1 = greedy_build()
        g2 = greedy_build_reversed()
        best = g1 if len(g1) >= len(g2) else g2
        best_size = len(best)

        # ---------- Incremental state ----------
        class State:
            __slots__ = ('arr', 'n', 'counts', 'conflict', 'val_counts', 'fixed', 'available')

            def __init__(self, arr, fixed):
                self.arr = arr
                self.n = len(arr)
                self.fixed = fixed
                self.counts = {}
                self.conflict = 0
                self.val_counts = {}
                self.available = []
                used = set(arr)
                for x in range(1, N + 1):
                    if x not in used:
                        self.available.append(x)
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

            def random_available(self, exclude_idx):
                """Pick a random available value not equal to arr[exclude_idx]."""
                n = self.n
                arr = self.arr
                # Try a few random picks from available
                for _ in range(8):
                    if not self.available:
                        return None
                    idx = random.randrange(len(self.available))
                    val = self.available[idx]
                    if val != arr[exclude_idx]:
                        return val
                return None

        def swap_propose(self, i, j, new_val):
                """Swap arr[i] and arr[j], then set arr[i] to new_val. Returns conflict or None."""
                if i == j:
                    return None
                old_i = self.arr[i]
                old_j = self.arr[j]
                if self.val_counts.get(new_val, 0) >= 1:
                    return None
                # Remove old_i and old_j sums
                for k in range(self.n):
                    self._dec_sum(old_i + self.arr[k])
                    self._dec_sum(old_j + self.arr[k])
                self.val_counts[old_i] -= 1
                if self.val_counts[old_i] == 0:
                    del self.val_counts[old_i]
                self.val_counts[old_j] -= 1
                if self.val_counts[old_j] == 0:
                    del self.val_counts[old_j]
                # Set arr[i]=new_val, arr[j]=old_i
                self.arr[i] = new_val
                self.arr[j] = old_i
                for k in range(self.n):
                    self._inc_sum(new_val + self.arr[k])
                    self._inc_sum(old_i + self.arr[k])
                self.val_counts[new_val] = self.val_counts.get(new_val, 0) + 1
                self.val_counts[old_i] = self.val_counts.get(old_i, 0) + 1
                return self.conflict

        def swap_revert(self, i, j, new_val):
                """Revert a swap_propose."""
                old_i = self.arr[j]  # old_i is now at j
                old_j = self.arr[i]  # new_val is at i
                # Remove new_val and old_i sums
                for k in range(self.n):
                    self._dec_sum(new_val + self.arr[k])
                    self._dec_sum(old_i + self.arr[k])
                self.val_counts[new_val] -= 1
                if self.val_counts[new_val] == 0:
                    del self.val_counts[new_val]
                self.val_counts[old_i] -= 1
                if self.val_counts[old_i] == 0:
                    del self.val_counts[old_i]
                # Restore arr[i]=old_i, arr[j]=old_j
                self.arr[i] = old_i
                self.arr[j] = old_j
                for k in range(self.n):
                    self._inc_sum(old_i + self.arr[k])
                    self._inc_sum(old_j + self.arr[k])
                self.val_counts[old_i] = self.val_counts.get(old_i, 0) + 1
                self.val_counts[old_j] = self.val_counts.get(old_j, 0) + 1

        def make_init(target, diversity=0.0):
            """Create initial array of target size. diversity in [0,1] controls how random."""
            base = list(best)
            used = set(base)
            need = target - len(base)
            if need <= 0:
                return base
            pool = [x for x in range(1, N + 1) if x not in used]
            random.shuffle(pool)
            if diversity < 0.3:
                # Keep most of best, add random
                return base + pool[:need]
            elif diversity < 0.6:
                # Mix: keep prescribed + some from best + random
                keep_count = max(4, int(len(base) * (1 - diversity)))
                keep = list(base[:keep_count])
                used2 = set(keep)
                need2 = target - len(keep)
                pool2 = [x for x in range(1, N + 1) if x not in used2]
                random.shuffle(pool2)
                return keep + pool2[:need2]
            else:
                # More random: keep only prescribed + random
                keep = list(prescribed)
                used2 = set(keep)
                need2 = target - len(keep)
                pool2 = [x for x in range(1, N + 1) if x not in used2]
                random.shuffle(pool2)
                return keep + pool2[:need2]

        def run_sa(target, init, budget, T_start=None, T_end=None, cool=None):
            end = min(time.time() + budget, DEADLINE - 2.0)
            fixed = [v in pres_set for v in init]
            st = State(init, fixed)
            best_obj = st.conflict
            best_arr = list(init)
            
            if T_start is None:
                T_start = max(0.5, best_obj * 0.3)
            if T_end is None:
                T_end = 0.005
            if cool is None:
                cool = 0.9997
            
            T = T_start
            n = target
            no_improve_count = 0
            improve_threshold = 500  # steps without improvement before early exit
            step_count = 0
            
            while time.time() < end and T > T_end:
                i = random.randrange(n)
                if fixed[i]:
                    continue
                old = st.arr[i]
                cur = st.conflict
                
                # Try to find a valid new value
                new = st.random_available(i)
                if new is None:
                    # Fallback to random
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
                    # Accept with probability based on temperature
                    if random.random() < math.exp((cur - nc) / T):
                        no_improve_count += 1
                    else:
                        st.revert(i, old, new)
                        no_improve_count += 1
                
                T *= cool
                
                # Early exit if stuck for too long
                if no_improve_count > improve_threshold and best_obj > 0:
                    break
            
            if best_obj == 0:
                return best_arr
            
            # Local descent polish with more aggressive search
            if best_obj <= 15:
                st2 = State(best_arr, fixed)
                polish_end = min(end, time.time() + min(5.0, (DEADLINE - 2.0) - time.time()))
                while time.time() < polish_end and st2.conflict > 0:
                    i = random.randrange(n)
                    if fixed[i]:
                        continue
                    old = st2.arr[i]
                    cur = st2.conflict
                    improved = False
                    # Try multiple candidates for this position
                    for _ in range(12):
                        new = st2.random_available(i)
                        if new is None:
                            new = random.randint(1, N)
                        nc = st2.propose(i, new)
                        if nc is None:
                            continue
                        if nc < cur:
                            cur = nc
                            improved = True
                            break
                        else:
                            st2.revert(i, old, new)
                    if not improved:
                        # Try a few more random positions
                        for _ in range(4):
                            i2 = random.randrange(n)
                            if fixed[i2]:
                                continue
                            old2 = st2.arr[i2]
                            cur2 = st2.conflict
                            new2 = st2.random_available(i2)
                            if new2 is None:
                                new2 = random.randint(1, N)
                            nc2 = st2.propose(i2, new2)
                            if nc2 is None:
                                continue
                            if nc2 < cur2:
                                improved = True
                                break
                            else:
                                st2.revert(i2, old2, new2)
                        if not improved:
                            break
                    if st2.conflict == 0:
                        return list(st2.arr)
            
            return None

        def sa_loop(target, budget):
            t0 = time.time()
            attempts = 0
            while time.time() - t0 < budget and remaining() > 4:
                sub = min(6.0, budget - (time.time() - t0), remaining() - 4)
                if sub <= 0.3:
                    break
                diversity = random.random()
                init = make_init(target, diversity)
                # Vary temperature parameters
                T_start = random.uniform(0.2, 1.5)
                T_end = random.uniform(0.003, 0.015)
                cool = random.uniform(0.9995, 0.9999)
                res = run_sa(target, init, sub, T_start, T_end, cool)
                if res is not None and is_valid_set(res):
                    return res
                attempts += 1
            return None

        # ---------- Incremental growth to maximize size ----------
        for target in range(best_size + 1, 21):
            if remaining() < 10:
                break
            # First try with moderate budget
            budget = min(12.0, remaining() - 5)
            found = sa_loop(target, budget)
            if found is not None:
                best = found
                best_size = target
            else:
                # If not found, try with larger budget if time permits
                if remaining() > 20:
                    budget2 = min(remaining() - 10, 60.0)
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
