import time
import random
import math
import sys


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
                # Save old arr values before modification
                old_arr = list(self.arr)
                for j in range(self.n):
                    self._dec_sum(old + old_arr[j])
                self.val_counts[old] -= 1
                if self.val_counts[old] == 0:
                    del self.val_counts[old]
                self.arr[i] = new_val
                for j in range(self.n):
                    self._inc_sum(new_val + old_arr[j])
                self.val_counts[new_val] = self.val_counts.get(new_val, 0) + 1
                return self.conflict

            def revert(self, i, old_val, new_val):
                # Save current arr values
                cur_arr = list(self.arr)
                for j in range(self.n):
                    self._dec_sum(new_val + cur_arr[j])
                self.val_counts[new_val] -= 1
                if self.val_counts[new_val] == 0:
                    del self.val_counts[new_val]
                self.arr[i] = old_val
                for j in range(self.n):
                    self._inc_sum(old_val + cur_arr[j])
                self.val_counts[old_val] = self.val_counts.get(old_val, 0) + 1

            def best_available(self, exclude_idx, num_trials=80):
                """Pick an available value that minimizes immediate conflict increase."""
                n = self.n
                arr = self.arr
                old = arr[exclude_idx]
                if not self.available:
                    return None
                best_val = None
                best_delta = float('inf')
                # Also try values near existing elements for better heuristic
                candidates = set()
                for _ in range(min(num_trials, len(self.available))):
                    idx = random.randrange(len(self.available))
                    candidates.add(self.available[idx])
                # Add neighbors of existing elements
                for a in arr:
                    for offset in range(-3, 4):
                        v = a + offset
                        if 1 <= v <= N and v != old:
                            candidates.add(v)
                for val in candidates:
                    if val == old:
                        continue
                    # Compute delta in conflict count if we replace old with val
                    delta = 0
                    for j in range(n):
                        s_new = val + arr[j]
                        s_old = old + arr[j]
                        c_old = self.counts.get(s_old, 0)
                        c_new = self.counts.get(s_new, 0)
                        if c_old == 2:
                            delta -= 1
                        if c_new == 1:
                            delta += 1
                    if delta < best_delta:
                        best_delta = delta
                        best_val = val
                        if delta < 0:
                            break
                if best_val is None:
                    idx = random.randrange(len(self.available))
                    best_val = self.available[idx]
                    if best_val == old:
                        return None
                return best_val

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
            improve_threshold = 300
            step_count = 0
            
            while time.time() < end and T > T_end:
                move_type = random.random()
                if move_type < 0.75:
                    # 1-opt: replace one non-fixed element
                    i = random.randrange(n)
                    if fixed[i]:
                        continue
                    old = st.arr[i]
                    cur = st.conflict
                    
                    new = st.best_available(i, num_trials=60)
                    if new is None:
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
                    # 2-opt: swap two non-fixed elements - FIXED LOGIC
                    i = random.randrange(n)
                    j = random.randrange(n)
                    if fixed[i] or fixed[j] or i == j:
                        continue
                    old_i = st.arr[i]
                    old_j = st.arr[j]
                    cur = st.conflict
                    
                    # Save state before modification
                    old_arr = list(st.arr)
                    old_val_counts = dict(st.val_counts)
                    old_counts = dict(st.counts)
                    old_conflict = st.conflict
                    
                    # Perform swap
                    st.arr[i] = old_j
                    st.arr[j] = old_i
                    # Update val_counts
                    st.val_counts[old_i] -= 1
                    st.val_counts[old_j] -= 1
                    if st.val_counts[old_i] == 0:
                        del st.val_counts[old_i]
                    if st.val_counts[old_j] == 0:
                        del st.val_counts[old_j]
                    st.val_counts[old_j] = st.val_counts.get(old_j, 0) + 1
                    st.val_counts[old_i] = st.val_counts.get(old_i, 0) + 1
                    # Update sums using OLD arr values
                    for k in range(n):
                        st._dec_sum(old_i + old_arr[k])
                        st._dec_sum(old_j + old_arr[k])
                        st._inc_sum(st.arr[i] + old_arr[k])
                        st._inc_sum(st.arr[j] + old_arr[k])
                    nc = st.conflict
                    
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
                            # Revert swap
                            st.arr[i] = old_i
                            st.arr[j] = old_j
                            st.val_counts[old_i] -= 1
                            st.val_counts[old_j] -= 1
                            if st.val_counts[old_i] == 0:
                                del st.val_counts[old_i]
                            if st.val_counts[old_j] == 0:
                                del st.val_counts[old_j]
                            st.val_counts[old_i] = st.val_counts.get(old_i, 0) + 1
                            st.val_counts[old_j] = st.val_counts.get(old_j, 0) + 1
                            for k in range(n):
                                st._dec_sum(st.arr[i] + old_arr[k])
                                st._dec_sum(st.arr[j] + old_arr[k])
                                st._inc_sum(old_i + old_arr[k])
                                st._inc_sum(old_j + old_arr[k])
                            no_improve_count += 1
                
                T *= cool
                
                if no_improve_count > improve_threshold and best_obj > 0:
                    break
            
            if best_obj == 0:
                return best_arr
            
            # Local descent polish
            if best_obj <= 15:
                st2 = State(best_arr, fixed)
                polish_end = min(end, time.time() + min(5.0, (DEADLINE - 2.0) - time.time()))
                while time.time() < polish_end and st2.conflict > 0:
                    improved = False
                    for _ in range(15):
                        i = random.randrange(n)
                        if fixed[i]:
                            continue
                        old = st2.arr[i]
                        cur = st2.conflict
                        for _ in range(20):
                            new = st2.best_available(i, num_trials=40)
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
                        if improved:
                            break
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
            if remaining() < 15:
                break
            budget = min(25.0, remaining() - 8)
            found = sa_loop(target, budget)
            if found is not None:
                best = found
                best_size = target
            else:
                if remaining() > 25:
                    budget2 = min(remaining() - 12, 80.0)
                    found = sa_loop(target, budget2)
                    if found is not None:
                        best = found
                        best_size = target
                break

        # Final greedy growth pass: try to add more elements greedily
        if remaining() > 5:
            arr = list(best)
            used = set(arr)
            sums = set()
            for i in range(len(arr)):
                for j in range(i, len(arr)):
                    sums.add(arr[i] + arr[j])
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
            if len(arr) > len(best):
                best = arr

        # Final validation
        if not is_valid_set(best):
            best = greedy_build()

        return best

    except Exception:
        try:
            return greedy_build()
        except Exception:
            return [1, 2, 5, 11]
