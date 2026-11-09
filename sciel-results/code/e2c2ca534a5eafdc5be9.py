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

    # ---------- Greedy baselines ----------
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
        best = max([g1, g2, g3, g4, g5], key=len)
        best_size = len(best)

        # ---------- Incremental state with simulated annealing ----------
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
                if not self.available:
                    return None
                for _ in range(8):
                    idx = random.randrange(len(self.available))
                    val = self.available[idx]
                    if val != self.arr[exclude_idx]:
                        return val
                return None

        def simulated_annealing(initial_arr, target_size, time_budget):
            """
            Simulated annealing to find a Sidon set of target_size.
            Starts from initial_arr and tries to add/remove/swap elements.
            """
            end_time = time.time() + time_budget
            rng = random.Random()

            # Start with initial arrangement
            arr = list(initial_arr)
            # If we have fewer than target_size, pad with random available values
            used = set(arr)
            for x in range(1, N + 1):
                if x not in used and len(arr) < target_size:
                    arr.append(x)
                    used.add(x)
                if len(arr) >= target_size:
                    break

            # Ensure we have exactly target_size elements
            if len(arr) > target_size:
                arr = arr[:target_size]
                used = set(arr)
            elif len(arr) < target_size:
                # Add more random values
                for x in range(1, N + 1):
                    if x not in used:
                        arr.append(x)
                        used.add(x)
                    if len(arr) >= target_size:
                        break

            state = State(arr, pres_set)
            current_conflict = state.conflict
            best_conflict = current_conflict
            best_arr = list(arr)

            # Simulated annealing parameters
            T = 10.0
            T_min = 0.01
            cooling_rate = 0.9995
            max_steps = 500000

            step = 0
            while time.time() < end_time and step < max_steps:
                step += 1

                # Choose move type: swap (80%), insert (10%), delete (10%)
                move_type = rng.random()
                i = rng.randrange(state.n)

                if move_type < 0.8:
                    # Swap move: replace arr[i] with a random available value
                    new_val = state.random_available(i)
                    if new_val is None:
                        continue
                    old_conflict = state.conflict
                    new_conflict = state.propose(i, new_val)
                    if new_conflict is None:
                        continue
                    delta = new_conflict - old_conflict
                    if delta < 0 or rng.random() < math.exp(-delta / T):
                        # Accept
                        pass
                    else:
                        # Revert
                        state.revert(i, state.arr[i], new_val)
                        # Note: state.arr[i] was changed, so we need old_val
                        # Actually, propose changed arr[i], so we need to revert properly
                        # Let's fix this: store old_val before propose
                        # Actually, the revert function takes (i, old_val, new_val)
                        # But we already changed arr[i] in propose. Let me re-examine.
                        # The propose method changes arr[i] and returns new conflict.
                        # The revert method expects (i, old_val, new_val) where new_val is current arr[i].
                        # So we need to pass the original old_val.
                        # Let me fix the logic.
                        # Actually, I'll restructure this.

                # Let me redo this more carefully
                break  # Temporary break, will rewrite

            return None

        # Let me rewrite the simulated annealing more carefully

        def simulated_annealing_v2(initial_arr, target_size, time_budget):
            """
            Simulated annealing to find a Sidon set of target_size.
            """
            end_time = time.time() + time_budget
            rng = random.Random(42)

            # Start with initial arrangement, pad if needed
            arr = list(initial_arr)
            used = set(arr)
            while len(arr) < target_size:
                x = rng.randint(1, N)
                if x not in used:
                    arr.append(x)
                    used.add(x)
            if len(arr) > target_size:
                arr = arr[:target_size]
                used = set(arr)

            state = State(arr, pres_set)
            current_conflict = state.conflict
            best_conflict = current_conflict
            best_arr = list(arr)
            best_valid = (current_conflict == 0)

            # Simulated annealing parameters
            T = 5.0
            T_min = 0.005
            cooling_rate = 0.9999
            max_steps = 1000000

            step = 0
            no_improve = 0
            while time.time() < end_time and step < max_steps:
                step += 1

                # Choose move type
                move_type = rng.random()
                i = rng.randrange(state.n)

                if move_type < 0.7:
                    # Swap: replace arr[i] with random available
                    new_val = state.random_available(i)
                    if new_val is None:
                        continue
                    old_val = state.arr[i]
                    old_conflict = state.conflict
                    new_conflict = state.propose(i, new_val)
                    if new_conflict is None:
                        continue
                    delta = new_conflict - old_conflict
                    if delta < 0 or rng.random() < math.exp(-delta / T):
                        # Accept
                        if new_conflict < best_conflict:
                            best_conflict = new_conflict
                            best_arr = list(state.arr)
                            best_valid = (new_conflict == 0)
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        # Revert
                        state.revert(i, old_val, new_val)

                elif move_type < 0.85:
                    # Insert: replace a random element with a new random value
                    new_val = state.random_available(i)
                    if new_val is None:
                        continue
                    old_val = state.arr[i]
                    old_conflict = state.conflict
                    new_conflict = state.propose(i, new_val)
                    if new_conflict is None:
                        continue
                    delta = new_conflict - old_conflict
                    if delta < 0 or rng.random() < math.exp(-delta / T):
                        if new_conflict < best_conflict:
                            best_conflict = new_conflict
                            best_arr = list(state.arr)
                            best_valid = (new_conflict == 0)
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        state.revert(i, old_val, new_val)

                else:
                    # Swap two elements
                    j = rng.randrange(state.n)
                    if i == j:
                        continue
                    old_i = state.arr[i]
                    old_j = state.arr[j]
                    # Swap them
                    state.arr[i], state.arr[j] = old_j, old_i
                    # Update conflicts
                    for k in range(state.n):
                        state._dec_sum(old_i + state.arr[k])
                        state._dec_sum(old_j + state.arr[k])
                    for k in range(state.n):
                        state._inc_sum(state.arr[i] + state.arr[k])
                        state._inc_sum(state.arr[j] + state.arr[k])
                    new_conflict = state.conflict
                    delta = new_conflict - old_conflict
                    if delta < 0 or rng.random() < math.exp(-delta / T):
                        if new_conflict < best_conflict:
                            best_conflict = new_conflict
                            best_arr = list(state.arr)
                            best_valid = (new_conflict == 0)
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        # Revert swap
                        state.arr[i], state.arr[j] = old_i, old_j
                        for k in range(state.n):
                            state._dec_sum(state.arr[i] + state.arr[k])
                            state._dec_sum(state.arr[j] + state.arr[k])
                        for k in range(state.n):
                            state._inc_sum(old_i + state.arr[k])
                            state._inc_sum(old_j + state.arr[k])

                # Cool down
                T = max(T_min, T * cooling_rate)

                # Early termination if we found a valid solution
                if best_valid and best_conflict == 0:
                    break

                # Periodic restart if stuck
                if no_improve > 50000:
                    # Try a fresh random restart
                    no_improve = 0
                    T = max(T, 2.0)

            if best_valid:
                return best_arr
            return None

        # ---------- Incremental growth using simulated annealing ----------
        for target in range(best_size + 1, 21):
            if remaining() < 15:
                break

            # Allocate time for simulated annealing
            time_budget = min(remaining() - 10, 30.0)
            if remaining() < 40:
                time_budget = min(remaining() - 10, 20.0)

            found = simulated_annealing_v2(best, target, time_budget)
            if found is not None and is_valid_set(found):
                best = found
                best_size = target
            else:
                # Try with more time if available
                if remaining() > 50:
                    time_budget2 = min(remaining() - 15, 60.0)
                    found = simulated_annealing_v2(best, target, time_budget2)
                    if found is not None and is_valid_set(found):
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
