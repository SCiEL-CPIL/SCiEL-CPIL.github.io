"""Rewritten program: keep the construct() entry point and return three sorted subsets."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    _t_start = time.time()
    _deadline = 120.0  # hard kill at 150s, keep safety margin

    def _time_left():
        return _deadline - (time.time() - _t_start)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        """Check if subset is sum-free (no a+b=c with a,b,c in subset, a=b allowed)."""
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(values, seed_offset=0, time_budget=35.0):
        """
        Stronger solver:
        - Precompute sum-triple conflicts as a 3-uniform hypergraph.
        - Greedy construction with multiple diverse orderings.
        - Local search with add-back, remove-and-retry, and swap moves.
        - Simulated annealing polish for the best candidates.
        """
        n = len(values)
        if n == 0:
            return []

        # Precompute conflict structure:
        # For each element c, store pairs (a,b) in the full set with a+b=c.
        # For each element x, store which other elements y have x+y in the full set.
        # This helps fast conflict checks during search.

        val_set = set(values)
        # conflict_pairs[c] = list of (a,b) such that a+b=c, a,b in values
        conflict_pairs = {c: [] for c in values}
        for a in values:
            for b in values:
                s = a + b
                if s in val_set:
                    conflict_pairs[s].append((a, b))

        # For fast "can I add v?" checks:
        # Adding v to current set S creates a conflict if:
        #   - exists a in S: v+a in S  (v + a = c)
        #   - exists a in S: v-a in S  (a + (v-a) = v, i.e., a + b = v)
        #   - 2*v in S
        # We can precompute for each v the set of "forbidden partners":
        #   partners_plus[v] = {a : v+a in val_set}  -> if a in S and v in S, conflict
        #   partners_minus[v] = {a : v-a in val_set}  -> if a in S and v in S, conflict
        #   double_conflict[v] = 2*v in val_set

        partners_plus = {}
        partners_minus = {}
        double_conflict = set()
        for v in values:
            pp = set()
            pm = set()
            for a in values:
                if v + a in val_set:
                    pp.add(a)
                if v - a in val_set:
                    pm.add(a)
            partners_plus[v] = pp
            partners_minus[v] = pm
            if 2 * v in val_set:
                double_conflict.add(v)

        def check_conflict_add(S, v):
            """Check if adding v to set S creates a sum-free violation."""
            # v + a = c for a,c in S
            for a in S:
                if v + a in S:
                    return True
            # a + b = v for a,b in S
            for a in S:
                if v - a in S:
                    return True
            # 2*v = c for c in S
            if 2 * v in S:
                return True
            return False

        def greedy_build(order, S_init=None):
            """Greedy construction following a given order."""
            S = set(S_init) if S_init else set()
            for v in order:
                if v not in S and not check_conflict_add(S, v):
                    S.add(v)
            return S

        def add_back_pass(S):
            """Try to add back all removable elements that don't conflict."""
            changed = True
            while changed:
                changed = False
                for v in values:
                    if v not in S and not check_conflict_add(S, v):
                        S.add(v)
                        changed = True
            return S

        def remove_and_retry(S, max_rounds=5):
            """Remove one element, then try to add back multiple others."""
            S = set(S)
            improved = False
            for _ in range(max_rounds):
                if _time_left() < 5:
                    break
                best_gain = 0
                best_remove = None
                best_trial = None
                for v in list(S):
                    trial = set(S)
                    trial.discard(v)
                    added = 0
                    for rv in values:
                        if rv not in trial and not check_conflict_add(trial, rv):
                            trial.add(rv)
                            added += 1
                    gain = added - 1  # removed 1, added 'added'
                    if gain > best_gain:
                        best_gain = gain
                        best_remove = v
                        best_trial = set(trial)
                if best_gain > 0 and best_trial is not None:
                    S = best_trial
                    improved = True
                else:
                    break
            return S

        def swap_search(S, max_rounds=10):
            """Try swap moves: remove one, add another."""
            S = set(S)
            for _ in range(max_rounds):
                if _time_left() < 5:
                    break
                improved = False
                for v_remove in list(S):
                    trial = set(S)
                    trial.discard(v_remove)
                    for v_add in values:
                        if v_add not in trial and not check_conflict_add(trial, v_add):
                            new_trial = set(trial)
                            new_trial.add(v_add)
                            if len(new_trial) > len(S):
                                S = new_trial
                                improved = True
                                break
                    if improved:
                        break
                if not improved:
                    break
            return S

        def simulated_annealing(S, steps=2000, t_start=1.0, t_end=0.01):
            """Simulated annealing with add/remove/swap moves."""
            S = set(S)
            current_energy = -len(S)  # minimize negative size = maximize size
            best_S = set(S)
            best_energy = current_energy
            rng = random.Random(hash(tuple(sorted(S))) & 0xFFFFFFFF)

            for step in range(steps):
                if _time_left() < 5:
                    break
                temp = t_start * (t_end / t_start) ** (step / max(steps, 1))

                # Choose a move
                move_type = rng.random()
                if move_type < 0.3:
                    # Add a random element
                    candidates = [v for v in values if v not in S]
                    if not candidates:
                        continue
                    v = rng.choice(candidates)
                    trial = set(S)
                    trial.add(v)
                    # Check if valid
                    valid = True
                    for a in trial:
                        if v + a in trial or (v - a) in trial:
                            valid = False
                            break
                    if 2 * v in trial:
                        valid = False
                    if not valid:
                        continue
                    delta_e = -1  # size increased by 1
                elif move_type < 0.6:
                    # Remove a random element
                    if not S:
                        continue
                    v = rng.choice(list(S))
                    trial = set(S)
                    trial.discard(v)
                    delta_e = 1  # size decreased by 1
                else:
                    # Swap: remove one, add another
                    if not S:
                        continue
                    v_remove = rng.choice(list(S))
                    candidates = [v for v in values if v not in S]
                    if not candidates:
                        continue
                    v_add = rng.choice(candidates)
                    trial = set(S)
                    trial.discard(v_remove)
                    trial.add(v_add)
                    # Check validity
                    valid = True
                    for a in trial:
                        if v_add + a in trial or (v_add - a) in trial:
                            valid = False
                            break
                    if 2 * v_add in trial:
                        valid = False
                    if not valid:
                        continue
                    delta_e = 0  # same size

                new_energy = current_energy + delta_e
                if new_energy < current_energy or rng.random() < __import__('math').exp(-(new_energy - current_energy) / max(temp, 1e-10)):
                    S = trial
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_S = set(S)

            return best_S

        # Main search loop
        best_subset = set()
        best_size = 0

        # Generate diverse orderings
        orderings = []
        # 1. By magnitude descending
        orderings.append(sorted(values, key=lambda x: abs(x), reverse=True))
        # 2. By value descending
        orderings.append(sorted(values, reverse=True))
        # 3. By value ascending
        orderings.append(sorted(values))
        # 4. Random shuffles
        rng_init = random.Random(seed_offset)
        for _ in range(8):
            shuffled = values[:]
            rng_init.shuffle(shuffled)
            orderings.append(shuffled)

        # Time budget per instance
        t_instance_start = time.time()
        time_per_ordering = time_budget / len(orderings)

        for i, order in enumerate(orderings):
            if _time_left() < 10 or (time.time() - t_instance_start) > time_budget:
                break

            # Greedy build
            S = greedy_build(order)
            # Add-back pass
            S = add_back_pass(S)
            # Remove-and-retry
            S = remove_and_retry(S)
            # Swap search
            S = swap_search(S)

            if len(S) > best_size:
                best_size = len(S)
                best_subset = set(S)

            # Simulated annealing polish on the best candidate
            if i >= 0 and (time.time() - t_instance_start) < time_budget * 0.7:
                S_sa = simulated_annealing(S, steps=1500)
                if len(S_sa) > best_size:
                    best_size = len(S_sa)
                    best_subset = set(S_sa)

        # Final add-back on the best solution
        best_subset = add_back_pass(best_subset)

        return sorted(best_subset)

    # Solve each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if _time_left() < 15:
            # Fallback: use parity (odd numbers are sum-free)
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue

        vals = instance(size, seed)
        # Allocate more time to larger instances
        if size <= 60:
            tb = 25.0
        elif size <= 120:
            tb = 35.0
        else:
            tb = 50.0
        result = solve_instance(vals, seed_offset=seed, time_budget=tb)

        # Verify sum-free
        if not is_sum_free(result):
            # Fallback to parity
            result = [x for x in vals if x % 2 != 0]

        results.append(result)

    return results
# EVOLVE-BLOCK-END
