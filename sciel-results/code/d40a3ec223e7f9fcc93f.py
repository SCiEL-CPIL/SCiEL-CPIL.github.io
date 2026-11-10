import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85

    N = 256

    # Precompute all 4-APs in [1, N]
    all_4aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            all_4aps.append((a, d))

    # For each element x, precompute which 4-APs it participates in
    elem_4aps = {x: [] for x in range(1, N + 1)}
    for (a, d) in all_4aps:
        terms = [a, a + d, a + 2*d, a + 3*d]
        for pos, t in enumerate(terms):
            elem_4aps[t].append((a, d, pos))

    # Precompute all 3-APs
    all_3aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            all_3aps.append((a, d))

    # For each element x, precompute which 3-APs it participates in
    elem_3aps = {x: [] for x in range(1, N + 1)}
    for (a, d) in all_3aps:
        terms = [a, a + d, a + 2*d]
        for pos, t in enumerate(terms):
            elem_3aps[t].append((a, d, pos))

    def count_3aps(S_set):
        count = 0
        for (a, d) in all_3aps:
            if a in S_set and a + d in S_set and a + 2*d in S_set:
                count += 1
        return count

    def has_4ap(S_set):
        for (a, d) in all_4aps:
            if a in S_set and a + d in S_set and a + 2*d in S_set and a + 3*d in S_set:
                return True
        return False

    def get_4aps_created_by_add(x, S_set):
        for (a, d, pos) in elem_4aps[x]:
            terms = [a, a + d, a + 2*d, a + 3*d]
            others = [terms[i] for i in range(4) if i != pos]
            if all(t in S_set for t in others):
                return True
        return False

    def get_3aps_gained_by_add(x, S_set):
        gained = 0
        for (a, d, pos) in elem_3aps[x]:
            terms = [a, a + d, a + 2*d]
            others = [terms[i] for i in range(3) if i != pos]
            if all(t in S_set for t in others):
                gained += 1
        return gained

    def get_3aps_lost_by_remove(x, S_set):
        lost = 0
        for (a, d, pos) in elem_3aps[x]:
            terms = [a, a + d, a + 2*d]
            others = [terms[i] for i in range(3) if i != pos]
            if all(t in S_set for t in others):
                lost += 1
        return lost

    def greedy_density_init():
        """Greedy that picks elements with highest marginal 3-AP gain."""
        S = set()
        remaining = list(range(1, N + 1))
        while remaining:
            if time.time() > DEADLINE - 10:
                break
            best_x = None
            best_gain = -1
            for x in remaining:
                if get_4aps_created_by_add(x, S):
                    continue
                gain = get_3aps_gained_by_add(x, S)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            S.add(best_x)
            remaining.remove(best_x)
        return S

    def greedy_random_init():
        """Random-order greedy."""
        S = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if not get_4aps_created_by_add(x, S):
                S.add(x)
        return S

    def simulated_annealing(S_init, max_steps=50000, t_start=5.0, t_end=0.1):
        """Simulated annealing with add/remove/swap moves."""
        S_set = set(S_init)
        current_score = count_3aps(S_set)
        best_S = set(S_set)
        best_score = current_score

        in_S = [False] * (N + 1)
        for x in S_set:
            in_S[x] = True

        total_steps = max_steps
        for step in range(total_steps):
            if time.time() > DEADLINE:
                break

            T = t_start * (t_end / t_start) ** (step / max(1, total_steps - 1))

            # Choose move type: add, remove, or swap
            move_type = random.random()
            if move_type < 0.4:
                # Add a random element not in S
                not_in = [x for x in range(1, N + 1) if not in_S[x]]
                if not not_in:
                    continue
                x = random.choice(not_in)
                if get_4aps_created_by_add(x, S_set):
                    continue
                delta = get_3aps_gained_by_add(x, S_set)
                S_set.add(x)
                in_S[x] = True
                new_score = current_score + delta
                if new_score > best_score:
                    best_score = new_score
                    best_S = set(S_set)
                if delta < 0 and random.random() < (2.718281828 ** (delta / max(T, 1e-9))):
                    pass  # accept
                else:
                    if delta < 0:
                        S_set.discard(x)
                        in_S[x] = False
                        continue
                current_score = new_score
            elif move_type < 0.7:
                # Remove a random element from S
                if not S_set:
                    continue
                x = random.choice(list(S_set))
                delta = -get_3aps_lost_by_remove(x, S_set)
                S_set.discard(x)
                in_S[x] = False
                new_score = current_score + delta
                if new_score > best_score:
                    best_score = new_score
                    best_S = set(S_set)
                if delta < 0 and random.random() < (2.718281828 ** (delta / max(T, 1e-9))):
                    pass
                else:
                    if delta < 0:
                        S_set.add(x)
                        in_S[x] = True
                        continue
                current_score = new_score
            else:
                # Swap: remove one, add another
                if not S_set:
                    continue
                x_rm = random.choice(list(S_set))
                not_in = [x for x in range(1, N + 1) if not in_S[x]]
                if not not_in:
                    continue
                x_add = random.choice(not_in)

                # Check if adding x_add is valid after removing x_rm
                temp_S = S_set - {x_rm}
                if get_4aps_created_by_add(x_add, temp_S):
                    continue

                # Calculate delta: gain from add - loss from remove
                gain = get_3aps_gained_by_add(x_add, temp_S)
                loss = get_3aps_lost_by_remove(x_rm, S_set)
                # But we need to be careful: some 3-APs might involve both x_rm and x_add
                # Simple approximation: compute score change directly
                new_S = temp_S | {x_add}
                new_score = count_3aps(new_S)
                delta = new_score - current_score

                S_set = new_S
                in_S[x_rm] = False
                in_S[x_add] = True
                if new_score > best_score:
                    best_score = new_score
                    best_S = set(S_set)
                if delta < 0 and random.random() < (2.718281828 ** (delta / max(T, 1e-9))):
                    pass
                else:
                    if delta < 0:
                        S_set = S_set - {x_add} | {x_rm}
                        in_S[x_add] = False
                        in_S[x_rm] = True
                        continue
                current_score = new_score

        return best_S, best_score

    best_overall = set()
    best_overall_score = 0

    # Phase 1: Density-driven greedy initialization
    S0 = greedy_density_init()
    if S0:
        best_S, best_score = simulated_annealing(S0, max_steps=40000, t_start=5.0, t_end=0.1)
        best_overall = best_S
        best_overall_score = best_score

    # Phase 2: Random greedy + SA
    num_restarts = 12
    for i in range(num_restarts):
        if time.time() > DEADLINE - 5:
            break
        S_i = greedy_random_init()
        if not S_i:
            continue
        steps = 30000 if i < 4 else 20000
        best_S, best_score = simulated_annealing(S_i, max_steps=steps, t_start=4.0, t_end=0.1)
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_S

    # Phase 3: Cantor-like set + SA
    S_cantor = set()
    for x in range(1, N + 1):
        t = x
        valid = True
        while t > 0:
            if t % 4 >= 2:
                valid = False
                break
            t //= 4
        if valid:
            S_cantor.add(x)

    if S_cantor and not has_4ap(S_cantor):
        best_S, best_score = simulated_annealing(S_cantor, max_steps=30000, t_start=4.0, t_end=0.1)
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_S

    # Phase 4: More random restarts if time permits
    while time.time() < DEADLINE - 8:
        S_rand = greedy_random_init()
        if not S_rand:
            continue
        best_S, best_score = simulated_annealing(S_rand, max_steps=15000, t_start=3.0, t_end=0.1)
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_S

    return sorted(best_overall)
