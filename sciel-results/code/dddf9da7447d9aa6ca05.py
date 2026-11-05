"""Minimum maximal no-isosceles set with simulated annealing."""
import random
import time
import math

N = 16
DEADLINE_FRACTION = 0.90  # safety margin


def _dist_sq(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _creates_isosceles(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle."""
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        d_pa = _dist_sq(p, a)
        for j in range(i + 1, n):
            b = chosen[j]
            d_pb = _dist_sq(p, b)
            d_ab = _dist_sq(a, b)
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False


def _is_valid_set(chosen):
    """Check that chosen contains no isosceles triangles."""
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        for j in range(i + 1, n):
            b = chosen[j]
            d_ab = _dist_sq(a, b)
            for k in range(j + 1, n):
                c = chosen[k]
                d_ac = _dist_sq(a, c)
                d_bc = _dist_sq(b, c)
                if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                    return False
    return True


def _is_maximal(chosen):
    """Check that no grid point can be added without creating an isosceles triangle."""
    chosen_set = set(chosen)
    for x in range(N):
        for y in range(N):
            p = (x, y)
            if p in chosen_set:
                continue
            if not _creates_isosceles(p, chosen):
                return False
    return True


def _greedy_construct(order=None):
    """Greedy construction: add points in given order if valid."""
    if order is None:
        order = [(x, y) for x in range(N) for y in range(N)]
    chosen = []
    for p in order:
        if not _creates_isosceles(p, chosen):
            chosen.append(p)
    return chosen


def _make_random_order():
    order = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(order)
    return order


def _simulated_annealing(initial_set, start_temp, end_temp, steps, deadline):
    """
    Simulated annealing on the set of chosen points.
    Moves: add a random non-chosen point, remove a random chosen point,
    or swap (remove one, add another).
    Objective: minimize |chosen|, with hard constraint that the set is
    no-isosceles and maximal.
    """
    chosen = list(initial_set)
    if not _is_valid_set(chosen):
        return None

    all_points = [(x, y) for x in range(N) for y in range(N)]
    chosen_set = set(chosen)

    def is_maximal_fast(chosen_list, chosen_set_local):
        for p in all_points:
            if p in chosen_set_local:
                continue
            if not _creates_isosceles(p, chosen_list):
                return False
        return True

    # Check initial maximality
    if not is_maximal_fast(chosen, chosen_set):
        # Try to make it maximal by adding points greedily
        changed = True
        while changed:
            changed = False
            for p in all_points:
                if p not in chosen_set and not _creates_isosceles(p, chosen):
                    chosen.append(p)
                    chosen_set.add(p)
                    changed = True
        if not _is_valid_set(chosen):
            return None
        if not is_maximal_fast(chosen, chosen_set):
            return None

    best = len(chosen)
    best_set = set(chosen)
    current_energy = len(chosen)

    for step in range(steps):
        if time.time() > deadline:
            break

        temp = start_temp * (end_temp / start_temp) ** (step / max(steps, 1))

        # Choose a move
        move_type = random.random()
        if move_type < 0.4:
            # Remove a random point
            if len(chosen) <= 1:
                continue
            idx = random.randrange(len(chosen))
            removed = chosen[idx]
            new_chosen = chosen[:idx] + chosen[idx + 1:]
            new_chosen_set = chosen_set - {removed}
            # Check validity (removing can't create isosceles, so it's always valid)
            # Check maximality
            if is_maximal_fast(new_chosen, new_chosen_set):
                new_energy = len(new_chosen)
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < math.exp(-delta_e / max(temp, 1e-9)):
                    chosen = new_chosen
                    chosen_set = new_chosen_set
                    current_energy = new_energy
                    if current_energy < best:
                        best = current_energy
                        best_set = set(chosen)
        elif move_type < 0.7:
            # Add a random point
            p = random.choice(all_points)
            if p in chosen_set:
                continue
            if _creates_isosceles(p, chosen):
                continue
            new_chosen = chosen + [p]
            new_chosen_set = chosen_set | {p}
            if is_maximal_fast(new_chosen, new_chosen_set):
                new_energy = len(new_chosen)
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < math.exp(-delta_e / max(temp, 1e-9)):
                    chosen = new_chosen
                    chosen_set = new_chosen_set
                    current_energy = new_energy
                    if current_energy < best:
                        best = current_energy
                        best_set = set(chosen)
        else:
            # Swap: remove one, add another
            if len(chosen) <= 1:
                continue
            idx = random.randrange(len(chosen))
            removed = chosen[idx]
            p = random.choice(all_points)
            if p == removed or p in chosen_set:
                continue
            # Check if adding p (without removed) creates isosceles
            temp_chosen = chosen[:idx] + chosen[idx + 1:]
            if _creates_isosceles(p, temp_chosen):
                continue
            new_chosen = temp_chosen + [p]
            new_chosen_set = (chosen_set - {removed}) | {p}
            if is_maximal_fast(new_chosen, new_chosen_set):
                new_energy = len(new_chosen)
                delta_e = new_energy - current_energy
                if delta_e < 0 or random.random() < math.exp(-delta_e / max(temp, 1e-9)):
                    chosen = new_chosen
                    chosen_set = new_chosen_set
                    current_energy = new_energy
                    if current_energy < best:
                        best = current_energy
                        best_set = set(chosen)

    return list(best_set)


def construct():
    start_time = time.time()
    deadline = start_time + 330.0 * DEADLINE_FRACTION

    random.seed(42)

    # Phase 1: Greedy with different orderings
    best_set = None
    best_size = float('inf')

    # Try row-major greedy first
    greedy = _greedy_construct()
    if _is_valid_set(greedy) and _is_maximal(greedy):
        best_set = greedy
        best_size = len(greedy)

    # Try several random greedy orderings
    for i in range(200):
        if time.time() > deadline:
            break
        order = _make_random_order()
        g = _greedy_construct(order)
        if _is_valid_set(g):
            if len(g) < best_size:
                # Check maximality
                if _is_maximal(g):
                    best_set = g
                    best_size = len(g)

    if best_set is None:
        return _greedy_construct()

    # Phase 2: Simulated annealing from best greedy result
    if time.time() < deadline:
        sa_result = _simulated_annealing(
            best_set,
            start_temp=5.0,
            end_temp=0.1,
            steps=50000,
            deadline=deadline
        )
        if sa_result is not None and len(sa_result) < best_size:
            best_set = sa_result
            best_size = len(sa_result)

    # Phase 3: More random restarts with SA if time permits
    restarts = 0
    while time.time() < deadline and restarts < 50:
        restarts += 1
        order = _make_random_order()
        g = _greedy_construct(order)
        if not _is_valid_set(g):
            continue
        if len(g) >= best_size:
            continue
        if not _is_maximal(g):
            continue
        best_set = g
        best_size = len(g)

        if time.time() < deadline:
            sa_result = _simulated_annealing(
                best_set,
                start_temp=3.0,
                end_temp=0.05,
                steps=20000,
                deadline=deadline
            )
            if sa_result is not None and len(sa_result) < best_size:
                best_set = sa_result
                best_size = len(sa_result)

    # Final verification
    if best_set is not None:
        if not _is_valid_set(best_set):
            # Fallback to simple greedy
            return _greedy_construct()
        if not _is_maximal(best_set):
            # Try to fix by adding points
            changed = True
            while changed:
                changed = False
                for x in range(N):
                    for y in range(N):
                        p = (x, y)
                        if p not in best_set and not _creates_isosceles(p, best_set):
                            best_set.append(p)
                            changed = True
            if not _is_valid_set(best_set) or not _is_maximal(best_set):
                return _greedy_construct()

    return best_set if best_set is not None else _greedy_construct()
