"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N = 256  # numbers 1..256, indexed 0..255

    # Precompute conflict: (a*b) % (a+b) == 0 for distinct a, b
    # comp_mask[v] = bitmask of vertices that conflict with v (v is index 0..255, number = v+1)
    comp_mask = [0] * N
    for a_idx in range(N):
        a = a_idx + 1
        mask = 0
        for b_idx in range(N):
            if a_idx == b_idx:
                continue
            b = b_idx + 1
            if (a * b) % (a + b) == 0:
                mask |= (1 << b_idx)
        comp_mask[a_idx] = mask

    # Degree of each vertex
    deg = [comp_mask[i].bit_count() for i in range(N)]

    def greedy_max_independent_set(order):
        """Greedy MIS: process vertices in given order, add if no conflict with already-added."""
        selected_mask = 0
        selected_set = set()
        for v in order:
            if comp_mask[v] & selected_mask:
                continue
            selected_mask |= (1 << v)
            selected_set.add(v)
        return selected_set

    # Try multiple greedy orderings
    best_set = set()
    best_size = 0

    # Ordering 1: minimum degree first
    order_min_deg = sorted(range(N), key=lambda v: (deg[v], v))
    s = greedy_max_independent_set(order_min_deg)
    if len(s) > best_size:
        best_size = len(s)
        best_set = set(s)

    # Ordering 2: maximum degree first
    order_max_deg = sorted(range(N), key=lambda v: (-deg[v], v))
    s = greedy_max_independent_set(order_max_deg)
    if len(s) > best_size:
        best_size = len(s)
        best_set = set(s)

    # Ordering 3: random shuffles
    for trial in range(50):
        if time.time() > deadline - 15:
            break
        order = list(range(N))
        random.shuffle(order)
        s = greedy_max_independent_set(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)

    # Local search: try to improve by removing one element and greedily adding
    current = set(best_set)
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        elems = list(current)
        random.shuffle(elems)
        for a in elems:
            if time.time() > deadline - 10:
                break
            trial_set = current - {a}
            # Build mask of selected vertices
            trial_mask = 0
            for x in trial_set:
                trial_mask |= (1 << x)
            # Try to add all compatible candidates
            candidates = [x for x in range(N) if x not in trial_set]
            random.shuffle(candidates)
            for c in candidates:
                if time.time() > deadline - 10:
                    break
                if not (comp_mask[c] & trial_mask):
                    trial_set.add(c)
                    trial_mask |= (1 << c)
            if len(trial_set) > len(current):
                current = trial_set
                improved = True
                break

    best_set = current
    best_size = len(best_set)

    # 2-swap local search: remove two elements, greedily add back
    current = set(best_set)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        elems = list(current)
        random.shuffle(elems)
        n_elems = len(elems)
        for i in range(min(n_elems, 40)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, min(n_elems, 40)):
                if time.time() > deadline - 5:
                    break
                a, b = elems[i], elems[j]
                trial_set = current - {a, b}
                trial_mask = 0
                for x in trial_set:
                    trial_mask |= (1 << x)
                candidates = [x for x in range(N) if x not in trial_set]
                random.shuffle(candidates)
                for c in candidates:
                    if time.time() > deadline - 5:
                        break
                    if not (comp_mask[c] & trial_mask):
                        trial_set.add(c)
                        trial_mask |= (1 << c)
                if len(trial_set) > len(current):
                    current = trial_set
                    improved = True
                    break
            if improved:
                break

    best_set = current
    best_size = len(best_set)

    # Final polish: try single additions and 1-swap with multiple restarts
    current = set(best_set)
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Try adding any single vertex
        current_mask = 0
        for x in current:
            current_mask |= (1 << x)
        for c in range(N):
            if c in current:
                continue
            if not (comp_mask[c] & current_mask):
                current.add(c)
                current_mask |= (1 << c)
                improved = True
        # 1-swap: try removing each element and see if we can add more than 1
        if not improved:
            elems = list(current)
            random.shuffle(elems)
            for a in elems:
                if time.time() > deadline - 2:
                    break
                trial_set = current - {a}
                trial_mask = 0
                for x in trial_set:
                    trial_mask |= (1 << x)
                candidates = [x for x in range(N) if x not in trial_set]
                random.shuffle(candidates)
                for c in candidates:
                    if time.time() > deadline - 2:
                        break
                    if not (comp_mask[c] & trial_mask):
                        trial_set.add(c)
                        trial_mask |= (1 << c)
                if len(trial_set) > len(current):
                    current = trial_set
                    improved = True
                    break

    best_set = current
    best_size = len(best_set)

    # Convert to list of numbers (1-indexed)
    result = sorted(v + 1 for v in best_set)

    if not result:
        return [1]

    return result
# EVOLVE-BLOCK-END
