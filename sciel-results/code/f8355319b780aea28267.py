# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses simulated annealing with a precomputed conflict matrix for fast evaluation.
    """
    m = 205
    deadline = time.time() + 620.0  # safety margin before 630s limit

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Build conflict matrix: conflict[i][j] = True if (i-j) mod m or (j-i) mod m is in forbidden
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(i + 1, m):
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[i, j] = True
                conflict[j, i] = True

    def count_conflicts(selection: set) -> int:
        """Count number of conflicting pairs in the selection."""
        cnt = 0
        elems = list(selection)
        for idx in range(len(elems)):
            a = elems[idx]
            for b in elems[idx + 1:]:
                if conflict[a, b]:
                    cnt += 1
        return cnt

    def try_add(selection: set, elem: int) -> bool:
        """Check if elem can be added without creating conflicts."""
        for s in selection:
            if conflict[elem, s]:
                return False
        return True

    def try_remove(selection: set, elem: int) -> set:
        """Remove elem from selection."""
        new_sel = selection.copy()
        new_sel.discard(elem)
        return new_sel

    # Greedy initialization
    def greedy_construct(order=None):
        if order is None:
            order = list(range(m))
        selected = set()
        for candidate in order:
            if try_add(selected, candidate):
                selected.add(candidate)
        return selected

    # Try multiple greedy orders to find a good starting point
    best_set = set()
    best_size = 0

    # Standard greedy
    s1 = greedy_construct()
    best_set = s1
    best_size = len(s1)

    # Randomized greedy attempts
    rng = random.Random(42)
    for attempt in range(50):
        order = list(range(m))
        rng.shuffle(order)
        s = greedy_construct(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Simulated annealing
    current_set = best_set.copy()
    best_sa_set = best_set.copy()
    best_sa_size = best_size
    current_conflicts = count_conflicts(current_set)

    # Temperature schedule
    T_start = 1.0
    T_end = 0.001
    alpha = 0.9995  # cooling factor

    T = T_start
    step = 0
    max_steps = 5000000  # upper bound, but we'll stop by deadline

    while time.time() < deadline:
        step += 1

        # Choose a random element to toggle
        elem = rng.randint(0, m - 1)

        if elem in current_set:
            # Try removing
            new_set = current_set.copy()
            new_set.discard(elem)
            new_conflicts = count_conflicts(new_set)
            delta_conflicts = new_conflicts - current_conflicts
            delta_size = -1

            # Accept if it reduces conflicts, or with some probability if it doesn't increase too much
            if delta_conflicts < 0:
                # Removing reduces conflicts, but also reduces size
                # Accept if the trade-off is favorable
                # We want to minimize conflicts primarily, then maximize size
                if delta_conflicts <= -1:
                    current_set = new_set
                    current_conflicts = new_conflicts
                else:
                    # delta_conflicts == 0, removing doesn't change conflicts but reduces size
                    # Don't accept unless we're stuck
                    pass
            else:
                # delta_conflicts >= 0, removing increases or maintains conflicts
                # Only accept with small probability
                if delta_conflicts == 0:
                    # Same conflicts, but smaller set - don't accept
                    pass
                else:
                    # Conflicts increase - very unlikely to accept
                    pass
        else:
            # Try adding
            if try_add(current_set, elem):
                new_set = current_set.copy()
                new_set.add(elem)
                new_conflicts = count_conflicts(new_set)
                delta_conflicts = new_conflicts - current_conflicts
                delta_size = 1

                if delta_conflicts == 0:
                    # Adding without new conflicts - always accept
                    current_set = new_set
                    current_conflicts = new_conflicts
                    if len(current_set) > best_sa_size:
                        best_sa_size = len(current_set)
                        best_sa_set = current_set.copy()
                else:
                    # Adding creates conflicts - accept with probability based on temperature
                    # delta_conflicts > 0
                    prob = np.exp(-delta_conflicts / T)
                    if rng.random() < prob:
                        current_set = new_set
                        current_conflicts = new_conflicts
            else:
                # Can't add because of direct conflict check - but we already checked
                # Actually try_add already checks, so this shouldn't happen with conflicts
                pass

        # Occasionally try swap moves: remove one, add another
        if step % 10 == 0 and len(current_set) > 0:
            elems_list = list(current_set)
            if elems_list:
                remove_elem = rng.choice(elems_list)
                add_elem = rng.randint(0, m - 1)
                if add_elem != remove_elem and add_elem not in current_set:
                    new_set = current_set.copy()
                    new_set.discard(remove_elem)
                    # Check if add_elem conflicts with any remaining element
                    can_add = True
                    for s in new_set:
                        if conflict[add_elem, s]:
                            can_add = False
                            break
                    if can_add:
                        new_set.add(add_elem)
                        new_conflicts = count_conflicts(new_set)
                        delta_conflicts = new_conflicts - current_conflicts

                        if delta_conflicts < 0:
                            current_set = new_set
                            current_conflicts = new_conflicts
                            if len(current_set) > best_sa_size:
                                best_sa_size = len(current_set)
                                best_sa_set = current_set.copy()
                        elif delta_conflicts == 0:
                            # Same size, same conflicts - accept to explore
                            if rng.random() < 0.5:
                                current_set = new_set
                                current_conflicts = new_conflicts
                                if len(current_set) > best_sa_size:
                                    best_sa_size = len(current_set)
                                    best_sa_set = current_set.copy()
                        else:
                            prob = np.exp(-delta_conflicts / T)
                            if rng.random() < prob:
                                current_set = new_set
                                current_conflicts = new_conflicts

        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end

    # Final local search polish
    def local_search_polish(selection: set):
        """Try to improve the set by adding elements or swapping."""
        improved = True
        while improved:
            improved = False
            # Try adding any element that doesn't create conflicts
            for elem in range(m):
                if elem not in selection and try_add(selection, elem):
                    selection.add(elem)
                    improved = True
            # Try swapping: remove one, add another
            elems_list = list(selection)
            for rem in elems_list:
                for add in range(m):
                    if add in selection or add == rem:
                        continue
                    new_sel = selection.copy()
                    new_sel.discard(rem)
                    can_add = True
                    for s in new_sel:
                        if conflict[add, s]:
                            can_add = False
                            break
                    if can_add:
                        new_sel.add(add)
                        # Only swap if it doesn't increase conflicts (should be same or better)
                        if count_conflicts(new_sel) <= count_conflicts(selection):
                            if len(new_sel) > len(selection):
                                selection = new_sel
                                improved = True
                                break
                if improved:
                    break
        return selection

    # Polish the best set found
    best_sa_set = local_search_polish(best_sa_set)
    if len(best_sa_set) > best_size:
        best_size = len(best_sa_set)
        best_set = best_sa_set

    # Also polish the original greedy best
    polished_greedy = local_search_polish(best_set.copy())
    if len(polished_greedy) > len(best_set):
        best_set = polished_greedy
        best_size = len(best_set)

    return np.array(sorted(best_set), dtype=np.int64)


# EVOLVE-BLOCK-END
