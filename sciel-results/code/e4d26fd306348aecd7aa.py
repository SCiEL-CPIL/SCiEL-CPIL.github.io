import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin before 630s kill

    # 1. Precompute forbidden differences (non-zero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        sq = (x * x) % m
        if sq != 0:
            forbidden.add(sq)

    # 2. Build neighbor lists: neighbors[u] = list of v that conflict with u
    neighbors = [[] for _ in range(m)]
    for u in range(m):
        for v in range(m):
            if u != v:
                d = (u - v) % m
                if d in forbidden:
                    neighbors[u].append(v)

    # Also build a set for fast membership check
    neighbor_sets = [set(n) for n in neighbors]

    def is_independent(verts):
        """Check if a list of vertices forms an independent set."""
        vset = set(verts)
        for v in verts:
            for n in neighbors[v]:
                if n in vset:
                    return False
        return True

    def can_add(v, current_set):
        """Check if vertex v can be added to current_set (a set of vertices)."""
        return not any(n in current_set for n in neighbors[v])

    def greedy_construct():
        """Construct a maximal independent set using greedy with random ordering."""
        current_set = set()
        available = list(range(m))
        random.shuffle(available)
        for v in available:
            if v not in current_set and can_add(v, current_set):
                current_set.add(v)
        return current_set

    def local_search(initial_set, max_iters=2000):
        """
        Perform local search to improve the independent set.
        Uses 1-add, 1-swap, and 2-swap moves.
        Returns (best_set, best_size).
        """
        current_set = set(initial_set)
        current_size = len(current_set)
        best_set = set(current_set)
        best_size = current_size
        no_improvement_count = 0
        iterations = 0

        while iterations < max_iters and time.time() < deadline - 5:
            iterations += 1
            improved = False

            comp = set(range(m)) - current_set

            if not comp:
                break

            # 1. Try to add a valid vertex from complement
            comp_list = list(comp)
            random.shuffle(comp_list)
            added = False
            for u in comp_list:
                if can_add(u, current_set):
                    current_set.add(u)
                    current_size += 1
                    improved = True
                    added = True
                    break

            if added:
                no_improvement_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
                continue

            # 2. Try 1-swap: Remove v, add u
            # u must be a neighbor of v in comp, and v must be the only conflict for u
            set_list = list(current_set)
            random.shuffle(set_list)
            swapped = False
            for v in set_list:
                # Candidates for u: in comp, neighbor of v, and v is the only conflict
                candidates = []
                for u in comp:
                    if u in neighbor_sets[v]:
                        # Check if v is the only conflict for u in current_set
                        conflicts = [n for n in neighbors[u] if n in current_set]
                        if len(conflicts) == 1 and conflicts[0] == v:
                            candidates.append(u)

                if candidates:
                    u = random.choice(candidates)
                    current_set.remove(v)
                    current_set.add(u)
                    # Size remains same, but structure improved potentially
                    improved = True
                    swapped = True
                    break

            if swapped:
                no_improvement_count = 0
                continue

            # 3. Try 2-swap: Remove v1, v2, add u1, u2
            if no_improvement_count > 10:
                found_2swap = False
                if len(current_set) >= 2:
                    # Try a few random pairs from set
                    set_list2 = list(current_set)
                    random.shuffle(set_list2)
                    for trial in range(min(5, len(set_list2) - 1)):
                        v1 = set_list2[trial]
                        v2 = set_list2[trial + 1]

                        # Try to find two vertices in comp that can replace them
                        comp_list2 = list(comp)
                        random.shuffle(comp_list2)
                        found = False
                        for i in range(min(15, len(comp_list2))):
                            u1 = comp_list2[i]
                            for j in range(i + 1, min(15, len(comp_list2))):
                                u2 = comp_list2[j]
                                # Check u1, u2 independent
                                if u2 in neighbor_sets[u1]:
                                    continue
                                # Check u1, u2 against current_set without v1, v2
                                test_set = set(current_set)
                                test_set.discard(v1)
                                test_set.discard(v2)
                                if can_add(u1, test_set) and can_add(u2, test_set):
                                    current_set = test_set
                                    current_set.add(u1)
                                    current_set.add(u2)
                                    improved = True
                                    found = True
                                    found_2swap = True
                                    break
                            if found:
                                break
                        if found_2swap:
                            break

                if found_2swap:
                    no_improvement_count = 0
                    continue

            no_improvement_count += 1
            if no_improvement_count > 50:
                break

        return best_set, best_size

    def simulated_annealing(initial_set, max_iters=3000):
        """
        Simulated annealing to escape local optima.
        Accepts worsening moves with decreasing probability.
        """
        current_set = set(initial_set)
        current_size = len(current_set)
        best_set = set(current_set)
        best_size = current_size

        T = 1.0  # initial temperature
        cooling = 0.999

        for it in range(max_iters):
            if time.time() >= deadline - 5:
                break

            comp = set(range(m)) - current_set
            set_list = list(current_set)
            comp_list = list(comp)

            if not comp_list or not set_list:
                break

            # Random move: either add, remove, or swap
            move_type = random.random()
            improved = False

            if move_type < 0.4:
                # Try to add a random vertex from comp
                if comp_list:
                    u = random.choice(comp_list)
                    if can_add(u, current_set):
                        current_set.add(u)
                        current_size += 1
                        improved = True
                    else:
                        # Accept worsening with probability exp(-delta/T)
                        # Here delta = -1 (we're not adding, so no change)
                        # Actually, if we can't add, we just skip
                        pass

            elif move_type < 0.8:
                # Try to remove a random vertex from set and add a random from comp
                v = random.choice(set_list)
                u = random.choice(comp_list)
                # Check if {u} can be added after removing v
                test_set = set(current_set)
                test_set.discard(v)
                if can_add(u, test_set):
                    current_set = test_set
                    current_set.add(u)
                    # Size remains same
                    improved = True

            else:
                # Try to remove a random vertex (worsening move)
                v = random.choice(set_list)
                # Accept with probability exp(-1/T)
                if random.random() < np.exp(-1.0 / T):
                    current_set.discard(v)
                    current_size -= 1

            T *= cooling

            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)

        return best_set, best_size

    best_set = set()
    best_size = 0

    # Run multiple rounds of Greedy + Local Search + Simulated Annealing
    while time.time() < deadline - 30:
        # Randomized Greedy
        init_set = greedy_construct()
        init_size = len(init_set)

        # Local Search
        ls_set, ls_size = local_search(init_set, max_iters=1000)

        # Simulated Annealing on the local search result
        sa_set, sa_size = simulated_annealing(ls_set, max_iters=500)

        if sa_size > best_size:
            best_size = sa_size
            best_set = sa_set

    # Final polish: try to add more to the best solution
    if time.time() < deadline - 10:
        polished_set, polished_size = local_search(best_set, max_iters=5000)
        if polished_size > best_size:
            best_set = polished_set
            best_size = polished_size

    # One more specific attempt: try to add to best_set
    if time.time() < deadline - 5:
        final_set, final_size = local_search(best_set, max_iters=2000)
        if final_size > best_size:
            best_set = final_set
            best_size = final_size

    # Extract result
    result = sorted(best_set)

    if not result:
        # Fallback to a single valid element
        result = [0]

    return np.array(result, dtype=np.int64)
