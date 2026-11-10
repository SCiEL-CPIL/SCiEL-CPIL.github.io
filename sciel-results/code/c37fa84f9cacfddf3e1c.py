# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 620  # 620s out of 630s, safety margin

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Precompute conflict matrix: conflict[a][b] = True if (a-b)%m or (b-a)%m is in forbidden
    conflict = np.zeros((m, m), dtype=bool)
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            d1 = (a - b) % m
            d2 = (b - a) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[a, b] = True

    def can_add_fast(selected, candidate):
        """Check if candidate can be added to selected using conflict matrix."""
        for s in selected:
            if conflict[candidate, s]:
                return False
        return True

    def greedy_construct(order):
        """Greedy construction following a given order."""
        selected = []
        selected_set = set()
        for candidate in order:
            ok = True
            for s in selected_set:
                if conflict[candidate, s]:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
                selected_set.add(candidate)
        return selected

    def local_search_1move(selected, max_iter=10000):
        """Local search with 1-move: try adding elements not in set."""
        selected = list(selected)
        selected_set = set(selected)
        
        for _ in range(max_iter):
            improved = False
            for a in range(m):
                if a in selected_set:
                    continue
                if can_add_fast(selected, a):
                    selected.append(a)
                    selected_set.add(a)
                    improved = True
                    break
            if not improved:
                break
        
        return selected

    def local_search_1swap(selected, max_iter=5000):
        """Local search with 1-swap moves: remove one, add one."""
        selected = list(selected)
        selected_set = set(selected)
        
        for _ in range(max_iter):
            improved = False
            for r in list(selected):
                for a in range(m):
                    if a in selected_set:
                        continue
                    new_selected = [x for x in selected if x != r]
                    if can_add_fast(new_selected, a):
                        selected = new_selected + [a]
                        selected_set = set(selected)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return selected

    def simulated_annealing(initial_set, time_budget, seed=42):
        """
        Simulated annealing using conflict matrix for speed.
        """
        start_time = time.time()
        rng = random.Random(seed)
        
        current = list(initial_set)
        current_set = set(current)
        current_size = len(current)
        best_size = current_size
        best_set = set(current)
        
        T = 1.0
        cooling = 0.9995
        min_T = 0.005
        
        while T > min_T and (time.time() - start_time) < time_budget:
            if len(current) == 0:
                break
            
            # Generate a neighbor: remove one random element and add one random element
            rem = rng.choice(current)
            add = rng.randint(0, m - 1)
            
            if add == rem:
                T *= cooling
                continue
            
            # Check if add can be added to current minus rem
            new_set = current_set - {rem}
            
            can_add = True
            for s in new_set:
                if conflict[add, s]:
                    can_add = False
                    break
            
            if not can_add:
                T *= cooling
                continue
            
            new_set.add(add)
            new_size = len(new_set)
            
            # Acceptance criterion
            if new_size >= current_size:
                current = list(new_set)
                current_set = new_set
                current_size = new_size
                if new_size > best_size:
                    best_size = new_size
                    best_set = set(current)
            else:
                # New size is smaller; accept with probability exp(-delta/T)
                delta = current_size - new_size
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current = list(new_set)
                    current_set = new_set
                    current_size = new_size
            
            T *= cooling
        
        return best_set

    def is_valid_set(s_set):
        """Verify the set is valid."""
        s_list = list(s_set)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                a, b = s_list[i], s_list[j]
                if conflict[a, b]:
                    return False
        return True

    best_set = set()
    best_size = 0

    # Phase 1: Multiple random greedy constructions with local search
    for seed in range(30):
        if time.time() > deadline - 180:
            break
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        greedy_result = greedy_construct(order)
        
        # Apply local search
        improved = local_search_1move(greedy_result)
        improved = local_search_1swap(improved)
        
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = set(improved)

    # Phase 2: Simulated annealing from best solution
    remaining = deadline - time.time() - 15
    if remaining > 10:
        sa_result = simulated_annealing(list(best_set), time_budget=remaining, seed=123)
        if len(sa_result) > best_size:
            best_size = len(sa_result)
            best_set = sa_result

    # Phase 3: Final local search polish
    if time.time() < deadline - 5:
        final = local_search_1move(list(best_set))
        final = local_search_1swap(final)
        if len(final) > best_size:
            best_size = len(final)
            best_set = set(final)

    # Verify the best set is valid
    if not is_valid_set(best_set):
        # Fallback to a simple greedy
        fallback = greedy_construct(list(range(m)))
        fallback = local_search_1move(fallback)
        fallback = local_search_1swap(fallback)
        best_set = set(fallback)

    return np.array(sorted(list(best_set)), dtype=np.int64)


# EVOLVE-BLOCK-END
