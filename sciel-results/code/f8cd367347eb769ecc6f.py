"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    import time
    import random
    from itertools import combinations
    from functools import lru_cache

    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin before 90s limit

    n = 128
    elements = list(range(1, n + 1))

    # Precompute LCM table
    def lcm(a, b):
        return a * b // math.gcd(a, b)

    # Precompute lcm[i][j] for all pairs
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Build hyperedges: sets {a,b,c} such that lcm(a,b)==lcm(a,c)==lcm(b,c)
    # Use a set of frozensets for fast lookup, and also a mapping from element to set of hyperedges containing it
    hyperedges = set()
    elem_to_edges = {i: set() for i in range(1, n + 1)}

    for a, b, c in combinations(elements, 3):
        if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
            edge = frozenset((a, b, c))
            hyperedges.add(edge)
            for x in (a, b, c):
                elem_to_edges[x].add(edge)

    # Greedy initial solution: add elements in order, skipping those that form a bad triple
    def greedy_initial():
        selected = set()
        for x in elements:
            # Check if adding x creates any forbidden triple with existing elements
            conflict = False
            for edge in elem_to_edges[x]:
                # If the other two elements of this edge are both in selected, conflict
                other = list(edge - {x})
                if len(other) == 2 and other[0] in selected and other[1] in selected:
                    conflict = True
                    break
            if not conflict:
                selected.add(x)
        return selected

    # Local search with penalty-based simulated annealing
    def local_search(initial_set, max_iters=200000):
        current = set(initial_set)
        best = set(initial_set)
        best_score = len(best)

        # Penalty function: number of forbidden triples in the set
        def count_conflicts(s):
            count = 0
            for x in s:
                for edge in elem_to_edges[x]:
                    other = list(edge - {x})
                    if len(other) == 2 and other[0] in s and other[1] in s:
                        count += 1
            return count // 3  # Each triple counted 3 times (once per element)

        current_conflicts = count_conflicts(current)
        temperature = 10.0
        cooling = 0.9999

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Try a random move: add, remove, or swap
            move_type = random.random()
            if move_type < 0.3:
                # Add a random element not in current
                candidates = [x for x in elements if x not in current]
                if not candidates:
                    continue
                x = random.choice(candidates)
                new_set = current | {x}
                new_conflicts = count_conflicts(new_set)
                delta_conflicts = new_conflicts - current_conflicts
                delta_size = 1
            elif move_type < 0.6:
                # Remove a random element from current
                if not current:
                    continue
                x = random.choice(list(current))
                new_set = current - {x}
                new_conflicts = count_conflicts(new_set)
                delta_conflicts = new_conflicts - current_conflicts
                delta_size = -1
            else:
                # Swap: remove one, add one
                if not current:
                    continue
                x_out = random.choice(list(current))
                candidates = [x for x in elements if x not in current]
                if not candidates:
                    continue
                x_in = random.choice(candidates)
                new_set = (current - {x_out}) | {x_in}
                new_conflicts = count_conflicts(new_set)
                delta_conflicts = new_conflicts - current_conflicts
                delta_size = 0

            # Objective: maximize size, minimize conflicts
            # Score = size - penalty * conflicts
            penalty = 5.0
            old_score = len(current) - penalty * current_conflicts
            new_score = len(new_set) - penalty * new_conflicts
            delta_score = new_score - old_score

            if delta_score > 0 or (temperature > 0.01 and random.random() < min(1.0, delta_score / temperature)):
                current = new_set
                current_conflicts = new_conflicts
                if len(current) > best_score and current_conflicts == 0:
                    best = set(current)
                    best_score = len(best)
                elif current_conflicts == 0 and len(current) > best_score:
                    best = set(current)
                    best_score = len(best)

            temperature *= cooling

        return best, best_score

    # 2-for-3 swap moves to escape local optima
    def swap_2_for_3(current_set):
        """Try to replace 2 elements with 3 elements, maintaining validity."""
        if len(current_set) < 2:
            return current_set, False

        current_list = list(current_set)
        # Try random pairs to remove
        for _ in range(50):
            i, j = random.sample(range(len(current_list)), 2)
            a, b = current_list[i], current_list[j]
            remaining = current_set - {a, b}

            # Find candidates to add that don't create conflicts with remaining
            valid_adds = []
            for x in elements:
                if x in current_set:
                    continue
                # Check if x can be added to remaining without conflict
                can_add = True
                for edge in elem_to_edges[x]:
                    other = list(edge - {x})
                    if len(other) == 2 and other[0] in remaining and other[1] in remaining:
                        can_add = False
                        break
                if can_add:
                    valid_adds.append(x)

            if len(valid_adds) >= 3:
                # Try to find 3 that can all be added together
                for combo in combinations(valid_adds, 3):
                    new_set = remaining | set(combo)
                    if count_conflicts(new_set) == 0:
                        return new_set, True
        return current_set, False

    # Main search loop
    best_solution = greedy_initial()
    best_size = len(best_solution)

    # Run multiple local search iterations
    for iteration in range(50):
        if time.time() > deadline:
            break

        # Start from current best or a perturbed version
        if iteration == 0:
            initial = best_solution
        else:
            # Perturb: randomly remove some elements
            perturbed = set(best_solution)
            num_remove = random.randint(1, min(5, len(perturbed)))
            to_remove = random.sample(list(perturbed), num_remove)
            for x in to_remove:
                perturbed.discard(x)
            initial = perturbed

        solution, size = local_search(initial, max_iters=100000)
        if size > best_size:
            best_solution = solution
            best_size = size

        # Try 2-for-3 swaps
        if time.time() > deadline:
            break
        for _ in range(20):
            if time.time() > deadline:
                break
            new_sol, improved = swap_2_for_3(best_solution)
            if improved and len(new_sol) > best_size:
                best_solution = new_sol
                best_size = len(best_solution)

    # Final verification
    def is_valid(s):
        for edge in hyperedges:
            if edge.issubset(s):
                return False
        return True

    if not is_valid(best_solution):
        # Fall back to greedy
        best_solution = greedy_initial()

    return sorted(best_solution)

# EVOLVE-BLOCK-END
