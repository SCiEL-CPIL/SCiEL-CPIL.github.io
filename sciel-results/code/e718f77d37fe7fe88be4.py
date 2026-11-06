"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import math
    from fractions import Fraction

    deadline = time.time() + 55  # 55 second deadline for safety

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    unit_vals = [Fraction(1, d) for d in denominators]

    def greedy(order):
        selected = set()
        total = Fraction(0)
        for idx in order:
            v = unit_vals[idx]
            if total + v < 1:
                selected.add(idx)
                total += v
        return selected, total

    best_indices = []
    best_total = Fraction(0)
    best_deficit = Fraction(1)

    # Try multiple orderings for greedy
    random.seed(123)
    orderings = [
        list(range(n)),
        list(reversed(range(n))),
        sorted(range(n), key=lambda i: -denominators[i]),
        sorted(range(n), key=lambda i: denominators[i]),
    ]
    for _ in range(50):
        o = list(range(n))
        random.shuffle(o)
        orderings.append(o)

    for order in orderings:
        idxs, total = greedy(order)
        if idxs:
            deficit = 1 - total
            if deficit > 0 and deficit < best_deficit:
                best_deficit = deficit
                best_total = total
                best_indices = sorted(idxs)

    # Local search with simulated annealing
    def local_search(start_set, max_iterations=50000):
        nonlocal best_indices, best_total, best_deficit

        if not start_set:
            return

        cur = set(start_set)
        cur_total = sum(unit_vals[i] for i in cur)
        cur_deficit = Fraction(1) - cur_total
        if cur_deficit <= 0:
            return

        not_in_set = set(range(n)) - cur
        temperature = Fraction(1, 1000)
        cooling = 0.999

        improved = True
        iters = 0
        while iters < max_iterations and time.time() < deadline:
            iters += 1

            # Choose a random move
            move_type = random.random()
            best_new_def = cur_deficit
            best_new_set = None
            best_new_total = None

            if move_type < 0.3 and not_in_set:
                # Add a random element
                add = random.choice(tuple(not_in_set))
                nt = cur_total + unit_vals[add]
                if nt < 1:
                    nd = Fraction(1) - nt
                    if nd < best_new_def:
                        best_new_def = nd
                        best_new_set = cur | {add}
                        best_new_total = nt
            elif move_type < 0.6 and len(cur) > 1:
                # Remove a random element
                rm = random.choice(tuple(cur))
                nt = cur_total - unit_vals[rm]
                nd = Fraction(1) - nt
                if nd < best_new_def:
                    best_new_def = nd
                    best_new_set = cur - {rm}
                    best_new_total = nt
            elif move_type < 0.9 and len(cur) > 0 and not_in_set:
                # 1-swap
                rm = random.choice(tuple(cur))
                add = random.choice(tuple(not_in_set))
                nt = cur_total - unit_vals[rm] + unit_vals[add]
                if nt < 1:
                    nd = Fraction(1) - nt
                    if nd < best_new_def:
                        best_new_def = nd
                        best_new_set = (cur - {rm}) | {add}
                        best_new_total = nt
            else:
                # 2-swap
                if len(cur) >= 2 and len(not_in_set) >= 2:
                    rm1, rm2 = random.sample(tuple(cur), 2)
                    add1, add2 = random.sample(tuple(not_in_set), 2)
                    nt = cur_total - unit_vals[rm1] - unit_vals[rm2] + unit_vals[add1] + unit_vals[add2]
                    if nt < 1:
                        nd = Fraction(1) - nt
                        if nd < best_new_def:
                            best_new_def = nd
                            best_new_set = (cur - {rm1, rm2}) | {add1, add2}
                            best_new_total = nt

            if best_new_set is not None:
                # Accept if better or with probability exp(-delta/T)
                delta = best_new_def - cur_deficit
                if delta < 0:
                    # Better, always accept
                    cur = best_new_set
                    cur_total = best_new_total
                    cur_deficit = best_new_def
                    not_in_set = set(range(n)) - cur
                    if cur_deficit < best_deficit:
                        best_deficit = cur_deficit
                        best_total = cur_total
                        best_indices = sorted(cur)
                else:
                    # Worse, accept with probability
                    if random.random() < math.exp(-float(delta) / float(temperature)):
                        cur = best_new_set
                        cur_total = best_new_total
                        cur_deficit = best_new_def
                        not_in_set = set(range(n)) - cur
                    temperature *= cooling

        return cur_deficit

    # Run local search from best greedy
    local_search(set(best_indices), max_iterations=100000)

    # Try random restarts with local search
    for restart in range(300):
        if time.time() >= deadline:
            break
        o = list(range(n))
        random.shuffle(o)
        idxs, total = greedy(o)
        if idxs:
            deficit = 1 - total
            if deficit > 0 and deficit < best_deficit:
                best_deficit = deficit
                best_total = total
                best_indices = sorted(idxs)
            local_search(idxs, max_iterations=5000)

    # Final: ensure we return a valid solution
    if not best_indices:
        best_indices = [0]
        best_total = Fraction(1, 2)

    return best_indices
# EVOLVE-BLOCK-END
