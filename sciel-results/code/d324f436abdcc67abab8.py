import itertools
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    points = list(range(16))
    n_points = 16

    # Precompute all triples
    all_triples = list(itertools.combinations(points, 3))
    n_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all transversals (sets of 8 triple indices)
    # For each 6-subset, generate all 15 perfect matchings
    transversals = []
    for six_pts in itertools.combinations(points, 6):
        sp = list(six_pts)
        # Generate all 15 perfect matchings on 6 labeled positions
        # Fix sp[0] paired with sp[1..5], then match remaining 4
        matchings = set()
        # Enumerate: pair sp[0] with sp[i] for i in 1..5, then pair remaining 4 in 3 ways
        for i in range(1, 6):
            rem = [j for j in range(6) if j != 0 and j != i]
            # rem has 4 elements; 3 perfect matchings
            a, b, c, d = rem
            # Matchings: (a,b)(c,d), (a,c)(b,d), (a,d)(b,c)
            for m in [((a,b),(c,d)), ((a,c),(b,d)), ((a,d),(b,c))]:
                pairs = [(0, i)] + m
                key = tuple(sorted(tuple(sorted((sp[p], sp[q]))) for p, q in pairs))
                matchings.add(key)

        for m in matchings:
            (p1, p2, p3) = m
            a, b = p1
            c, d = p2
            e, f = p3
            t_indices = []
            for x in (a, b):
                for y in (c, d):
                    for z in (e, f):
                        t = tuple(sorted((x, y, z)))
                        t_indices.append(triple_to_idx[t])
            transversals.append(tuple(sorted(t_indices)))

    n_transversals = len(transversals)

    # Build triple -> list of transversals it belongs to
    triple_to_transversals = [[] for _ in range(n_triples)]
    for ti, trans in enumerate(transversals):
        for t_idx in trans:
            triple_to_transversals[t_idx].append(ti)

    # Convert to numpy for speed
    # Represent transversals as a list of arrays
    trans_arrays = [np.array(t, dtype=np.int32) for t in transversals]

    def solve_with_seed(seed):
        rng = np.random.default_rng(seed)
        current = np.ones(n_triples, dtype=bool)
        # conflict_count[ti] = number of present triples in transversal ti
        conflict = np.array([8] * n_transversals, dtype=np.int32)

        def is_valid():
            return np.all(conflict < 8)

        # Greedy removal phase
        while not is_valid() and time.time() < deadline - 30:
            conflicting = np.where(conflict == 8)[0]
            if len(conflicting) == 0:
                break
            # Pick the conflicting transversal with most "pressure"
            # Choose a random conflicting transversal
            trans_idx = rng.choice(conflicting)
            trans = trans_arrays[trans_idx]
            # Choose the triple in this transversal that appears in most conflicting transversals
            # Approximate: pick triple with highest degree among conflicting
            best_t = None
            best_score = -1
            for t_idx in trans:
                # Count how many of its transversals are conflicting
                score = 0
                for tj in triple_to_transversals[t_idx]:
                    if conflict[tj] == 8:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_t = t_idx
            if best_t is None:
                best_t = rng.choice(trans)
            current[best_t] = False
            for tj in triple_to_transversals[best_t]:
                conflict[tj] -= 1

        # Greedy fill: repeatedly add any triple that doesn't create conflict
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            absent = np.where(~current)[0]
            rng.shuffle(absent)
            for t_idx in absent:
                if time.time() > deadline - 20:
                    break
                can_add = True
                for tj in triple_to_transversals[t_idx]:
                    if conflict[tj] == 7:
                        can_add = False
                        break
                if can_add:
                    current[t_idx] = True
                    for tj in triple_to_transversals[t_idx]:
                        conflict[tj] += 1
                    improved = True

        best_count = np.sum(current)
        best_state = current.copy()
        best_conflict = conflict.copy()

        # Local search: simulated annealing with swaps
        # Move: remove k present, try to add up to k+1 absent
        temp = 5.0
        cooling = 0.995
        min_temp = 0.1
        step = 0
        max_steps = 200000
        while temp > min_temp and time.time() < deadline - 10 and step < max_steps:
            step += 1
            # Pick a random present triple to remove
            present = np.where(current)[0]
            if len(present) == 0:
                break
            remove_t = rng.choice(present)
            # Save state
            old_current = current.copy()
            old_conflict = conflict.copy()

            current[remove_t] = False
            for tj in triple_to_transversals[remove_t]:
                conflict[tj] -= 1

            # Try to add back as many absent triples as possible
            absent = np.where(~current)[0]
            rng.shuffle(absent)
            added = 0
            for add_t in absent:
                if time.time() > deadline - 10:
                    break
                can_add = True
                for tj in triple_to_transversals[add_t]:
                    if conflict[tj] == 7:
                        can_add = False
                        break
                if can_add:
                    current[add_t] = True
                    for tj in triple_to_transversals[add_t]:
                        conflict[tj] += 1
                    added += 1

            new_count = np.sum(current)
            delta = new_count - best_count

            if delta > 0 or rng.random() < np.exp(delta / temp):
                best_count = new_count
                best_state = current.copy()
                best_conflict = conflict.copy()
            else:
                current = old_current
                conflict = old_conflict

            temp *= cooling

        return best_state, int(best_count)

    # Run multiple restarts
    best_state = None
    best_count = 0
    seeds = [42, 123, 7, 999, 314, 271, 555, 888]
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        state, count = solve_with_seed(seed)
        if count > best_count:
            best_count = count
            best_state = state

    # Final greedy fill on best state
    if best_state is not None:
        current = best_state.copy()
        conflict = np.zeros(n_transversals, dtype=np.int32)
        # Recompute conflict from current
        for ti, trans in enumerate(trans_arrays):
            conflict[ti] = np.sum(current[trans])
        # Greedy fill
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            absent = np.where(~current)[0]
            rng = np.random.default_rng(0)
            rng.shuffle(absent)
            for t_idx in absent:
                if time.time() > deadline - 5:
                    break
                can_add = True
                for tj in triple_to_transversals[t_idx]:
                    if conflict[tj] == 7:
                        can_add = False
                        break
                if can_add:
                    current[t_idx] = True
                    for tj in triple_to_transversals[t_idx]:
                        conflict[tj] += 1
                    improved = True
        best_state = current
        best_count = int(np.sum(best_state))

    result = [list(all_triples[i]) for i in range(n_triples) if best_state[i]]
    return result
