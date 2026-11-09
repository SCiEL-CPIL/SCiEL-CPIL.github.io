import time
import random
import sys

# Precompute all triples and their masks
ALL_TRIPLES = []
TRIPLE_MASKS = []
PAIR_TO_TRIPLE = {}

for i in range(16):
    for j in range(i + 1, 16):
        for k in range(j + 1, 16):
            t = (i, j, k)
            idx = len(ALL_TRIPLES)
            ALL_TRIPLES.append(t)
            mask = (1 << i) | (1 << j) | (1 << k)
            TRIPLE_MASKS.append(mask)
            for p in [(i, j), (i, k), (j, k)]:
                if p not in PAIR_TO_TRIPLE:
                    PAIR_TO_TRIPLE[p] = []
                PAIR_TO_TRIPLE[p].append(idx)

def compute_alpha_fast(triples):
    """
    Fast independence number computation using a precomputed union-find approach.
    For each mask, we can quickly check if it contains a full triple by looking
    at the precomputed triple_masks and using bitwise operations.
    """
    if not triples:
        return 16

    # Build a list of triple masks for quick checking
    t_masks = [TRIPLE_MASKS[i] for i in triples]

    # Maximum independent set: iterate all 2^16 masks, check independence
    # A mask S is independent if for every triple, (mask & triple_mask) != triple_mask
    # i.e., S does not contain any triple as a subset.

    # Optimization: we can precompute for each triple which masks contain it,
    # but that's 2^16 per triple which is too many.
    # Instead, for each mask, check all triples. With up to ~50 triples, this is fast.

    best = 0
    # Iterate all masks
    for mask in range(1 << 16):
        # Quick check: if popcount(mask) <= best, skip
        cnt = mask.bit_count()
        if cnt <= best:
            continue

        # Check independence
        is_ind = True
        for tm in t_masks:
            if (mask & tm) == tm:
                is_ind = False
                break
        if is_ind:
            best = cnt

    return best

def compute_alpha_ilp(triples):
    """
    Use ILP to compute the independence number exactly.
    """
    try:
        import numpy as np
        from scipy.optimize import milp, LinearConstraint, Bounds

        if not triples:
            return 16

        c = [-1] * 16

        A = []
        for idx in triples:
            row = [0] * 16
            i, j, k = ALL_TRIPLES[idx]
            row[i] = 1
            row[j] = 1
            row[k] = 1
            A.append(row)

        A = np.array(A, dtype=float)
        b = [2] * len(A)

        constraints = LinearConstraint(A, -np.inf, b)
        integrality = [1] * 16
        bounds = Bounds(0, 1)

        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

        if res.success:
            return int(round(sum(res.x)))
        else:
            return compute_alpha_fast(triples)
    except Exception:
        return compute_alpha_fast(triples)

def _fallback_system():
    return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]

def construct_safe():
    try:
        result = construct()
        used_pairs = set()
        for triple in result:
            if len(triple) != 3:
                raise ValueError("Triple must have 3 elements")
            for x in triple:
                if x < 0 or x > 15:
                    raise ValueError("Point out of range")
            a, b, c = sorted(triple)
            if a == b or b == c:
                raise ValueError("Duplicate points in triple")
            for p in [(a, b), (a, c), (b, c)]:
                if p in used_pairs:
                    raise ValueError("Pair conflict")
                used_pairs.add(p)
        return result
    except Exception:
        return _fallback_system()

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    best_alpha = 17
    best_system = []

    def is_valid_triple(triple_idx, used_pairs):
        t = ALL_TRIPLES[triple_idx]
        for p in [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]:
            if p in used_pairs:
                return False
        return True

    def add_triple(triple_idx, system, used_pairs):
        t = ALL_TRIPLES[triple_idx]
        for p in [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]:
            used_pairs[p] = triple_idx
        system.append(triple_idx)

    def remove_triple(triple_idx, system, used_pairs):
        t = ALL_TRIPLES[triple_idx]
        for p in [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]:
            del used_pairs[p]
        system.remove(triple_idx)

    # Phase 1: Generate random maximal packings and keep the best
    random.seed(42)

    # Try multiple random maximal packings
    for restart in range(500):
        if time.time() > deadline:
            break

        system = []
        used_pairs = {}

        shuffled = list(range(len(ALL_TRIPLES)))
        random.shuffle(shuffled)

        for idx in shuffled:
            if is_valid_triple(idx, used_pairs):
                add_triple(idx, system, used_pairs)

        # Try to improve with local search
        alpha = compute_alpha_fast(system)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = list(system)

        # Local search: try to add triples that reduce alpha
        for _ in range(200):
            if time.time() > deadline:
                break

            # Find valid triples
            valid = []
            for idx in range(len(ALL_TRIPLES)):
                if is_valid_triple(idx, used_pairs):
                    valid.append(idx)

            if not valid:
                break

            new_idx = random.choice(valid)
            add_triple(new_idx, system, used_pairs)

            new_alpha = compute_alpha_fast(system)
            if new_alpha <= alpha:
                alpha = new_alpha
                if alpha < best_alpha:
                    best_alpha = alpha
                    best_system = list(system)
            else:
                remove_triple(new_idx, system, used_pairs)

    # Phase 2: Simulated annealing on the best system
    if best_system:
        system = list(best_system)
        used_pairs = {}
        for idx in system:
            t = ALL_TRIPLES[idx]
            for p in [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]:
                used_pairs[p] = idx

        current_alpha = best_alpha
        T = 3.0
        cooling = 0.9995
        sa_start = time.time()

        while time.time() < sa_start + 60.0:
            if time.time() > deadline:
                break

            move_type = random.random()

            if move_type < 0.4:
                # Add a random valid triple
                valid = []
                for idx in range(len(ALL_TRIPLES)):
                    if is_valid_triple(idx, used_pairs):
                        valid.append(idx)

                if valid:
                    new_idx = random.choice(valid)
                    add_triple(new_idx, system, used_pairs)

                    new_alpha = compute_alpha_fast(system)
                    delta = new_alpha - current_alpha

                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_system = list(system)
                    else:
                        remove_triple(new_idx, system, used_pairs)

            elif move_type < 0.7:
                # Remove a random triple
                if system:
                    out_idx = random.choice(system)
                    remove_triple(out_idx, system, used_pairs)

                    new_alpha = compute_alpha_fast(system)
                    delta = new_alpha - current_alpha

                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_system = list(system)
                    else:
                        add_triple(out_idx, system, used_pairs)

            else:
                # Swap
                if system:
                    out_idx = random.choice(system)
                    remove_triple(out_idx, system, used_pairs)

                    valid = []
                    for idx in range(len(ALL_TRIPLES)):
                        if idx != out_idx and is_valid_triple(idx, used_pairs):
                            valid.append(idx)

                    if valid:
                        in_idx = random.choice(valid)
                        add_triple(in_idx, system, used_pairs)

                        new_alpha = compute_alpha_fast(system)
                        delta = new_alpha - current_alpha

                        if delta < 0 or random.random() < pow(2.718, -delta / T):
                            current_alpha = new_alpha
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_system = list(system)
                        else:
                            remove_triple(in_idx, system, used_pairs)
                            add_triple(out_idx, system, used_pairs)
                    else:
                        add_triple(out_idx, system, used_pairs)

            T *= cooling

    # Phase 3: Final ILP check if time permits
    if time.time() < deadline - 10.0 and best_system:
        try:
            alpha = compute_alpha_ilp(best_system)
            if alpha < best_alpha:
                best_alpha = alpha
        except Exception:
            pass

    if not best_system:
        return _fallback_system()

    return [list(ALL_TRIPLES[idx]) for idx in best_system]


if __name__ == "__main__":
    res = construct_safe()
    print(f"Number of triples: {len(res)}")
    triple_indices = []
    for t in res:
        a, b, c = sorted(t)
        for idx in range(len(ALL_TRIPLES)):
            if ALL_TRIPLES[idx] == (a, b, c):
                triple_indices.append(idx)
                break
    alpha = compute_alpha_fast(triple_indices)
    print(f"Independence number: {alpha}")
    print(f"Score: {1.0 / alpha if alpha > 0 else 0}")
