import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110

    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)  # 560

    # Precompute pairs for each triple
    triple_pairs = []
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        triple_pairs.append(pairs)

    # Precompute triple masks
    triple_masks = []
    for t in all_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)

    # Map from triple to its index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Map each pair to the list of triple indices containing it
    pair_to_triples = {}
    for i, t in enumerate(all_triples):
        for p in triple_pairs[i]:
            if p not in pair_to_triples:
                pair_to_triples[p] = []
            pair_to_triples[p].append(i)

    def get_alpha_fast(triple_list):
        """Fast independence number calculation."""
        if not triple_list:
            return 16

        masks = [triple_masks[triple_to_idx[t]] for t in triple_list]
        n_masks = len(masks)

        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                s_mask = 0
                for p in subset:
                    s_mask |= (1 << p)

                is_ind = True
                for m in masks:
                    if (m & s_mask) == m:
                        is_ind = False
                        break
                if is_ind:
                    return k
        return 0

    def get_alpha_with_set(triple_list):
        """Returns (alpha, independent_set)."""
        if not triple_list:
            return 16, set(points)

        masks = [triple_masks[triple_to_idx[t]] for t in triple_list]

        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                s_mask = 0
                for p in subset:
                    s_mask |= (1 << p)

                is_ind = True
                for m in masks:
                    if (m & s_mask) == m:
                        is_ind = False
                        break
                if is_ind:
                    return k, set(subset)
        return 0, set()

    def can_add(triple_idx, used_pairs):
        t_pairs = triple_pairs[triple_idx]
        for p in t_pairs:
            if p in used_pairs:
                return False
        return True

    best_alpha = 16
    best_triples = []

    num_restarts = 8000
    for i in range(num_restarts):
        if time.time() > deadline:
            break

        # 1. Construct an initial linear system via randomized greedy
        indices = list(range(n_triples))
        random.shuffle(indices)

        current_triples = []
        used_pairs = set()

        for idx in indices:
            if can_add(idx, used_pairs):
                t = all_triples[idx]
                current_triples.append(t)
                for p in triple_pairs[idx]:
                    used_pairs.add(p)

        # Initial alpha
        alpha = get_alpha_fast(current_triples)

        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = list(current_triples)

        # Local improvement loop
        improved = True
        while improved and time.time() < deadline:
            improved = False
            alpha, indep_set = get_alpha_with_set(current_triples)

            if alpha <= best_alpha:
                break

            # Try to add a triple fully inside the independent set
            candidates = []
            for idx in range(n_triples):
                t = all_triples[idx]
                if all(p in indep_set for p in t):
                    if can_add(idx, used_pairs):
                        candidates.append(idx)

            if candidates:
                idx = random.choice(candidates)
                t = all_triples[idx]
                current_triples.append(t)
                for p in triple_pairs[idx]:
                    used_pairs.add(p)
                improved = True
            else:
                # No triple can be added inside the independent set.
                # Try a swap: remove a random triple, add a different one.
                if current_triples and random.random() < 0.4:
                    rem_pos = random.randrange(len(current_triples))
                    t_rem = current_triples.pop(rem_pos)
                    rem_idx = triple_to_idx[t_rem]
                    for p in triple_pairs[rem_idx]:
                        used_pairs.discard(p)
                    # Find a replacement that is compatible
                    new_candidates = []
                    for idx2 in range(n_triples):
                        if can_add(idx2, used_pairs):
                            # Prefer triples that hit the independent set
                            t2 = all_triples[idx2]
                            if any(p in indep_set for p in t2):
                                new_candidates.append(idx2)
                    if new_candidates:
                        idx2 = random.choice(new_candidates)
                        t2 = all_triples[idx2]
                        current_triples.append(t2)
                        for p in triple_pairs[idx2]:
                            used_pairs.add(p)
                        improved = True

        # Final check
        alpha = get_alpha_fast(current_triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = list(current_triples)

    return best_triples
