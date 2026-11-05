import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85

    all_sets = [frozenset(s) for s in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)

    # Precompute pair intersections as a lookup: for each pair (i,j), the intersection frozenset
    # Also precompute which sets contain each possible intersection pattern
    # For efficiency, precompute intersection for all pairs
    pair_inter = [[None] * n_sets for _ in range(n_sets)]
    for i in range(n_sets):
        pair_inter[i][i] = all_sets[i]
        for j in range(i + 1, n_sets):
            pair_inter[i][j] = all_sets[i] & all_sets[j]
            pair_inter[j][i] = pair_inter[i][j]

    def is_valid_addition(cand_idx, family_indices):
        """Check if all_sets[cand_idx] can be added without creating a 3-sunflower."""
        cand = all_sets[cand_idx]
        # Group existing family members by their intersection with cand
        groups = {}
        for idx in family_indices:
            inter = pair_inter[cand_idx][idx]
            key = inter
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)

        for S, members in groups.items():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                mi = members[i]
                for j in range(i + 1, len(members)):
                    if pair_inter[mi][mj] == S:
                        return False
        return True

    best_family = []

    rng = random.Random(42)

    # Try multiple greedy orderings
    num_orders = 20
    for trial in range(num_orders):
        if time.time() > deadline:
            break

        if trial == 0:
            order = list(range(n_sets))
        else:
            order = list(range(n_sets))
            rng.shuffle(order)

        family_indices = []
        family_set = set()

        for idx in order:
            if time.time() > deadline:
                break

            if idx in family_set:
                continue
            if is_valid_addition(idx, family_indices):
                family_indices.append(idx)
                family_set.add(idx)

        if len(family_indices) > len(best_family):
            best_family = list(family_indices)

    # Local search: try to improve by adding random sets
    if time.time() < deadline - 15:
        family_indices = list(best_family)
        family_set = set(family_indices)

        for iteration in range(100):
            if time.time() > deadline:
                break

            improved = False
            # Try multiple random additions per iteration
            for _ in range(50):
                idx = rng.randint(0, n_sets - 1)
                if idx not in family_set:
                    if is_valid_addition(idx, family_indices):
                        family_indices.append(idx)
                        family_set.add(idx)
                        improved = True
                        break

            if not improved:
                # Try swap: remove one, try to add a different one
                if len(family_indices) > 0:
                    removed = None
                    for _ in range(20):
                        pos = rng.randint(0, len(family_indices) - 1)
                        removed_idx = family_indices[pos]
                        family_indices.pop(pos)
                        family_set.remove(removed_idx)
                        # Try to add a different set
                        for _ in range(10):
                            new_idx = rng.randint(0, n_sets - 1)
                            if new_idx not in family_set:
                                if is_valid_addition(new_idx, family_indices):
                                    family_indices.append(new_idx)
                                    family_set.add(new_idx)
                                    improved = True
                                    break
                        if not improved:
                            # Restore
                            family_indices.append(removed_idx)
                            family_set.add(removed_idx)
                        else:
                            break
                    if not improved:
                        break

        if len(family_indices) > len(best_family):
            best_family = list(family_indices)

    return [sorted(all_sets[i]) for i in best_family]
