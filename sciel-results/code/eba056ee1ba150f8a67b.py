"""Circle packing in unit square using NSGA-II multi-objective optimization."""
import numpy as np

N = 26


def _pairwise_dists(pos):
    """Compute all pairwise distances between circle centers."""
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    return d


def _objective(pop):
    """
    Compute objectives for a population.
    Returns (overlap_penalty, boundary_penalty, neg_sum_radii).
    Lower is better for each.
    """
    n_pop = pop.shape[0]
    n = pop.shape[1]
    overlap = np.zeros(n_pop)
    boundary = np.zeros(n_pop)
    neg_sum_r = np.zeros(n_pop)

    for i in range(n_pop):
        x = pop[i, 0, :]
        y = pop[i, 1, :]
        r = pop[i, 2, :]

        # Boundary violations: circle must be inside [0,1]^2
        # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        bv = np.maximum(0.0, r - x) + np.maximum(0.0, r - (1.0 - x)) + \
             np.maximum(0.0, r - y) + np.maximum(0.0, r - (1.0 - y))
        boundary[i] = np.sum(bv)

        # Pairwise overlap: for each pair (i,j), overlap = max(0, r_i + r_j - dist_ij)
        d = _pairwise_dists(np.stack([x, y], axis=-1))
        rsum = r[:, None] + r[None, :]
        ov = np.maximum(0.0, rsum - d)
        overlap[i] = np.sum(ov)

        neg_sum_r[i] = -np.sum(r)

    return overlap, boundary, neg_sum_r


def _repair(pop):
    """
    Repair population to be feasible: clip positions and shrink radii to fit.
    This is a simple heuristic repair.
    """
    n_pop = pop.shape[0]
    n = pop.shape[1]
    repaired = pop.copy()

    for i in range(n_pop):
        x = repaired[i, 0, :].copy()
        y = repaired[i, 1, :].copy()
        r = repaired[i, 2, :].copy()

        # Clip positions to [0,1]
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)

        # Shrink radii so they fit within boundary
        max_r_boundary = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        r = np.minimum(r, np.maximum(0.0, max_r_boundary))

        # Iteratively resolve overlaps by shrinking radii
        for _ in range(20):
            d = _pairwise_dists(np.stack([x, y], axis=-1))
            rsum = r[:, None] + r[None, :]
            # If r_i + r_j > d_ij, we need to reduce
            # Simple approach: for each pair, if overlapping, reduce both
            changed = False
            for a in range(n):
                for b in range(a + 1, n):
                    if r[a] + r[b] > d[a, b]:
                        # Reduce both radii to make them just touch
                        total = r[a] + r[b]
                        excess = total - d[a, b]
                        # Split excess
                        r[a] = np.maximum(0.0, r[a] - excess / 2.0)
                        r[b] = np.maximum(0.0, r[b] - excess / 2.0)
                        changed = True
            if not changed:
                break

        # Final boundary check
        max_r_boundary = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        r = np.minimum(r, np.maximum(0.0, max_r_boundary))

        repaired[i, 0, :] = x
        repaired[i, 1, :] = y
        repaired[i, 2, :] = r

    return repaired


def _fast_non_dominated_sort(objs):
    """
    Fast non-dominated sorting.
    objs: (n_pop, n_obj) array, lower is better.
    Returns list of fronts, each front is a list of indices.
    """
    n_pop = objs.shape[0]
    n_obj = objs.shape[1]

    domination_count = np.zeros(n_pop, dtype=int)
    dominated_by = [list() for _ in range(n_pop)]

    for i in range(n_pop):
        for j in range(i + 1, n_pop):
            # Check if i dominates j
            i_dom_j = True
            j_dom_i = True
            for k in range(n_obj):
                if objs[i, k] < objs[j, k]:
                    j_dom_i = False
                elif objs[i, k] > objs[j, k]:
                    i_dom_j = False
            if i_dom_j and not j_dom_i:
                domination_count[j] += 1
                dominated_by[i].append(j)
            elif j_dom_i and not i_dom_j:
                domination_count[i] += 1
                dominated_by[j].append(i)

    fronts = []
    current_front = [i for i in range(n_pop) if domination_count[i] == 0]
    if not current_front:
        return [list(range(n_pop))]

    front_index = 0
    while current_front:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front
        front_index += 1

    # Any remaining points (shouldn't happen normally)
    remaining = [i for i in range(n_pop) if i not in [x for f in fronts for x in f]]
    if remaining:
        fronts.append(remaining)

    return fronts


def _crowding_distance(objs, front):
    """
    Compute crowding distance for a front.
    Returns array of crowding distances for indices in front.
    """
    n_front = len(front)
    if n_front <= 2:
        return np.full(n_front, np.inf)

    n_obj = objs.shape[1]
    crowd = np.zeros(n_front)

    for k in range(n_obj):
        # Sort front by objective k
        sorted_idx = sorted(range(n_front), key=lambda i: objs[front[i], k])
        crowd[sorted_idx[0]] = np.inf
        crowd[sorted_idx[-1]] = np.inf

        obj_min = objs[front[sorted_idx[0]], k]
        obj_max = objs[front[sorted_idx[-1]], k]
        range_obj = obj_max - obj_min
        if range_obj < 1e-12:
            continue

        for i in range(1, n_front - 1):
            crowd[sorted_idx[i]] += (objs[front[sorted_idx[i + 1]], k] -
                                      objs[front[sorted_idx[i - 1]], k]) / range_obj

    return crowd


def _select_next_generation(pop, objs, n_elite, n_offspring):
    """
    Select next generation using NSGA-II.
    Returns indices of selected individuals.
    """
    n_pop = pop.shape[0]
    fronts = _fast_non_dominated_sort(objs)

    selected = []
    for front in fronts:
        if len(selected) + len(front) <= n_elite + n_offspring:
            selected.extend(front)
        else:
            # Need to select from this front
            need = n_elite + n_offspring - len(selected)
            crowd = _crowding_distance(objs, front)
            # Sort by crowding distance descending
            sorted_front = sorted(range(len(front)), key=lambda i: -crowd[i])
            selected.extend([front[i] for i in sorted_front[:need]])
            break

    return selected


def construct():
    """
    NSGA-II based circle packing in unit square.
    Returns array of shape (N, 3) with (x, y, radius) for each circle.
    """
    n_pop = 100
    n_gen = 50
    n = N

    # Initialize population
    # Mix of random and grid-like initializations
    pop = np.zeros((n_pop, 3, n))

    for i in range(n_pop):
        if i < n_pop // 3:
            # Grid-like initialization
            side = int(np.ceil(np.sqrt(n)))
            for j in range(n):
                col = j % side
                row = j // side
                pop[i, 0, j] = (col + 0.5) / side + np.random.uniform(-0.1, 0.1)
                pop[i, 1, j] = (row + 0.5) / side + np.random.uniform(-0.1, 0.1)
                pop[i, 2, j] = 0.5 / side * np.random.uniform(0.8, 1.0)
        elif i < 2 * n_pop // 3:
            # Random positions, small radii
            pop[i, 0, :] = np.random.uniform(0.1, 0.9, n)
            pop[i, 1, :] = np.random.uniform(0.1, 0.9, n)
            pop[i, 2, :] = np.random.uniform(0.02, 0.1, n)
        else:
            # Random with larger radii
            pop[i, 0, :] = np.random.uniform(0.05, 0.95, n)
            pop[i, 1, :] = np.random.uniform(0.05, 0.95, n)
            pop[i, 2, :] = np.random.uniform(0.05, 0.18, n)

    # Initial repair
    pop = _repair(pop)

    # NSGA-II loop
    best_sum_r = -np.inf
    best_pop = pop.copy()

    for gen in range(n_gen):
        # Evaluate objectives
        overlap, boundary, neg_sum_r = _objective(pop)
        objs = np.stack([overlap, boundary, neg_sum_r], axis=-1)

        # Track best feasible solution
        feasible_mask = (overlap < 1e-6) & (boundary < 1e-6)
        if np.any(feasible_mask):
            idx = np.where(feasible_mask)[0]
            sum_r = -objs[idx, 2]
            best_idx = idx[np.argmax(sum_r)]
            if sum_r[np.argmax(sum_r)] > best_sum_r:
                best_sum_r = sum_r[np.argmax(sum_r)]
                best_pop = pop[best_idx:best_idx+1].copy()

        # Selection
        n_elite = n_pop // 5
        n_offspring = n_pop - n_elite
        selected_idx = _select_next_generation(pop, objs, n_elite, n_offspring)
        new_pop = pop[selected_idx].copy()

        # Crossover and mutation
        for i in range(n_pop):
            if np.random.random() < 0.9:
                # Crossover
                j = np.random.randint(n_pop)
                if i != j:
                    # Blend crossover
                    alpha = np.random.uniform(0.3, 0.7)
                    new_pop[i] = alpha * pop[i] + (1 - alpha) * pop[j]
            # Mutation
            if np.random.random() < 0.8:
                # Position mutation
                mut_mask = np.random.random(n) < 0.3
                new_pop[i, 0, mut_mask] += np.random.normal(0, 0.05, np.sum(mut_mask))
                new_pop[i, 1, mut_mask] += np.random.normal(0, 0.05, np.sum(mut_mask))
                # Radius mutation
                r_mut_mask = np.random.random(n) < 0.2
                new_pop[i, 2, r_mut_mask] *= np.random.uniform(0.8, 1.2, np.sum(r_mut_mask))

        # Clip and repair
        new_pop[:, 0, :] = np.clip(new_pop[:, 0, :], 0.0, 1.0)
        new_pop[:, 1, :] = np.clip(new_pop[:, 1, :], 0.0, 1.0)
        new_pop[:, 2, :] = np.clip(new_pop[:, 2, :], 0.001, 0.5)
        new_pop = _repair(new_pop)
        pop = new_pop

    # Final repair of best solution
    best_pop = _repair(best_pop)

    # Verify and return
    x = best_pop[0, 0, :]
    y = best_pop[0, 1, :]
    r = best_pop[0, 2, :]

    # Final boundary enforcement
    max_r = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    r = np.minimum(r, np.maximum(0.001, max_r))

    result = np.stack([x, y, r], axis=-1)
    return result
