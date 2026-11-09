import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    n = 16

    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Generate all K(2,2,2) configurations and their transversals
    # A K(2,2,2) is determined by 3 disjoint pairs on 6 vertices
    # For each set of 6 vertices, there are 15 ways to partition into 3 pairs
    # Each partition has 8 transversal triples

    config_transversals = []  # List of lists of triple indices
    triple_to_configs = [[] for _ in range(num_triples)]

    # Precompute the 15 matchings on 6 vertices (0,1,2,3,4,5)
    matchings_6 = [
        ((0, 1), (2, 3), (4, 5)),
        ((0, 1), (2, 4), (3, 5)),
        ((0, 1), (2, 5), (3, 4)),
        ((0, 2), (1, 3), (4, 5)),
        ((0, 2), (1, 4), (3, 5)),
        ((0, 2), (1, 5), (3, 4)),
        ((0, 3), (1, 2), (4, 5)),
        ((0, 3), (1, 4), (2, 5)),
        ((0, 3), (1, 5), (2, 4)),
        ((0, 4), (1, 2), (3, 5)),
        ((0, 4), (1, 3), (2, 5)),
        ((0, 4), (1, 5), (2, 3)),
        ((0, 5), (1, 2), (3, 4)),
        ((0, 5), (1, 3), (2, 4)),
        ((0, 5), (1, 4), (2, 3)),
    ]

    for v6 in itertools.combinations(range(n), 6):
        v0, v1, v2, v3, v4, v5 = v6
        for m in matchings_6:
            p1, p2, p3 = m
            a, b = p1
            c, d = p2
            e, f = p3

            transversals = [
                (a, c, e), (a, c, f), (a, d, e), (a, d, f),
                (b, c, e), (b, c, f), (b, d, e), (b, d, f)
            ]

            indices = []
            valid = True
            for t in transversals:
                t_sorted = tuple(sorted(t))
                idx = triple_to_idx.get(t_sorted, -1)
                if idx == -1:
                    valid = False
                    break
                indices.append(idx)

            if valid:
                cfg_idx = len(config_transversals)
                config_transversals.append(indices)
                for t_idx in indices:
                    triple_to_configs[t_idx].append(cfg_idx)

    num_configs = len(config_transversals)

    # State arrays
    selected = [False] * num_triples
    config_counts = [0] * num_configs

    def is_valid_add(t_idx):
        for cfg_idx in triple_to_configs[t_idx]:
            if config_counts[cfg_idx] >= 7:
                return False
        return True

    def add_triple(t_idx):
        selected[t_idx] = True
        for cfg_idx in triple_to_configs[t_idx]:
            config_counts[cfg_idx] += 1

    def remove_triple(t_idx):
        selected[t_idx] = False
        for cfg_idx in triple_to_configs[t_idx]:
            config_counts[cfg_idx] -= 1

    def try_add(t_idx):
        if is_valid_add(t_idx):
            add_triple(t_idx)
            return True
        return False

    def get_blocking_configs(t_idx):
        """Return list of configs that would be violated by adding t_idx"""
        blocking = []
        for cfg_idx in triple_to_configs[t_idx]:
            if config_counts[cfg_idx] >= 7:
                blocking.append(cfg_idx)
        return blocking

    def find_swap(t_add):
        """Try to find a triple to remove to make t_add addable"""
        blocking = get_blocking_configs(t_add)
        if not blocking:
            return None  # Can add directly

        for cfg_block in blocking:
            indices = config_transversals[cfg_block]
            # Find selected triples in this config
            selected_in_cfg = [t for t in indices if selected[t]]
            if not selected_in_cfg:
                continue

            # Try removing each selected triple in this config
            for t_remove in selected_in_cfg:
                # Check if removing t_remove and adding t_add is valid
                # First, check if t_remove's removal helps
                # Temporarily remove t_remove
                for cfg in triple_to_configs[t_remove]:
                    config_counts[cfg] -= 1

                if is_valid_add(t_add):
                    # Restore and return the swap
                    for cfg in triple_to_configs[t_remove]:
                        config_counts[cfg] += 1
                    return t_remove

                # Restore
                for cfg in triple_to_configs[t_remove]:
                    config_counts[cfg] += 1

        return None

    best_system = None
    best_count = 0

    random.seed(42)

    max_runs = 100
    for run in range(max_runs):
        if time.time() > deadline - 5:
            break

        # Reset
        for i in range(num_triples):
            if selected[i]:
                remove_triple(i)
        for i in range(num_configs):
            config_counts[i] = 0

        # Randomized greedy initialization
        order = list(range(num_triples))
        random.shuffle(order)
        for t_idx in order:
            try_add(t_idx)

        current_count = sum(selected)

        # Local search with multiple passes
        improved = True
        max_sweeps = 20
        sweep = 0
        while improved and sweep < max_sweeps and time.time() < deadline - 5:
            improved = False
            sweep += 1

            # Pass 1: Try to add any unselected triple
            for t_idx in range(num_triples):
                if not selected[t_idx]:
                    if try_add(t_idx):
                        current_count += 1
                        improved = True

            if not improved:
                # Pass 2: Try swaps
                unselected = [i for i in range(num_triples) if not selected[i]]
                if not unselected:
                    break

                swap_found = False
                # Try systematic swaps
                for t_add in unselected:
                    if time.time() > deadline - 5:
                        break
                    if is_valid_add(t_add):
                        continue  # Should have been added in pass 1

                    t_remove = find_swap(t_add)
                    if t_remove is not None:
                        remove_triple(t_remove)
                        add_triple(t_add)
                        improved = True
                        swap_found = True

                if not swap_found:
                    # Try random swaps
                    for _ in range(50):
                        if time.time() > deadline - 5:
                            break
                        t_add = random.choice(unselected)
                        if is_valid_add(t_add):
                            continue
                        blocking = get_blocking_configs(t_add)
                        if not blocking:
                            continue
                        cfg_block = random.choice(blocking)
                        indices = config_transversals[cfg_block]
                        selected_in_cfg = [t for t in indices if selected[t]]
                        if not selected_in_cfg:
                            continue
                        t_remove = random.choice(selected_in_cfg)
                        remove_triple(t_remove)
                        if is_valid_add(t_add):
                            add_triple(t_add)
                            improved = True
                            break
                        else:
                            add_triple(t_remove)

        if current_count > best_count:
            best_count = current_count
            best_system = [list(all_triples[i]) for i in range(num_triples) if selected[i]]

    if best_system is None:
        # Fallback: star construction
        return [[0, a, b] for a in range(1, n) for b in range(a + 1, n)]

    return best_system
