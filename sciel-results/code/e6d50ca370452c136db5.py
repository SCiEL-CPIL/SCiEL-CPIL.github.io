import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (evaluator kills at 150s)

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)  # 560

    # Precompute index for O(1) lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Precompute all K(2,2,2) configurations
    configs = []
    config_id_map = {}
    triples_in_config = []  # List of lists of 8 triple indices

    points = range(N)
    for pts in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = pts
        pairs = [
            [(p0, p1), (p2, p3), (p4, p5)],
            [(p0, p1), (p2, p4), (p3, p5)],
            [(p0, p1), (p2, p5), (p3, p4)],
            [(p0, p2), (p1, p3), (p4, p5)],
            [(p0, p2), (p1, p4), (p3, p5)],
            [(p0, p2), (p1, p5), (p3, p4)],
            [(p0, p3), (p1, p2), (p4, p5)],
            [(p0, p3), (p1, p4), (p2, p5)],
            [(p0, p3), (p1, p5), (p2, p4)],
            [(p0, p4), (p1, p2), (p3, p5)],
            [(p0, p4), (p1, p3), (p2, p5)],
            [(p0, p4), (p1, p5), (p2, p3)],
            [(p0, p5), (p1, p2), (p3, p4)],
            [(p0, p5), (p1, p3), (p2, p4)],
            [(p0, p5), (p1, p4), (p2, p3)]
        ]
        for pair_set in pairs:
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)

                (a, b), (c, d), (e, f) = norm_pairs
                tris = []
                for x in (a, b):
                    for y in (c, d):
                        for z in (e, f):
                            t = tuple(sorted((x, y, z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)

    num_configs = len(configs)

    # For each triple, which configs does it belong to?
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)

    # State
    present = np.zeros(num_triples, dtype=bool)
    config_counts = np.zeros(num_configs, dtype=np.int8)

    def add_triple(t_idx):
        present[t_idx] = True
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] += 1
            if config_counts[cid] == 8:
                return False
        return True

    def remove_triple(t_idx):
        present[t_idx] = False
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] -= 1
        return True

    def is_valid_addition(t_idx):
        if present[t_idx]:
            return False
        for cid in triple_to_configs[t_idx]:
            if config_counts[cid] == 7:
                return False
        return True

    def get_missing():
        return [i for i in range(num_triples) if not present[i]]

    def run_walksat(seed_idx, time_limit):
        random.seed(seed_idx)
        present[:] = False
        config_counts[:] = 0
        current_set = set()

        # Phase 1: Greedy initialization
        order = list(range(num_triples))
        random.shuffle(order)
        for t_idx in order:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.add(t_idx)

        count = len(current_set)
        best_count = count
        best_set = list(current_set)

        # Phase 2: WalkSAT-style local search
        # Parameters
        flip_prob = 0.3  # Probability of random flip (fluctuation)
        max_steps = 50000
        step = 0

        while step < max_steps and time.time() < deadline - 5:
            missing = get_missing()
            if not missing:
                break

            # Find blocked triples (can't be added)
            blocked = [t for t in missing if not is_valid_addition(t)]
            addable = [t for t in missing if is_valid_addition(t)]

            # If we can add something, do it
            if addable:
                t_idx = random.choice(addable)
                add_triple(t_idx)
                current_set.add(t_idx)
                count += 1
                step += 1
                if count > best_count:
                    best_count = count
                    best_set = list(current_set)
                continue

            # All missing are blocked. Try moves.
            if not blocked:
                break

            # Choose a blocked triple
            t_idx = random.choice(blocked)

            # Find conflicting configs
            conflicts = [cid for cid in triple_to_configs[t_idx] if config_counts[cid] == 7]
            if not conflicts:
                # Shouldn't happen if blocked, but safety
                step += 1
                continue

            # With probability flip_prob, do a random fluctuation (remove random triple)
            if random.random() < flip_prob:
                # Remove a random present triple
                if current_set:
                    u_idx = random.choice(list(current_set))
                    remove_triple(u_idx)
                    current_set.discard(u_idx)
                    count -= 1
                    # Try to add t_idx
                    if is_valid_addition(t_idx):
                        add_triple(t_idx)
                        current_set.add(t_idx)
                        count += 1
                    step += 1
                    if count > best_count:
                        best_count = count
                        best_set = list(current_set)
                continue

            # Otherwise, try a targeted move
            cid = random.choice(conflicts)
            present_in_cid = [t for t in triples_in_config[cid] if present[t]]

            if not present_in_cid:
                step += 1
                continue

            # Try 1-out 1-in: remove one from conflict, add t_idx
            u_idx = random.choice(present_in_cid)
            remove_triple(u_idx)
            current_set.discard(u_idx)
            count -= 1

            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.add(t_idx)
                count += 1
                step += 1
                if count > best_count:
                    best_count = count
                    best_set = list(current_set)
            else:
                # Failed, try 2-out 2-in
                # Remove one more random triple to create space
                if current_set:
                    v_idx = random.choice(list(current_set))
                    if v_idx != t_idx:
                        remove_triple(v_idx)
                        current_set.discard(v_idx)
                        count -= 1

                        if is_valid_addition(t_idx):
                            add_triple(t_idx)
                            current_set.add(t_idx)
                            count += 1
                            step += 1
                            if count > best_count:
                                best_count = count
                                best_set = list(current_set)
                        else:
                            # Restore
                            add_triple(v_idx)
                            current_set.add(v_idx)
                            count += 1
                # Restore u_idx
                add_triple(u_idx)
                current_set.add(u_idx)
                count += 1
                step += 1

        return best_count, best_set

    # Try 2-out/2-in swap moves more aggressively
    def run_swap_search(seed_idx, time_limit):
        random.seed(seed_idx)
        present[:] = False
        config_counts[:] = 0
        current_set = set()

        # Phase 1: Greedy initialization
        order = list(range(num_triples))
        random.shuffle(order)
        for t_idx in order:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.add(t_idx)

        count = len(current_set)
        best_count = count
        best_set = list(current_set)

        # Phase 2: 2-out/2-in swap local search
        max_steps = 30000
        step = 0
        no_improve = 0
        max_no_improve = 50

        while step < max_steps and time.time() < deadline - 5:
            missing = get_missing()
            if not missing:
                break

            blocked = [t for t in missing if not is_valid_addition(t)]
            addable = [t for t in missing if is_valid_addition(t)]

            # Add any addable
            for t_idx in addable:
                if time.time() > deadline - 5:
                    break
                add_triple(t_idx)
                current_set.add(t_idx)
                count += 1
                no_improve = 0

            if count > best_count:
                best_count = count
                best_set = list(current_set)

            if not blocked:
                break

            # Try 2-out/2-in swaps
            t_idx = random.choice(blocked)
            conflicts = [cid for cid in triple_to_configs[t_idx] if config_counts[cid] == 7]

            if not conflicts:
                no_improve += 1
                step += 1
                if no_improve > max_no_improve:
                    # Kick: remove random triples
                    kick_size = random.randint(2, 4)
                    to_remove = random.sample(list(current_set), min(kick_size, len(current_set)))
                    for t in to_remove:
                        remove_triple(t)
                        current_set.discard(t)
                        count -= 1
                    no_improve = 0
                continue

            # Pick a conflict
            cid = random.choice(conflicts)
            present_in_cid = [t for t in triples_in_config[cid] if present[t]]

            if len(present_in_cid) < 1:
                step += 1
                continue

            # Try removing 1 or 2 from the conflict and adding t_idx
            removed = []
            # Try 1-out
            u_idx = random.choice(present_in_cid)
            remove_triple(u_idx)
            removed.append(u_idx)
            count -= 1
            current_set.discard(u_idx)

            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.add(t_idx)
                count += 1
                no_improve = 0
                step += 1
                if count > best_count:
                    best_count = count
                    best_set = list(current_set)
            else:
                # Try 2-out: remove one more
                other_candidates = [t for t in present_in_cid if t != u_idx]
                if other_candidates:
                    v_idx = random.choice(other_candidates)
                    remove_triple(v_idx)
                    removed.append(v_idx)
                    count -= 1
                    current_set.discard(v_idx)

                    if is_valid_addition(t_idx):
                        add_triple(t_idx)
                        current_set.add(t_idx)
                        count += 1
                        no_improve = 0
                        step += 1
                        if count > best_count:
                            best_count = count
                            best_set = list(current_set)
                    else:
                        # Restore
                        for r in removed:
                            add_triple(r)
                            current_set.add(r)
                            count += 1
                        no_improve += 1
                        step += 1
                else:
                    # Restore
                    for r in removed:
                        add_triple(r)
                        current_set.add(r)
                        count += 1
                    no_improve += 1
                    step += 1

            if no_improve > max_no_improve:
                # Kick
                kick_size = random.randint(3, 6)
                if len(current_set) > kick_size:
                    to_remove = random.sample(list(current_set), kick_size)
                    for t in to_remove:
                        remove_triple(t)
                        current_set.discard(t)
                        count -= 1
                no_improve = 0

        return best_count, best_set

    best_count = 0
    best_set = []

    # Run multiple strategies with different seeds
    num_restarts = 0
    while time.time() < deadline - 10:
        seed = num_restarts * 7 + 42

        # Alternate between walksat and swap search
        if num_restarts % 2 == 0:
            count, s = run_walksat(seed, deadline - 10 - time.time())
        else:
            count, s = run_swap_search(seed, deadline - 10 - time.time())

        if count > best_count:
            best_count = count
            best_set = s

        num_restarts += 1

    # Final pass: try to add any remaining addable triples
    for t_idx in best_set:
        if not present[t_idx]:
            add_triple(t_idx)

    # Convert indices back to triples
    result = [all_triples[i] for i in best_set]

    return result
