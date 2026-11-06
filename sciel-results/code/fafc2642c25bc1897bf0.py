import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    def solve(n, time_limit, bias=0.5, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Precomputation
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        # Generate K4s and K5s
        k4s = []
        for k4 in combinations(range(n), 4):
            ts = [triple_to_idx[t] for t in combinations(k4, 3)]
            k4s.append(ts)

        k5s = []
        for k5 in combinations(range(n), 5):
            ts = [triple_to_idx[t] for t in combinations(k5, 3)]
            k5s.append(ts)

        num_k4 = len(k4s)
        num_k5 = len(k5s)

        # Map triple index to affected K4s and K5s
        affected_k4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k4s):
            for t in ts:
                affected_k4[t].append(i)

        affected_k5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k5s):
            for t in ts:
                affected_k5[t].append(i)

        # Precompute triple indices for vectorized updates
        k4_triples_arr = np.array(k4s, dtype=np.int32)
        k5_triples_arr = np.array(k5s, dtype=np.int32)

        # State: use biased initialization
        colors = (np.random.rand(num_triples) < bias).astype(np.int8)

        # Count-based violation tracking:
        # For each K4, count how many of its 4 triples are red (0)
        # For each K5, count how many of its 10 triples are blue (1)
        k4_red_count = np.sum(colors[k4_triples_arr] == 0, axis=1).astype(np.int8)
        k5_blue_count = np.sum(colors[k5_triples_arr] == 1, axis=1).astype(np.int8)

        viol_k4 = (k4_red_count == 4).astype(np.int8)
        viol_k5 = (k5_blue_count == 10).astype(np.int8)

        E = int(np.sum(viol_k4)) + int(np.sum(viol_k5))

        if E == 0:
            return colors.tolist()

        # Local Search with kicks
        T = 1.0
        cooling = 0.99995
        min_T = 0.01
        kick_interval = 5000  # every N steps, do a kick

        step_count = 0
        last_kick = 0

        while time.time() < start_time + time_limit:
            # Check for deadline more frequently
            if time.time() > start_time + time_limit - 0.1:
                break

            step_count += 1

            # Kick mechanism: if stuck, randomly flip several bits
            if step_count - last_kick > kick_interval and E > 0:
                # Flip 3-5 random bits
                num_flips = random.randint(3, 5)
                for _ in range(num_flips):
                    t_idx = random.randint(0, num_triples - 1)
                    old_c = colors[t_idx]
                    new_c = 1 - old_c
                    colors[t_idx] = new_c

                    # Update counts
                    k4_idxs = affected_k4[t_idx]
                    if k4_idxs:
                        for k4_i in k4_idxs:
                            if new_c == 0:
                                k4_red_count[k4_i] += 1
                            else:
                                k4_red_count[k4_i] -= 1

                    k5_idxs = affected_k5[t_idx]
                    if k5_idxs:
                        for k5_i in k5_idxs:
                            if new_c == 1:
                                k5_blue_count[k5_i] += 1
                            else:
                                k5_blue_count[k5_i] -= 1

                # Recompute violations
                old_viol_k4 = viol_k4.copy()
                old_viol_k5 = viol_k5.copy()
                viol_k4 = (k4_red_count == 4).astype(np.int8)
                viol_k5 = (k5_blue_count == 10).astype(np.int8)
                E = int(np.sum(viol_k4)) + int(np.sum(viol_k5))
                last_kick = step_count

            if T < min_T:
                break

            t_idx = random.randint(0, num_triples - 1)
            old_c = colors[t_idx]
            new_c = 1 - old_c

            colors[t_idx] = new_c

            delta = 0

            # Update K4 violations using counts
            k4_idxs = affected_k4[t_idx]
            if k4_idxs:
                for k4_i in k4_idxs:
                    if new_c == 0:
                        k4_red_count[k4_i] += 1
                    else:
                        k4_red_count[k4_i] -= 1

                # Check which K4s became or stopped being violations
                for k4_i in k4_idxs:
                    was_viol = viol_k4[k4_i]
                    is_viol = 1 if k4_red_count[k4_i] == 4 else 0
                    if was_viol != is_viol:
                        viol_k4[k4_i] = is_viol
                        delta += (1 if is_viol else -1)

            # Update K5 violations using counts
            k5_idxs = affected_k5[t_idx]
            if k5_idxs:
                for k5_i in k5_idxs:
                    if new_c == 1:
                        k5_blue_count[k5_i] += 1
                    else:
                        k5_blue_count[k5_i] -= 1

                for k5_i in k5_idxs:
                    was_viol = viol_k5[k5_i]
                    is_viol = 1 if k5_blue_count[k5_i] == 10 else 0
                    if was_viol != is_viol:
                        viol_k5[k5_i] = is_viol
                        delta += (1 if is_viol else -1)

            if delta <= 0:
                E += delta
            else:
                if random.random() < np.exp(-delta / T):
                    E += delta
                else:
                    colors[t_idx] = old_c
                    # Revert counts
                    for k4_i in k4_idxs:
                        if old_c == 0:
                            k4_red_count[k4_i] += 1
                        else:
                            k4_red_count[k4_i] -= 1
                    for k5_i in k5_idxs:
                        if old_c == 1:
                            k5_blue_count[k5_i] += 1
                        else:
                            k5_blue_count[k5_i] -= 1
                    # Revert violations
                    for k4_i in k4_idxs:
                        viol_k4[k4_i] = 1 if k4_red_count[k4_i] == 4 else 0
                    for k5_i in k5_idxs:
                        viol_k5[k5_i] = 1 if k5_blue_count[k5_i] == 10 else 0

            T *= cooling

            if E == 0:
                return colors.tolist()

        return None

    # Try multiple biases and seeds for n=34
    best_result = None

    # Allocate time for different strategies
    time_per_attempt = 15

    attempts = [
        (0.5, 42),
        (0.4, 123),
        (0.6, 456),
        (0.3, 789),
        (0.7, 1011),
        (0.5, 2022),
    ]

    for bias, seed in attempts:
        if time.time() > deadline - 5:
            break

        result = solve(34, time_per_attempt, bias=bias, seed=seed)
        if result is not None:
            return {'n': 34, 'colors': result}

    # Try n=33
    if time.time() < deadline - 10:
        result = solve(33, 10, bias=0.5, seed=42)
        if result is not None:
            return {'n': 33, 'colors': result}

    # Try n=32
    if time.time() < deadline - 5:
        result = solve(32, 5, bias=0.5, seed=42)
        if result is not None:
            return {'n': 32, 'colors': result}

    # Fallback to n=9 (known valid from seed logic)
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
