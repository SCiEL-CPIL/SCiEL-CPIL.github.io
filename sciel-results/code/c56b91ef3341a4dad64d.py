import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def verify(n, colors):
        """Verify coloring has no red K4 or blue K5."""
        triple_to_idx = {}
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_to_idx[(a,b,c)] = idx
                    idx += 1
        for k4 in combinations(range(n), 4):
            for t in combinations(k4, 3):
                if colors[triple_to_idx[t]] != 0:
                    break
            else:
                return False  # red K4 found
        for k5 in combinations(range(n), 5):
            all_blue = True
            for t in combinations(k5, 3):
                if colors[triple_to_idx[t]] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False  # blue K5 found
        return True

    def cyclic_coloring(n, shift=0):
        """Cyclic coloring: color triple (a,b,c) red if (a+b+c) mod n is in a specific set."""
        triples = list(combinations(range(n), 3))
        colors = []
        for (a, b, c) in triples:
            s = (a + b + c) % n
            # Use a threshold-based approach
            colors.append(0 if s < n // 2 else 1)
        return colors

    def vectorized_solve(n, time_limit, seed=42):
        """Vectorized local search for n=34."""
        rng = np.random.RandomState(seed)
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        # Build K4 and K5 constraint arrays
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

        # Build incidence: which triples affect which K4s/K5s
        # Use sparse-like representation with precomputed lists
        affected_k4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k4s):
            for t in ts:
                affected_k4[t].append(i)

        affected_k5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k5s):
            for t in ts:
                affected_k5[t].append(i)

        # Convert to numpy arrays for fast access
        k4_arr = np.array(k4s, dtype=np.int32)
        k5_arr = np.array(k5s, dtype=np.int32)

        # Initialize with bias
        colors = (rng.rand(num_triples) < 0.5).astype(np.int8)

        # Compute initial violation counts
        k4_red = np.sum(colors[k4_arr] == 0, axis=1)
        k5_blue = np.sum(colors[k5_arr] == 1, axis=1)
        E = int(np.sum(k4_red == 4)) + int(np.sum(k5_blue == 10))

        if E == 0:
            return colors.tolist()

        # Vectorized local search: try flipping a batch of random positions
        T = 1.0
        cooling = 0.9999
        min_T = 0.005
        batch_size = 200  # try 200 random flips per "step"

        steps = 0
        while time.time() < start_time + time_limit:
            steps += 1
            if T < min_T:
                break

            # Pick batch of random triples to flip
            batch = rng.randint(0, num_triples, size=batch_size)
            old_vals = colors[batch].copy()
            new_vals = 1 - old_vals
            colors[batch] = new_vals

            # Recompute violations (vectorized)
            new_k4_red = np.sum(colors[k4_arr] == 0, axis=1)
            new_k5_blue = np.sum(colors[k5_arr] == 1, axis=1)
            new_E = int(np.sum(new_k4_red == 4)) + int(np.sum(new_k5_blue == 10))

            delta = new_E - E
            if delta <= 0:
                E = new_E
                k4_red = new_k4_red
                k5_blue = new_k5_blue
            else:
                # Accept with probability exp(-delta/T)
                if rng.random() < np.exp(-delta / T):
                    E = new_E
                    k4_red = new_k4_red
                    k5_blue = new_k5_blue
                else:
                    # Revert
                    colors[batch] = old_vals

            if E == 0:
                return colors.tolist()

            T *= cooling

            # Periodic kicks: randomize a larger fraction
            if steps % 50 == 0 and E > 0:
                kick_frac = rng.randint(50, 200)
                kick_idx = rng.randint(0, num_triples, size=kick_frac)
                colors[kick_idx] = rng.randint(0, 2, size=kick_frac).astype(np.int8)
                k4_red = np.sum(colors[k4_arr] == 0, axis=1)
                k5_blue = np.sum(colors[k5_arr] == 1, axis=1)
                E = int(np.sum(k4_red == 4)) + int(np.sum(k5_blue == 10))
                T = max(T, 0.5)

        return None

    # Strategy 1: Try cyclic/coloring constructions (fast, deterministic)
    for n_try in [34, 33, 32]:
        if time.time() > deadline - 30:
            break
        # Try several cyclic variants
        found = False
        for shift in range(0, n_try, max(1, n_try // 10)):
            if time.time() > deadline - 25:
                break
            # Different coloring rules
            for rule in range(3):
                triples = list(combinations(range(n_try), 3))
                colors = []
                for (a, b, c) in triples:
                    s = (a + b + c) % n_try
                    if rule == 0:
                        colors.append(0 if s < n_try // 2 else 1)
                    elif rule == 1:
                        colors.append(0 if s < n_try // 3 else 1)
                    else:
                        colors.append(0 if s % 2 == 0 else 1)
                # Quick check: count red K4s and blue K5s
                triple_to_idx = {t: i for i, t in enumerate(triples)}
                valid = True
                for k4 in combinations(range(n_try), 4):
                    all_red = True
                    for t in combinations(k4, 3):
                        if colors[triple_to_idx[t]] != 0:
                            all_red = False
                            break
                    if all_red:
                        valid = False
                        break
                if valid:
                    for k5 in combinations(range(n_try), 5):
                        all_blue = True
                        for t in combinations(k5, 3):
                            if colors[triple_to_idx[t]] != 1:
                                all_blue = False
                                break
                        if all_blue:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return {'n': n_try, 'colors': colors}
        if not found:
            continue

    # Strategy 2: Vectorized local search for n=34
    if time.time() < deadline - 20:
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 10:
                break
            result = vectorized_solve(34, time_limit=10, seed=seed)
            if result is not None:
                return {'n': 34, 'colors': result}

    # Strategy 3: Vectorized local search for n=33
    if time.time() < deadline - 10:
        result = vectorized_solve(33, time_limit=8, seed=42)
        if result is not None:
            return {'n': 33, 'colors': result}

    # Strategy 4: Fallback - small valid coloring
    # n=8: check if simple coloring works
    for n_fb in [8, 7, 6, 5]:
        triples = list(combinations(range(n_fb), 3))
        colors = [0 if (a + b + c) % 2 == 0 else 1 for (a, b, c) in triples]
        # Verify
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        valid = True
        for k4 in combinations(range(n_fb), 4):
            all_red = all(colors[triple_to_idx[t]] == 0 for t in combinations(k4, 3))
            if all_red:
                valid = False
                break
        if valid:
            for k5 in combinations(range(n_fb), 5):
                all_blue = all(colors[triple_to_idx[t]] == 1 for t in combinations(k5, 3))
                if all_blue:
                    valid = False
                    break
        if valid:
            return {'n': n_fb, 'colors': colors}

    # Absolute fallback: n=4 with all blue (no red K4 possible with 4 vertices if not all red)
    # Actually with 4 vertices, if all 4 triples are blue, no red K4. But need to check blue K5 - need 5 vertices.
    # So n=4 all blue is valid.
    n = 4
    triples = list(combinations(range(n), 3))
    colors = [1] * len(triples)
    return {'n': n, 'colors': colors}
