import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 320  # Safety margin before 330s kill

    N = 64
    M = N * N

    # Precompute all points
    points = [(x, y) for x in range(N) for y in range(N)]
    coords = np.array(points, dtype=np.int32)

    # Precompute squared distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sum(diff**2, axis=-1)

    def try_add(S_set, dist_to_S, p):
        if p in S_set:
            return False
        seen_dists = set()
        d_vals = []
        for q in S_set:
            d = dist[p, q]
            if d in seen_dists:
                return False
            seen_dists.add(d)
            d_vals.append((q, d))
        for q, d in d_vals:
            if d in dist_to_S[q]:
                return False
        return True

    def add_point(S_set, dist_to_S, p):
        S_set.add(p)
        dist_to_S[p] = set()
        for q in S_set:
            if q == p:
                continue
            d = dist[p, q]
            dist_to_S[p].add(d)
            dist_to_S[q].add(d)

    def remove_point(S_set, dist_to_S, p):
        if p not in S_set:
            return
        for q in S_set:
            if q == p:
                continue
            d = dist[p, q]
            dist_to_S[q].discard(d)
        dist_to_S.pop(p, None)
        S_set.discard(p)

    def build_from_seed(seed_indices):
        """Build a valid set by greedily adding points from a seed ordering."""
        S_set = set()
        dist_to_S = {}
        for p in seed_indices:
            if p < 0 or p >= M:
                continue
            if try_add(S_set, dist_to_S, p):
                add_point(S_set, dist_to_S, p)
        return S_set, dist_to_S

    def golomb_ruler_seed(scale=1, offset_x=0, offset_y=0):
        """
        Generate a 1D Golomb-ruler-like seed placed along a line.
        Use a known good Golomb ruler pattern and scale it.
        """
        # A 13-mark Golomb ruler: 0,1,4,10,12,17,20,22,27,28,34,37,40
        # We'll use a smaller one and scale to fit in N
        base_ruler = [0, 1, 4, 10, 12, 17, 20, 22, 27, 28, 34, 37, 40]
        indices = []
        for mark in base_ruler:
            x = offset_x + mark * scale
            y = offset_y
            if 0 <= x < N and 0 <= y < N:
                idx = y * N + x
                indices.append(idx)
        return indices

    def sidon_2d_seed():
        """
        Generate a 2D Sidon-like set using a known construction.
        Use pairs (i, j) where i and j are from a Golomb ruler,
        and ensure all pairwise squared distances are unique.
        """
        base_ruler = [0, 1, 4, 10, 12, 17, 20, 22, 27, 28, 34, 37, 40]
        # Scale to fit in N=64
        scale = N // (base_ruler[-1] + 1)
        if scale < 1:
            scale = 1
        indices = []
        for i_mark in base_ruler:
            for j_mark in base_ruler:
                x = i_mark * scale
                y = j_mark * scale
                if 0 <= x < N and 0 <= y < N:
                    idx = y * N + x
                    indices.append(idx)
        return indices

    def hexagonal_lattice_seed():
        """
        Generate points on a hexagonal-like lattice pattern.
        Hexagonal lattices have good distance-uniqueness properties.
        """
        indices = []
        # Hexagonal lattice: rows offset by half the spacing
        spacing = 3  # minimum spacing to avoid small isosceles triangles
        for row in range(N):
            for col in range(N):
                if (row // spacing) % 2 == 0:
                    x = col
                else:
                    x = col + spacing // 2
                y = row
                if 0 <= x < N and 0 <= y < N:
                    # Only take every few points to increase density control
                    if (col % spacing == 0):
                        idx = y * N + x
                        indices.append(idx)
        return indices

    def diagonal_stripe_seed():
        """
        Generate points along diagonal stripes with Golomb-ruler spacing.
        """
        indices = []
        base_ruler = [0, 1, 4, 10, 12, 17, 20, 22, 27, 28, 34, 37, 40]
        scale = 2
        for d in range(N):
            # Diagonal d: points (x, y) where x - y = d
            for mark in base_ruler:
                x = mark * scale
                y = x - d
                if 0 <= x < N and 0 <= y < N:
                    idx = y * N + x
                    indices.append(idx)
        return indices

    def random_seed():
        """Pure random ordering."""
        indices = list(range(M))
        random.shuffle(indices)
        return indices

    def local_search(S_set, dist_to_S):
        S_set = set(S_set)
        dist_to_S = {k: set(v) for k, v in dist_to_S.items()}

        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S_list = list(S_set)
            random.shuffle(S_list)

            # Try adding points not in S
            candidates = [i for i in range(M) if i not in S_set]
            random.shuffle(candidates)

            for p in candidates:
                if time.time() > deadline - 10:
                    break
                if try_add(S_set, dist_to_S, p):
                    add_point(S_set, dist_to_S, p)
                    improved = True
                    break

            if improved:
                continue

            # If no addition, try a swap
            if S_list:
                swaps_tried = 0
                max_swaps = 80
                while swaps_tried < max_swaps and time.time() < deadline - 10:
                    q_to_remove = random.choice(S_list)
                    remove_point(S_set, dist_to_S, q_to_remove)

                    candidates = [i for i in range(M) if i not in S_set]
                    random.shuffle(candidates)

                    added = False
                    for p in candidates:
                        if time.time() > deadline - 10:
                            break
                        if p == q_to_remove:
                            continue
                        if try_add(S_set, dist_to_S, p):
                            add_point(S_set, dist_to_S, p)
                            added = True
                            break

                    if not added:
                        add_point(S_set, dist_to_S, q_to_remove)
                        swaps_tried += 1
                    else:
                        improved = True
                        break
                if not improved:
                    break
            else:
                break

        return S_set, dist_to_S

    def two_swap_local_search(S_set, dist_to_S):
        """Enhanced local search with 2-swap moves."""
        S_set = set(S_set)
        dist_to_S = {k: set(v) for k, v in dist_to_S.items()}

        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            S_list = list(S_set)
            random.shuffle(S_list)

            # Try 1-swap first
            candidates = [i for i in range(M) if i not in S_set]
            random.shuffle(candidates)
            for p in candidates:
                if time.time() > deadline - 15:
                    break
                if try_add(S_set, dist_to_S, p):
                    add_point(S_set, dist_to_S, p)
                    improved = True
                    break

            if improved:
                continue

            # Try 2-swap: remove 2, add 2
            if len(S_list) >= 2:
                swaps_tried = 0
                max_swaps = 40
                while swaps_tried < max_swaps and time.time() < deadline - 15:
                    q1, q2 = random.sample(S_list, 2)
                    remove_point(S_set, dist_to_S, q1)
                    remove_point(S_set, dist_to_S, q2)

                    candidates = [i for i in range(M) if i not in S_set]
                    random.shuffle(candidates)

                    added_count = 0
                    added_points = []
                    for p in candidates:
                        if time.time() > deadline - 15:
                            break
                        if p in (q1, q2):
                            continue
                        if try_add(S_set, dist_to_S, p):
                            add_point(S_set, dist_to_S, p)
                            added_points.append(p)
                            added_count += 1
                            if added_count >= 2:
                                break

                    if added_count < 2:
                        # Revert
                        for q in added_points:
                            remove_point(S_set, dist_to_S, q)
                        add_point(S_set, dist_to_S, q1)
                        add_point(S_set, dist_to_S, q2)
                        swaps_tried += 1
                    else:
                        improved = True
                        break
                if not improved:
                    break
            else:
                break

        return S_set, dist_to_S

    best_S_set = set()
    best_dist_to_S = {}

    # Generate multiple seed strategies
    seed_generators = [
        ("golomb_h", lambda: golomb_ruler_seed(scale=1, offset_x=0, offset_y=0)),
        ("golomb_v", lambda: golomb_ruler_seed(scale=1, offset_x=0, offset_y=N // 2)),
        ("golomb_d1", lambda: diagonal_stripe_seed()),
        ("sidon_2d", lambda: sidon_2d_seed()),
        ("hex", lambda: hexagonal_lattice_seed()),
        ("random1", lambda: random_seed()),
        ("random2", lambda: random_seed()),
        ("random3", lambda: random_seed()),
    ]

    # Also generate shifted/scaled variants of Golomb rulers
    for scale in [1, 2, 3]:
        for ox in [0, 5, 10]:
            for oy in [0, 5, 10]:
                seed_generators.append((f"g{scale}_{ox}_{oy}",
                                         lambda s=scale, x=ox, y=oy: golomb_ruler_seed(scale=s, offset_x=x, offset_y=y)))

    # Shuffle the order of seed generators for variety
    random.shuffle(seed_generators)

    num_restarts = len(seed_generators) + 10  # Extra random restarts
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break

        if i < len(seed_generators):
            name, gen_fn = seed_generators[i]
            seed_indices = gen_fn()
            # Shuffle within the seed to add some randomness
            random.shuffle(seed_indices)
            # Append random tail
            remaining = [p for p in range(M) if p not in set(seed_indices)]
            random.shuffle(remaining)
            seed_indices = seed_indices + remaining
        else:
            seed_indices = random_seed()

        S_set, dist_to_S = build_from_seed(seed_indices)

        # Apply local search
        S_set, dist_to_S = local_search(S_set, dist_to_S)

        if len(S_set) > len(best_S_set):
            best_S_set = S_set.copy()
            best_dist_to_S = {k: set(v) for k, v in dist_to_S.items()}

    # Final polish with 2-swap local search on the best set
    if time.time() < deadline - 20 and best_S_set:
        best_S_set, best_dist_to_S = two_swap_local_search(best_S_set, best_dist_to_S)

    # One more round of 1-swap polish
    if time.time() < deadline - 5 and best_S_set:
        best_S_set, best_dist_to_S = local_search(best_S_set, best_dist_to_S)

    return [points[i] for i in best_S_set]


# For testing
if __name__ == "__main__":
    res = construct()
    print(len(res))
