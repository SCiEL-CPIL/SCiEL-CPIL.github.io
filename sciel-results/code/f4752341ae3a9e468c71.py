import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 56.0  # Safety margin before 60s

    N = 256

    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency masks
    # adj_mask[i] is an integer bitmask where bit j is 1 if (i+1) + (j+1) is a square
    adj_mask = [0] * N
    forbidden = set()

    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden.add(i)
        for sq in squares:
            u = sq - v
            if 1 <= u <= N:
                j = u - 1
                if j != i:
                    adj_mask[i] |= (1 << j)

    candidates = [i for i in range(N) if i not in forbidden]

    degrees = [bin(adj_mask[i]).count('1') for i in range(N)]

    # Modular residue classes: squares mod 4 are {0,1}
    residue = [((i + 1) % 4) for i in range(N)]

    # Intra-residue conflict count (proxy for same-class density)
    intra_conflicts = [0] * N
    for i in range(N):
        if i in forbidden:
            continue
        m = adj_mask[i]
        cnt = 0
        # Count neighbors with same residue
        for j in range(N):
            if residue[j] == residue[i] and (m >> j) & 1:
                cnt += 1
        intra_conflicts[i] = cnt

    def get_size(mask):
        return bin(mask).count('1')

    def greedy_solve(order):
        current_mask = 0
        for v in order:
            if v in forbidden:
                continue
            if not (adj_mask[v] & current_mask):
                current_mask |= (1 << v)
        return current_mask

    def local_improve(mask):
        improved = True
        while improved:
            improved = False
            for v in candidates:
                if (mask >> v) & 1:
                    continue
                if not (adj_mask[v] & mask):
                    mask |= (1 << v)
                    improved = True
        return mask

    def local_search(mask, max_iterations=80):
        """1-removal and 2-removal swaps."""
        mask = local_improve(mask)
        best_size = get_size(mask)
        best_mask = mask
        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations and (time.time() - start_time < DEADLINE - 2):
            improved = False
            iter_count += 1

            verts_in_set = [v for v in range(N) if (mask >> v) & 1]
            random.shuffle(verts_in_set)

            # 1-Removal
            for v in verts_in_set:
                if time.time() - start_time > DEADLINE - 2:
                    return best_mask
                new_mask = mask & ~(1 << v)
                new_mask = local_improve(new_mask)
                if get_size(new_mask) > best_size:
                    best_size = get_size(new_mask)
                    best_mask = new_mask
                    mask = new_mask
                    improved = True
                    break

            if improved:
                continue

            # 2-Removal (limited pairs per iteration)
            verts_in_set = [v for v in range(N) if (mask >> v) & 1]
            random.shuffle(verts_in_set)
            pairs_tried = 0
            max_pairs = 80
            for i in range(len(verts_in_set)):
                if pairs_tried >= max_pairs:
                    break
                for j in range(i + 1, len(verts_in_set)):
                    if pairs_tried >= max_pairs:
                        break
                    pairs_tried += 1
                    v1 = verts_in_set[i]
                    v2 = verts_in_set[j]
                    new_mask = mask & ~(1 << v1) & ~(1 << v2)
                    new_mask = local_improve(new_mask)
                    if get_size(new_mask) > best_size:
                        best_size = get_size(new_mask)
                        best_mask = new_mask
                        mask = new_mask
                        improved = True
                        break
                if improved:
                    break

        return best_mask

    def final_polish(mask):
        """High-precision polish: more aggressive 1- and 2-swap with more iterations."""
        mask = local_improve(mask)
        best_size = get_size(mask)
        best_mask = mask
        for _ in range(30):
            if time.time() - start_time > DEADLINE - 0.5:
                break
            improved = False
            verts_in_set = [v for v in range(N) if (mask >> v) & 1]
            random.shuffle(verts_in_set)
            for v in verts_in_set:
                new_mask = mask & ~(1 << v)
                new_mask = local_improve(new_mask)
                if get_size(new_mask) > best_size:
                    best_size = get_size(new_mask)
                    best_mask = new_mask
                    mask = new_mask
                    improved = True
                    break
            if improved:
                continue
            # 2-swap
            verts_in_set = [v for v in range(N) if (mask >> v) & 1]
            random.shuffle(verts_in_set)
            for i in range(len(verts_in_set)):
                for j in range(i + 1, len(verts_in_set)):
                    v1 = verts_in_set[i]
                    v2 = verts_in_set[j]
                    new_mask = mask & ~(1 << v1) & ~(1 << v2)
                    new_mask = local_improve(new_mask)
                    if get_size(new_mask) > best_size:
                        best_size = get_size(new_mask)
                        best_mask = new_mask
                        mask = new_mask
                        improved = True
                        break
                if improved:
                    break
        return best_mask

    best_size = 0
    best_mask = 0

    # Modular Initialization Strategies
    strategies = []

    # Global Low Degree
    strategies.append(sorted(candidates, key=lambda v: (intra_conflicts[v], degrees[v])))
    # Global High Degree
    strategies.append(sorted(candidates, key=lambda v: (-intra_conflicts[v], -degrees[v])))

    # Modular Classes
    for r in range(4):
        class_verts = [v for v in candidates if (v + 1) % 4 == r]
        strategies.append(sorted(class_verts, key=lambda v: (intra_conflicts[v], degrees[v])))
        strategies.append(sorted(class_verts, key=lambda v: (-intra_conflicts[v], -degrees[v])))

    # Random
    for _ in range(4):
        strategies.append(candidates[:])

    for i in range(24):
        if time.time() - start_time > DEADLINE - 5:
            break
        strat_idx = i % len(strategies)
        order = strategies[strat_idx][:]
        if i >= 8:
            random.shuffle(order)
        else:
            if i > 0:
                for _ in range(3):
                    if len(order) > 1:
                        idx1 = random.randint(0, len(order) - 1)
                        idx2 = random.randint(0, len(order) - 1)
                        order[idx1], order[idx2] = order[idx2], order[idx1]

        s_mask = greedy_solve(order)
        s_mask = local_improve(s_mask)

        if get_size(s_mask) > best_size:
            best_size = get_size(s_mask)
            best_mask = s_mask

    # Main Loop: Iterated Local Search
    while time.time() - start_time < DEADLINE - 2:

        if random.random() < 0.5:
            order = candidates[:]
            random.shuffle(order)
            s_mask = greedy_solve(order)
            s_mask = local_improve(s_mask)
        else:
            s_mask = best_mask
            verts_in_set = [v for v in range(N) if (s_mask >> v) & 1]
            random.shuffle(verts_in_set)
            remove_count = max(1, len(verts_in_set) // 10)
            for v in verts_in_set[:remove_count]:
                s_mask &= ~(1 << v)
            s_mask = local_improve(s_mask)

        if time.time() - start_time < DEADLINE - 3:
            s_mask = local_search(s_mask, max_iterations=60)

        if get_size(s_mask) > best_size:
            best_size = get_size(s_mask)
            best_mask = s_mask

    # Final high-precision polish pass
    if time.time() - start_time < DEADLINE - 0.5:
        best_mask = final_polish(best_mask)
        best_size = get_size(best_mask)

    # Convert mask back to list of 1-based values
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)

    return sorted(result)

# EVOLVE-BLOCK-END
