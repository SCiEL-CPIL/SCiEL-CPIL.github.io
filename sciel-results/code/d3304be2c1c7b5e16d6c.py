import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin before 60s

    N = 256

    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency masks
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
        mask = local_improve(mask)
        best_size = get_size(mask)
        best_mask = mask

        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations and (time.time() - start_time < DEADLINE - 3):
            improved = False
            iter_count += 1

            verts_in_set = [v for v in range(N) if (mask >> v) & 1]
            random.shuffle(verts_in_set)

            # 1-removal
            for v in verts_in_set:
                if time.time() - start_time > DEADLINE - 3:
                    return best_mask
                new_mask = mask & ~(1 << v)
                new_mask = local_improve(new_mask)
                new_size = get_size(new_mask)
                if new_size > best_size:
                    best_size = new_size
                    best_mask = new_mask
                    mask = new_mask
                    improved = True
                    break

            if improved:
                continue

            # 2-removal
            if len(verts_in_set) >= 2:
                random.shuffle(verts_in_set)
                pairs_tried = 0
                max_pairs = 40
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

    def simulated_annealing(mask, initial_temp=8.0, cooling=0.999, max_steps=3000):
        mask = local_improve(mask)
        best_mask = mask
        best_size = get_size(mask)
        current_size = best_size
        temp = initial_temp

        for step in range(max_steps):
            if time.time() - start_time > DEADLINE - 3:
                break

            verts_in_set = [v for v in range(N) if (mask >> v) & 1]
            verts_not_in_set = [v for v in candidates if not ((mask >> v) & 1)]

            if random.random() < 0.5 and verts_not_in_set:
                # Try to add a vertex
                v = random.choice(verts_not_in_set)
                conflicts = adj_mask[v] & mask
                if conflicts == 0:
                    new_mask = mask | (1 << v)
                    new_size = get_size(new_mask)
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = new_mask
                    mask = new_mask
                    current_size = new_size
                else:
                    # Remove conflicting vertices and add v
                    new_mask = mask & ~conflicts | (1 << v)
                    new_size = get_size(new_mask)
                    delta = new_size - current_size
                    if delta >= 0 or random.random() < math.exp(delta / temp):
                        mask = new_mask
                        current_size = new_size
                        if new_size > best_size:
                            best_size = new_size
                            best_mask = new_mask
            elif verts_in_set:
                # Try to remove a vertex
                v = random.choice(verts_in_set)
                new_mask = mask & ~(1 << v)
                new_size = get_size(new_mask)
                delta = new_size - current_size
                if delta >= 0 or random.random() < math.exp(delta / temp):
                    mask = new_mask
                    current_size = new_size

            temp *= cooling

        # Final improvement
        final_mask = local_improve(mask)
        if get_size(final_mask) > best_size:
            best_mask = final_mask
            best_size = get_size(final_mask)

        return best_mask

    best_size = 0
    best_mask = 0

    # Diverse initialization strategies
    strategies = []
    strategies.append(sorted(candidates, key=lambda v: degrees[v]))
    strategies.append(sorted(candidates, key=lambda v: -degrees[v]))
    for r in range(4):
        class_verts = [v for v in candidates if (v + 1) % 4 == r]
        strategies.append(sorted(class_verts, key=lambda v: degrees[v]))
        strategies.append(sorted(class_verts, key=lambda v: -degrees[v]))
    for _ in range(3):
        strategies.append(candidates[:])

    # Initial greedy solutions
    for i in range(20):
        if time.time() - start_time > DEADLINE - 8:
            break
        strat_idx = i % len(strategies)
        order = strategies[strat_idx][:]
        if i >= 5:
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

    # Main loop: Iterated Local Search + Simulated Annealing
    while time.time() - start_time < DEADLINE - 1:

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

        if time.time() - start_time < DEADLINE - 4:
            s_mask = local_search(s_mask, max_iterations=50)

        if time.time() - start_time < DEADLINE - 2:
            s_mask = simulated_annealing(s_mask, initial_temp=5.0, cooling=0.9995, max_steps=2000)
            s_mask = local_improve(s_mask)

        if get_size(s_mask) > best_size:
            best_size = get_size(s_mask)
            best_mask = s_mask

    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)

    return sorted(result)
