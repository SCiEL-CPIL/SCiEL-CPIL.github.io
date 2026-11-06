import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before the 90s kill

    n = 256
    max_sum = 2 * n  # 512

    # Perfect squares that can equal a+b for 1<=a<=b<=256
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1

    # Diagonal constraint: 2a must not be a perfect square
    # 2a is a square only when a is 2*k^2 (since 2a=2^2*k^2 => a=2*k^2)
    bad_diag = set()
    k = 1
    while 2 * k * k <= n:
        bad_diag.add(2 * k * k)
        k += 1

    candidates = [a for a in range(1, n + 1) if a not in bad_diag]
    cand_set = set(candidates)
    m = len(candidates)

    # index map
    idx_of = {a: i for i, a in enumerate(candidates)}

    # Precompute conflict bitmasks per candidate
    conflict = [0] * m
    for i, a in enumerate(candidates):
        mask = 0
        for j in range(i + 1, m):
            b = candidates[j]
            if (a + b) in squares:
                mask |= (1 << j)
        conflict[i] = mask

    def mask_of(sol):
        mask = 0
        for a in sol:
            mask |= (1 << idx_of[a])
        return mask

    def greedy_randomized(candidates, conflict, m, rng):
        # pool as list of indices
        pool = list(range(m))
        # available mask
        avail = (1 << m) - 1
        selected = []
        while pool:
            # compute degrees
            best_deg = None
            best_nodes = []
            for node in pool:
                deg = bin(conflict[node] & avail).count("1")
                if best_deg is None or deg < best_deg:
                    best_deg = deg
                    best_nodes = [node]
                elif deg == best_deg:
                    best_nodes.append(node)
            node = rng.choice(best_nodes)
            selected.append(candidates[node])
            avail &= ~(1 << node)
            # remove node and its neighbors
            avail &= ~conflict[node]
            # rebuild pool from avail
            pool = [i for i in pool if (avail >> i) & 1]
        return selected

    def try_add(sol):
        sol_set = set(sol)
        sol_mask = mask_of(sol)
        improved = list(sol)
        changed = True
        while changed:
            changed = False
            for i, a in enumerate(candidates):
                if (sol_mask >> i) & 1:
                    continue
                if (conflict[i] & sol_mask) == 0:
                    improved.append(a)
                    sol_mask |= (1 << i)
                    changed = True
        return improved

    def local_search(sol, max_iters=500):
        sol_set = set(sol)
        sol_mask = mask_of(sol)
        best = list(sol)
        for _ in range(max_iters):
            if time.time() - start_time > deadline - 4.0:
                break
            improved = False
            # try to add
            for i, a in enumerate(candidates):
                if (sol_mask >> i) & 1:
                    continue
                if (conflict[i] & sol_mask) == 0:
                    sol_mask |= (1 << i)
                    sol_set.add(a)
                    best = list(sol_set)
                    improved = True
                    break
            if improved:
                continue
            # try swap: remove one, then try to add two (net +1)
            # For each candidate to remove, see if we can add at least 2
            for i, a in enumerate(candidates):
                if not ((sol_mask >> i) & 1):
                    continue
                # remove a
                new_mask = sol_mask & ~(1 << i)
                # count how many can be added
                add_count = 0
                added = []
                for j, b in enumerate(candidates):
                    if (new_mask >> j) & 1:
                        continue
                    if (conflict[j] & new_mask) == 0:
                        add_count += 1
                        added.append(j)
                        if add_count >= 2:
                            break
                if add_count >= 2:
                    # perform the swap: remove a, add the first two
                    sol_mask = new_mask
                    sol_set.discard(a)
                    for j in added[:2]:
                        sol_mask |= (1 << j)
                        sol_set.add(candidates[j])
                    best = list(sol_set)
                    improved = True
                    break
            if not improved:
                break
        return best

    # Initial deterministic greedy
    rng = random.Random(0)
    best = greedy_randomized(candidates, conflict, m, rng)
    best = try_add(best)
    best = local_search(best, max_iters=200)

    best_len = len(best)

    # Randomized restarts
    restart = 0
    while time.time() - start_time < deadline - 6.0:
        restart += 1
        rng = random.Random(restart)
        sol = greedy_randomized(candidates, conflict, m, rng)
        sol = try_add(sol)
        if len(sol) > best_len:
            best = sol
            best_len = len(sol)
        sol = local_search(sol, max_iters=100)
        if len(sol) > best_len:
            best = sol
            best_len = len(sol)

    # Final local search on the best
    best = local_search(best, max_iters=200)

    # Final verification
    best_set = set(best)
    for a in best:
        if 2 * a in squares:
            return []
    for i, a in enumerate(best):
        for b in best[i + 1:]:
            if (a + b) in squares:
                return []

    return best
