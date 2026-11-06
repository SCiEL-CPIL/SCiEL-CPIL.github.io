"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 9
SIZE = 1 << N  # 512

NEIGHBORS = []
for v in range(SIZE):
    NEIGHBORS.append([v ^ (1 << b) for b in range(N)])

# EVOLVE-BLOCK-START
def _deadline_ok(deadline, margin=2.0):
    return time.time() < deadline - margin

def _warnsdorff_search(seed_path, deadline, max_nodes=300_000, seed=None):
    """Backtracking with Warnsdorff heuristic from a seed path."""
    if seed is not None:
        random.seed(seed)

    best_path = list(seed_path)
    best_len = len(best_path)
    used = set(seed_path)
    path = list(seed_path)
    nodes = 0

    def dfs(v):
        nonlocal best_path, best_len, nodes
        nodes += 1
        if nodes > max_nodes or not _deadline_ok(deadline):
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)

        opts = []
        for nb in NEIGHBORS[v]:
            if nb in used:
                continue
            ok = True
            for p in path[:-1]:
                if (nb ^ p).bit_count() == 1:
                    ok = False
                    break
            if not ok:
                continue
            cnt = 0
            for nb2 in NEIGHBORS[nb]:
                if nb2 not in used and nb2 != v:
                    cnt += 1
            opts.append((cnt, nb))

        if not opts:
            return
        opts.sort(key=lambda x: x[0])
        for _, nb in opts:
            path.append(nb)
            used.add(nb)
            dfs(nb)
            path.pop()
            used.discard(nb)
            if nodes > max_nodes or not _deadline_ok(deadline):
                return

    dfs(seed_path[-1])
    return best_path

def _try_splice(a, b):
    set_a = set(a)
    set_b = set(b)
    if set_a & set_b:
        return None
    candidates = []
    for a_list in (a, a[::-1]):
        for b_list in (b, b[::-1]):
            if (a_list[-1] ^ b_list[0]).bit_count() == 1:
                merged = a_list + b_list
                ok = True
                for i in range(len(merged)):
                    for j in range(i + 2, len(merged)):
                        if (merged[i] ^ merged[j]).bit_count() == 1:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    candidates.append(merged)
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    return candidates[0]

def construct():
    import time
    import random
    start_time = time.time()
    effective_deadline = start_time + 145.0

    # Greedy seed
    path, used = [0], {0}
    while True:
        cur, nxt = path[-1], None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                continue
            nxt = v
            break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)

    best = list(path)
    best_len = len(best)
    rng = random.Random(42)

    attempt = 0
    while _deadline_ok(effective_deadline, margin=5.0):
        attempt += 1
        seed_path = list(best)
        seed_used = set(seed_path)
        for _ in range(rng.randint(0, 3)):
            cur = seed_path[-1]
            opts = []
            for nb in NEIGHBORS[cur]:
                if nb in seed_used:
                    continue
                ok = True
                for p in seed_path[:-1]:
                    if (nb ^ p).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    opts.append(nb)
            if not opts:
                break
            nb = rng.choice(opts)
            seed_path.append(nb)
            seed_used.add(nb)

        res = _warnsdorff_search(seed_path, effective_deadline,
                                  max_nodes=200_000, seed=attempt)
        if len(res) > best_len:
            best = res
            best_len = len(res)

        if _deadline_ok(effective_deadline, margin=10.0):
            start_v = rng.randrange(SIZE)
            path2, used2 = [start_v], {start_v}
            for _ in range(15):
                cur = path2[-1]
                opts = []
                for nb in NEIGHBORS[cur]:
                    if nb in used2:
                        continue
                    ok = True
                    for p in path2[:-1]:
                        if (nb ^ p).bit_count() == 1:
                            ok = False
                            break
                    if ok:
                        opts.append(nb)
                if not opts:
                    break
                nb = rng.choice(opts)
                path2.append(nb)
                used2.add(nb)
            res2 = _warnsdorff_search(path2, effective_deadline,
                                       max_nodes=150_000, seed=attempt + 1000)
            merged = _try_splice(best, res2)
            if merged and len(merged) > best_len:
                best = merged
                best_len = len(merged)

    return best
# EVOLVE-BLOCK-END
