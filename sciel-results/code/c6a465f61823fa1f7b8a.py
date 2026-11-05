import time
import random

def construct():
    start = time.time()
    DEADLINE = 145.0  # safety margin vs 150s kill

    n = 20
    adj = [0] * n

    # ---- fast K3,3 check over triples touching a small set of vertices ----
    def check_local(changed):
        # changed: iterable of vertex indices; check every triple with at least one member
        changed_set = set(changed)
        for i in range(n):
            if i not in changed_set:
                continue
            for j in range(n):
                if j <= i:
                    continue
                if j not in changed_set and i not in changed_set:
                    continue
                for k in range(n):
                    if k <= j:
                        continue
                    if (k not in changed_set and j not in changed_set and i not in changed_set):
                        continue
                    common = adj[i] & adj[j] & adj[k]
                    mask = (1 << i) | (1 << j) | (1 << k)
                    valid = common & ~mask
                    if valid & (valid - 1) & (valid - 1):  # >=3 bits set
                        return False
        return True

    def check_all():
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    common = adj[i] & adj[j] & adj[k]
                    mask = (1 << i) | (1 << j) | (1 << k)
                    valid = common & ~mask
                    if valid & (valid - 1) & (valid - 1):
                        return False
        return True

    def edge_count():
        return sum(bin(a).count('1') for a in adj) // 2

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # ---- structured Paley start (quadratic-residue graph on 20 vertices) ----
    for i in range(n):
        for j in range(i + 1, n):
            if ((j - i) ** 2) % 20 in (1, 4, 9, 16):
                add_edge(i, j)

    # ---- clean up: remove edges that create a K3,3 ----
    guard = 0
    while not check_all():
        guard += 1
        if guard > 20000:
            break
        # find a violating triple and break one edge
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    common = adj[i] & adj[j] & adj[k]
                    mask = (1 << i) | (1 << j) | (1 << k)
                    valid = common & ~mask
                    if valid & (valid - 1) & (valid - 1):
                        v = (valid & -valid).bit_length() - 1
                        # pick a random vertex among i,j,k to detach from v
                        u = random.choice((i, j, k))
                        remove_edge(u, v)
                        break
                else:
                    continue
                break
            else:
                continue
            break

    # ---- best-improvement local search: add edges ----
    rng = random.Random(12345)
    while time.time() < DEADLINE:
        improved = False
        candidates = [(u, v) for u in range(n) for v in range(u + 1, n)
                      if not ((adj[u] >> v) & 1)]
        rng.shuffle(candidates)
        for (u, v) in candidates:
            if time.time() > DEADLINE:
                break
            add_edge(u, v)
            if check_local((u, v)):
                improved = True
            else:
                remove_edge(u, v)
        if not improved:
            break

    # ---- best-improvement local search: edge swaps ----
    while time.time() < DEADLINE:
        improved = False
        # list of present and absent edges
        present = [(u, v) for u in range(n) for v in range(u + 1, n)
                   if (adj[u] >> v) & 1]
        absent = [(u, v) for u in range(n) for v in range(u + 1, n)
                  if not ((adj[u] >> v) & 1)]
        rng.shuffle(present)
        rng.shuffle(absent)
        for (a, b) in present:
            if time.time() > DEADLINE:
                break
            for (c, d) in absent:
                # avoid creating duplicate / self / reversing same edge
                if (a == c and b == d) or (a == d and b == c):
                    continue
                # perform swap
                remove_edge(a, b)
                add_edge(c, d)
                if check_local((a, b, c, d)):
                    improved = True
                else:
                    remove_edge(c, d)
                    add_edge(a, b)
        if not improved:
            break

    # ---- final safety: ensure validity ----
    if not check_all():
        g = 0
        while not check_all() and g < 20000:
            g += 1
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        common = adj[i] & adj[j] & adj[k]
                        mask = (1 << i) | (1 << j) | (1 << k)
                        valid = common & ~mask
                        if valid & (valid - 1) & (valid - 1):
                            v = (valid & -valid).bit_length() - 1
                            u = random.choice((i, j, k))
                            remove_edge(u, v)
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges
