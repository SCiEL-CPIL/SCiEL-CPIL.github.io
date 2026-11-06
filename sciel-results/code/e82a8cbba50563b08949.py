"""Pebbling solver: recompute-bounded pebbling to minimize transfers."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(100000)
from collections import deque

def pebble(dag, fast_memory_budget):
    n = dag['n']
    parents = dag['parents']
    outputs = list(dag['outputs'])
    B = max(1, int(fast_memory_budget))

    is_input = [len(parents[i]) == 0 for i in range(n)]
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)

    # topological order
    indeg = [len(parents[v]) for v in range(n)]
    q = deque([v for v in range(n) if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for c in children[u]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(topo) != n:
        topo = list(range(n))
    pos = {v: i for i, v in enumerate(topo)}

    # needed set
    needed = [False] * n
    for o in outputs:
        needed[o] = True
    changed = True
    while changed:
        changed = False
        for v in range(n):
            if needed[v]:
                for p in parents[v]:
                    if not needed[p]:
                        needed[p] = True
                        changed = True

    # distance to output for deletion heuristic
    dist = [10**9] * n
    for o in outputs:
        dist[o] = 0
    for v in reversed(topo):
        if dist[v] < 10**9:
            continue
        md = 10**9
        for c in children[v]:
            if dist[c] < md:
                md = dist[c]
        if md < 10**9:
            dist[v] = md + 1

    ops = []
    mem = set()
    outputs_set = set(outputs)
    pending = set(outputs)

    def make_space(k):
        while len(mem) > B - k:
            best = None
            best_key = None
            for u in mem:
                if u in pending:
                    continue
                d = dist[u] if dist[u] < 10**9 else 10**9
                key = (d, 0 if u not in outputs_set else 1, pos.get(u, n))
                if best_key is None or key < best_key:
                    best_key = key
                    best = u
            if best is None:
                best = next(iter(mem))
            ops.append(['delete', best])
            mem.remove(best)

    def ensure(v):
        stack = [(v, 0)]
        while stack:
            u, pi = stack[-1]
            if u in mem:
                stack.pop()
                continue
            if is_input[u]:
                make_space(1)
                ops.append(['load', u])
                mem.add(u)
                stack.pop()
                continue
            ps = parents[u]
            if pi < len(ps):
                p = ps[pi]
                stack[-1] = (u, pi + 1)
                if p not in mem:
                    stack.append((p, 0))
            else:
                make_space(1)
                ops.append(['compute', u])
                mem.add(u)
                stack.pop()

    for o in outputs:
        ensure(o)
        pending.discard(o)

    for o in outputs:
        if o not in mem:
            ensure(o)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops
# EVOLVE-BLOCK-END
