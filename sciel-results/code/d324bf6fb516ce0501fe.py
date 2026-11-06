"""Fresh pebbling schedule: minimize loads (transfers).
Inputs start blue (in fast memory, consuming budget). Outputs must finish blue.
Uses LIFO stack pebbling with recomputation in a top-down DFS order.
"""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s kill

    # ---- Parse dag structure ----
    if hasattr(dag, "parents"):
        parents = [list(p) for p in dag.parents]
        outputs = list(dag.outputs)
    elif isinstance(dag, dict):
        parents = [list(p) for p in dag.get("parents", dag.get("edges", []))]
        outputs = list(dag.get("outputs", dag.get("output", [])))
    else:
        parents = [list(p) for p in dag[0]]
        outputs = list(dag[1])

    n = len(parents)
    if n == 0:
        return []

    children = [[] for _ in range(n)]
    for v, ps in enumerate(parents):
        for p in ps:
            children[p].append(v)

    is_input = [len(p) == 0 for p in parents]

    # ---- Fast memory as LIFO stack ----
    stack = []
    in_fast = set()
    ops = []
    loads = 0

    def evict_if_full():
        while len(stack) >= fast_memory_budget and stack:
            ev = stack.pop()
            in_fast.discard(ev)

    def load(v):
        nonlocal loads
        if v in in_fast:
            return
        evict_if_full()
        stack.append(v)
        in_fast.add(v)
        ops.append(("load", v))
        loads += 1

    def compute(v):
        if v in in_fast:
            ops.append(("compute", v))
            return
        evict_if_full()
        stack.append(v)
        in_fast.add(v)
        ops.append(("compute", v))

    # Initialize: inputs start blue (in fast memory).
    for v in range(n):
        if is_input[v]:
            evict_if_full()
            stack.append(v)
            in_fast.add(v)
            # no operation for initial blue inputs (they start there)

    # ---- Top-down DFS order from outputs ----
    order = []
    visited = [False] * n
    def dfs(v):
        if visited[v]:
            return
        visited[v] = True
        order.append(v)
        for c in children[v]:
            dfs(c)
    for out in outputs:
        dfs(out)
    for v in range(n):
        if not visited[v]:
            dfs(v)

    computed = [False] * n
    for v in range(n):
        if is_input[v]:
            computed[v] = True

    for v in order:
        if time.time() - t0 > DEADLINE:
            break
        if computed[v]:
            continue
        for p in parents[v]:
            load(p)
        compute(v)
        computed[v] = True

    # Ensure outputs finish blue.
    for out in outputs:
        if time.time() - t0 > DEADLINE:
            break
        load(out)

    # Fallback for completeness if deadline hit early.
    if not all(computed):
        stack = []
        in_fast = set()
        ops = []
        computed = [False] * n
        for v in range(n):
            if is_input[v]:
                evict_if_full()
                stack.append(v)
                in_fast.add(v)
                computed[v] = True
                continue
            for p in parents[v]:
                load(p)
            compute(v)
            computed[v] = True
        for out in outputs:
            load(out)

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops
# EVOLVE-BLOCK-END
