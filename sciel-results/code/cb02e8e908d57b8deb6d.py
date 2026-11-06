"""DAG pebbling: minimize transfers (load+store) under fast-memory budget."""
import time
import heapq
from collections import deque

def _topo_order(parents, vertices):
    """Return topological order and children map."""
    indeg = {v: len(parents.get(v, [])) for v in vertices}
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children.get(u, []):
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(topo) != len(vertices):
        topo = list(vertices)
    return topo, children

def _last_use(parents, children, topo):
    """For each vertex, find its last consumer in topological order (index)."""
    n = len(topo)
    idx = {v: i for i, v in enumerate(topo)}
    last = {v: -1 for v in topo}
    for v in topo:
        for c in children.get(v, []):
            if idx[c] > last[v]:
                last[v] = idx[c]
    return last

def _pebble_bottom_up(dag, fast_memory_budget):
    """
    Bottom-up pebbling: process vertices in topological order, keep each
    intermediate in fast memory until its last consumer is processed, then
    delete it. Inputs are loaded once. Outputs are stored at the end.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    outset = set(outputs)
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else len(vertices))

    topo, children = _topo_order(parents, vertices)
    n = len(topo)
    last = _last_use(parents, children, topo)
    idx = {v: i for i, v in enumerate(topo)}

    ops = []
    in_mem = set()

    def ensure_space(needed=1):
        """Free space by deleting vertices whose last consumer has been processed."""
        freed = 0
        # Delete in reverse topological order (most recently used first)
        for v in reversed(topo):
            if freed >= needed:
                break
            if v in in_mem and v not in outset and idx[v] < len(topo) - 1:
                # Check if all children of v have been processed
                all_children_done = True
                for c in children.get(v, []):
                    if idx[c] > idx[v]:  # child is after v in topo
                        # The child hasn't been processed yet if we're at position v
                        all_children_done = False
                        break
                # Actually, we should delete v when we've passed its last consumer
                # Let's use a different approach: delete v when current position > last[v]
                pass
        # Better approach: delete any vertex v in in_mem where last[v] < current_pos
        # This is called at position pos
        return freed

    # Process in topological order
    for pos, v in enumerate(topo):
        ps = parents.get(v, [])

        # Ensure all parents are in memory
        for p in ps:
            if p not in in_mem:
                # Need to load p (it's an input)
                if len(in_mem) >= b:
                    # Free space: delete vertices whose last consumer is before current pos
                    deleted = False
                    for x in reversed(topo):
                        if x in in_mem and x not in outset and last[x] < pos:
                            ops.append(["delete", x])
                            in_mem.discard(x)
                            deleted = True
                            if len(in_mem) < b:
                                break
                    if len(in_mem) >= b:
                        # Force delete something (shouldn't happen with correct logic)
                        for x in list(in_mem):
                            if x not in outset:
                                ops.append(["delete", x])
                                in_mem.discard(x)
                                if len(in_mem) < b:
                                    break
                ops.append(["load", p])
                in_mem.add(p)

        # Compute v
        ops.append(["compute", v])
        in_mem.add(v)

        # If v is an output, store it
        if v in outset:
            ops.append(["store", v])
            in_mem.discard(v)

        # Delete parents that are no longer needed (last consumer was v)
        for p in ps:
            if p in in_mem and p not in outset and last[p] == pos:
                ops.append(["delete", p])
                in_mem.discard(p)

    # Store any remaining outputs that weren't stored
    for v in outputs:
        if v in in_mem:
            ops.append(["store", v])
            in_mem.discard(v)

    return ops[:20000]

def _pebble_astar(dag, fast_memory_budget, deadline):
    """A* search for small DAGs to find optimal or near-optimal schedule."""
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    n = len(vertices)
    if n == 0:
        return []
    if n > 14:
        return None

    idx = {v: i for i, v in enumerate(vertices)}
    out_mask = 0
    for o in outputs:
        out_mask |= (1 << idx[o])

    par_mask = [0] * n
    is_input = [False] * n
    for i, v in enumerate(vertices):
        ps = parents.get(v, [])
        if not ps:
            is_input[i] = True
        m = 0
        for p in ps:
            m |= (1 << idx[p])
        par_mask[i] = m

    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else n)

    def popcount(x):
        return x.bit_count() if hasattr(x, "bit_count") else bin(x).count("1")

    # Heuristic: minimum additional transfers needed
    def heuristic(mem_mask, produced):
        h = 0
        missing = out_mask & ~produced
        mm = missing
        while mm:
            lb = mm & (-mm)
            i = lb.bit_length() - 1
            h += 1  # store
            # Missing ancestors that are inputs
            anc = par_mask[i] & ~mem_mask
            h += popcount(anc)
            mm ^= lb
        return h

    start_state = (0, 0)
    counter = 0
    open_heap = [(heuristic(0, 0), 0, counter, (0, 0))]
    g_score = {start_state: 0}
    parent_op = {}
    best = None
    max_expansions = 30000

    expansions = 0
    while open_heap:
        if time.time() > deadline or expansions > max_expansions:
            break
        f, g, _, (mem, produced) = heapq.heappop(open_heap)
        expansions += 1

        if produced == out_mask:
            seq = []
            state = (mem, produced)
            while state != start_state:
                prev, op, v = parent_op[state]
                seq.append([op, v])
                state = prev
            seq.reverse()
            best = seq
            break

        if g > g_score.get(state, float("inf")):
            continue

        for i in range(n):
            bit = 1 << i
            v = vertices[i]

            # Load input
            if is_input[i] and not (mem & bit):
                if popcount(mem) < b:
                    new_mem = mem | bit
                    new_prod = produced
                    ng = g + 1
                    ns = (new_mem, new_prod)
                    if ng < g_score.get(ns, float("inf")):
                        g_score[ns] = ng
                        parent_op[ns] = ((mem, produced), "load", v)
                        counter += 1
                        heapq.heappush(open_heap, (ng + heuristic(new_mem, new_prod), ng, counter, ns))

            # Compute
            if (mem & par_mask[i]) == par_mask[i] and not (mem & bit):
                new_mem = mem | bit
                new_prod = produced | bit if (out_mask & bit) else produced
                ng = g
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "compute", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem, new_prod), ng, counter, ns))

            # Store output
            if (mem & bit) and (out_mask & bit) and not (produced & bit):
                new_mem = mem & ~bit
                new_prod = produced | bit
                ng = g + 1
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "store", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem, new_prod), ng, counter, ns))

            # Delete non-output
            if (mem & bit) and not (out_mask & bit):
                new_mem = mem & ~bit
                new_prod = produced
                ng = g
                ns = (new_mem, new_prod)
                if ng < g_score.get(ns, float("inf")):
                    g_score[ns] = ng
                    parent_op[ns] = ((mem, produced), "delete", v)
                    counter += 1
                    heapq.heappush(open_heap, (ng + heuristic(new_mem, new_prod), ng, counter, ns))

    return best

def _pebble_greedy(dag, fast_memory_budget):
    """
    Greedy pebbling: process outputs one by one in reverse topological order.
    For each output, recursively compute its dependencies, keeping values in
    memory until no longer needed.
    """
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    outset = set(outputs)
    b = max(1, int(fast_memory_budget) if fast_memory_budget is not None else len(vertices))

    topo, children = _topo_order(parents, vertices)
    last = _last_use(parents, children, topo)
    idx = {v: i for i, v in enumerate(topo)}

    ops = []
    in_mem = set()
    stored = set()

    def ensure(v):
        """Ensure v is in memory."""
        if v in in_mem:
            return
        ps = parents.get(v, [])
        if not ps:
            # Input
            if len(in_mem) >= b:
                # Free space
                for x in list(in_mem):
                    if x not in outset and x not in stored:
                        ops.append(["delete", x])
                        in_mem.discard(x)
                        if len(in_mem) < b:
                            break
            ops.append(["load", v])
            in_mem.add(v)
        else:
            for p in ps:
                ensure(p)
            ops.append(["compute", v])
            in_mem.add(v)

    # Process outputs in reverse topological order
    for v in reversed(topo):
        if v in outset and v not in stored:
            ensure(v)
            ops.append(["store", v])
            stored.add(v)
            in_mem.discard(v)
            # Clean up: delete vertices no longer needed
            for x in list(in_mem):
                if x not in outset and x not in stored:
                    # Check if x is needed by any remaining output
                    needed = False
                    for rem in outputs:
                        if rem in stored:
                            continue
                        # Check if x is ancestor of rem
                        stack = [rem]
                        visited = set()
                        while stack:
                            cur = stack.pop()
                            if cur in visited:
                                continue
                            visited.add(cur)
                            if cur == x:
                                needed = True
                                break
                            for pp in parents.get(cur, []):
                                if pp not in visited:
                                    stack.append(pp)
                        if needed:
                            break
                    if not needed:
                        ops.append(["delete", x])
                        in_mem.discard(x)

    return ops[:20000]

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 40
    candidates = []

    # Try A* for small DAGs
    try:
        r = _pebble_astar(dag, fast_memory_budget, deadline)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass

    # Try bottom-up pebbling
    try:
        r = _pebble_bottom_up(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass

    # Try greedy
    try:
        r = _pebble_greedy(dag, fast_memory_budget)
        if r is not None and len(r) <= 20000:
            candidates.append(r)
    except Exception:
        pass

    if not candidates:
        return []

    def transfers(ops):
        return sum(1 for op, v in ops if op in ("load", "store"))

    best = min(candidates, key=transfers)
    return best
