"""Communication-minimizing DAG pebbling: layered/greedy schedules + A* over red states."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def _parse_dag(dag):
    if isinstance(dag, dict):
        parents = dag.get('parents', None)
        outputs = dag.get('outputs', None)
        inputs = dag.get('inputs', None)
        vertices = dag.get('vertices', None)
        n = dag.get('n', None)

        if parents is None:
            if isinstance(dag, list):
                parents = [list(p) for p in dag]
                n = len(dag)
                if outputs is None:
                    outputs = [n - 1] if n > 0 else []
                if inputs is None:
                    inputs = [i for i in range(n) if not parents[i]]
            else:
                return None, None, None, None, None
        else:
            if isinstance(parents, list):
                parents = [list(p) for p in parents]
                n = len(parents)
            elif isinstance(parents, dict):
                n = max(parents.keys()) + 1 if parents else 0
                parents = [list(parents.get(i, [])) for i in range(n)]

            if outputs is None:
                children = [[] for _ in range(n)]
                for v in range(n):
                    for p in parents[v]:
                        children[p].append(v)
                outputs = [v for v in range(n) if not children[v]]
            if inputs is None:
                inputs = [v for v in range(n) if not parents[v]]
        return parents, outputs, inputs, vertices, n

    elif isinstance(dag, list):
        parents = [list(p) if isinstance(p, (list, tuple)) else [] for p in dag]
        n = len(dag)
        outputs = [n - 1] if n > 0 else []
        inputs = [i for i in range(n) if not parents[i]]
        return parents, outputs, inputs, None, n

    return None, None, None, None, None


def _build_children(parents, n):
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    return children


def _compute_depth(parents, n):
    depth = [0] * n
    for v in range(n):
        if parents[v]:
            depth[v] = max(depth[p] for p in parents[v]) + 1
    return depth


def _validate_ops(ops, parents, outputs, fast_memory_budget):
    n = len(parents)
    in_fast = set()
    stored = set()
    for op, v in ops:
        if op == 'load':
            if v in in_fast:
                continue
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.discard(v)
            stored.add(v)
        elif op == 'compute':
            for p in parents[v]:
                if p not in in_fast:
                    return False
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.discard(v)
        else:
            return False
    for o in outputs:
        if o not in stored:
            return False
    return True


def _count_transfers(ops):
    return sum(1 for op, v in ops if op in ('load', 'store'))


def _layered_pebbling(parents, outputs, fast_memory_budget, deadline):
    n = len(parents)
    if n == 0:
        return []
    children = _build_children(parents, n)
    depth = _compute_depth(parents, n)
    max_depth = max(depth) if depth else 0
    layers = [[] for _ in range(max_depth + 1)]
    for v in range(n):
        layers[depth[v]].append(v)

    last_child_layer = [-1] * n
    for v in range(n):
        if children[v]:
            last_child_layer[v] = max(depth[c] for c in children[v])

    is_output = set(outputs)
    ops = []
    in_fast = set()
    stored = set()

    def ensure_space():
        if len(in_fast) < fast_memory_budget:
            return True
        for u in sorted(in_fast, key=lambda x: (last_child_layer[x], x)):
            if u not in is_output:
                ops.append(['delete', u])
                in_fast.discard(u)
                if len(in_fast) < fast_memory_budget:
                    return True
        for u in sorted(in_fast, key=lambda x: (last_child_layer[x], x)):
            ops.append(['delete', u])
            in_fast.discard(u)
            if len(in_fast) < fast_memory_budget:
                return True
        return len(in_fast) < fast_memory_budget

    def make_available(p):
        if p in in_fast:
            return
        if p in stored:
            ensure_space()
            if len(in_fast) < fast_memory_budget:
                ops.append(['load', p])
                in_fast.add(p)
            return
        # recompute p from its parents
        for gp in parents[p]:
            make_available(gp)
        ensure_space()
        if len(in_fast) < fast_memory_budget:
            ops.append(['compute', p])
            in_fast.add(p)

    for d in range(max_depth + 1):
        if time.time() > deadline:
            return None
        layer = layers[d]
        if not layer:
            continue
        if d == 0:
            for v in layer:
                make_available(v)
                if v in in_fast:
                    ops.append(['store', v])
                    in_fast.discard(v)
                    stored.add(v)
        else:
            for v in layer:
                for p in parents[v]:
                    make_available(p)
                ensure_space()
                if len(in_fast) < fast_memory_budget:
                    ops.append(['compute', v])
                    in_fast.add(v)
                if v in in_fast:
                    ops.append(['store', v])
                    in_fast.discard(v)
                    stored.add(v)
            for v in list(in_fast):
                if v not in is_output and last_child_layer[v] < d:
                    ops.append(['delete', v])
                    in_fast.discard(v)

    for o in outputs:
        if o in in_fast:
            ops.append(['store', o])
            in_fast.discard(o)
    return ops


def _greedy_pebbling(parents, outputs, fast_memory_budget, deadline):
    n = len(parents)
    if n == 0:
        return []
    children = _build_children(parents, n)
    is_output = set(outputs)

    last_use = [-1] * n
    for v in range(n):
        if children[v]:
            last_use[v] = max(children[v])

    ops = []
    in_fast = set()
    stored = set()

    def try_evict():
        if not in_fast:
            return False
        best, best_key = None, (float('inf'), float('inf'))
        for u in in_fast:
            if u not in is_output:
                key = (last_use[u], u)
                if key < best_key:
                    best_key, best = key, u
        if best is None and in_fast:
            best = next(iter(in_fast))
        if best is not None:
            ops.append(['delete', best])
            in_fast.discard(best)
            return True
        return False

    def ensure_space():
        while len(in_fast) >= fast_memory_budget:
            if not try_evict():
                return False
        return True

    def make_available(p):
        if p in in_fast:
            return
        if p in stored:
            if ensure_space():
                ops.append(['load', p])
                in_fast.add(p)
            return
        for gp in parents[p]:
            make_available(gp)
        if ensure_space():
            ops.append(['compute', p])
            in_fast.add(p)

    for v in range(n):
        if time.time() > deadline:
            return None
        for p in parents[v]:
            make_available(p)
        if parents[v]:
            if ensure_space():
                ops.append(['compute', v])
                in_fast.add(v)
        if v in is_output and v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)
            stored.add(v)
        for p in parents[v]:
            if p in in_fast and p not in is_output and last_use[p] == v:
                ops.append(['delete', p])
                in_fast.discard(p)

    for o in outputs:
        if o in in_fast:
            ops.append(['store', o])
            in_fast.discard(o)
    return ops


def _dijkstra_pebbling(parents, outputs, fast_memory_budget, deadline):
    n = len(parents)
    if n == 0:
        return []
    children = _build_children(parents, n)
    is_output = set(outputs)
    parent_sets = [frozenset(parents[v]) for v in range(n)]

    # output-relevant: can reach an output
    relevant = set(outputs)
    for v in range(n - 1, -1, -1):
        if any(c in relevant for c in children[v]):
            relevant.add(v)

    def heuristic(fast, slow):
        h = 0
        for o in outputs:
            if o not in slow:
                h += 1
        for v in fast:
            if v in relevant and v not in slow:
                h += 1
        return h

    initial_state = (frozenset(), frozenset(range(n)))
    best_g = {initial_state: 0}
    parent_map = {}
    h0 = heuristic(initial_state[0], initial_state[1])
    pq = [(h0, 0, initial_state)]
    max_states = 400000

    while pq:
        if time.time() > deadline:
            return None
        if len(best_g) > max_states:
            return None
        f, g, state = heapq.heappop(pq)
        if state in best_g and g > best_g[state]:
            continue
        fast, slow = state
        if all(o in slow for o in outputs):
            ops = []
            cur = state
            while cur in parent_map:
                prev, op, v = parent_map[cur]
                ops.append([op, v])
                cur = prev
            ops.reverse()
            if len(ops) <= 20000:
                return ops
            best_g[state] = float('inf')
            continue

        # load
        if len(fast) < fast_memory_budget:
            for v in slow:
                if v not in fast:
                    nf = fast | frozenset((v,))
                    ns = slow - frozenset((v,))
                    st = (nf, ns)
                    ng = g + 1
                    if st not in best_g or ng < best_g[st]:
                        best_g[st] = ng
                        parent_map[st] = (state, 'load', v)
                        heapq.heappush(pq, (ng + heuristic(nf, ns), ng, st))
        # store
        for v in fast:
            nf = fast - frozenset((v,))
            ns = slow | frozenset((v,))
            st = (nf, ns)
            ng = g + 1
            if st not in best_g or ng < best_g[st]:
                best_g[st] = ng
                parent_map[st] = (state, 'store', v)
                heapq.heappush(pq, (ng + heuristic(nf, ns), ng, st))
        # compute
        if len(fast) < fast_memory_budget:
            for v in range(n):
                if v not in fast and parent_sets[v] and parent_sets[v] <= fast:
                    nf = fast | frozenset((v,))
                    st = (nf, slow)
                    ng = g
                    if st not in best_g or ng < best_g[st]:
                        best_g[st] = ng
                        parent_map[st] = (state, 'compute', v)
                        heapq.heappush(pq, (ng + heuristic(nf, slow), ng, st))
        # delete
        for v in fast:
            nf = fast - frozenset((v,))
            st = (nf, slow)
            ng = g
            if st not in best_g or ng < best_g[st]:
                best_g[st] = ng
                parent_map[st] = (state, 'delete', v)
                heapq.heappush(pq, (ng + heuristic(nf, slow), ng, st))
    return None


def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = t0 + 82.0

    parents, outputs, inputs, vertices, n = _parse_dag(dag)
    if parents is None or n is None:
        return []
    if outputs is None:
        outputs = [n - 1]
    if fast_memory_budget is None:
        fast_memory_budget = n
    if fast_memory_budget <= 0:
        fast_memory_budget = 1

    best_ops, best_transfers = None, float('inf')

    def consider(ops):
        nonlocal best_ops, best_transfers
        if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
            t = _count_transfers(ops)
            if t < best_transfers:
                best_transfers, best_ops = t, ops

    try:
        consider(_layered_pebbling(parents, outputs, fast_memory_budget, deadline))
    except Exception:
        pass
    try:
        consider(_greedy_pebbling(parents, outputs, fast_memory_budget, deadline))
    except Exception:
        pass
    if n <= 16:
        try:
            consider(_dijkstra_pebbling(parents, outputs, fast_memory_budget, deadline))
        except Exception:
            pass

    if best_ops is not None:
        return best_ops

    # last resort: trivial topological schedule with recompute
    try:
        children = _build_children(parents, n)
        is_output = set(outputs)
        last_use = [-1] * n
        for v in range(n):
            if children[v]:
                last_use[v] = max(children[v])
        ops, in_fast, stored = [], set(), set()

        def ensure_space():
            while len(in_fast) >= fast_memory_budget:
                ev = False
                for u in list(in_fast):
                    if u not in is_output:
                        ops.append(['delete', u])
                        in_fast.discard(u)
                        ev = True
                        break
                if not ev:
                    break
            return len(in_fast) < fast_memory_budget

        def make_available(p):
            if p in in_fast:
                return
            if p in stored:
                if ensure_space():
                    ops.append(['load', p])
                    in_fast.add(p)
                return
            for gp in parents[p]:
                make_available(gp)
            if ensure_space():
                ops.append(['compute', p])
                in_fast.add(p)

        for v in range(n):
            for p in parents[v]:
                make_available(p)
            if parents[v] and ensure_space():
                ops.append(['compute', v])
                in_fast.add(v)
            if v in is_output and v in in_fast:
                ops.append(['store', v])
                in_fast.discard(v)
                stored.add(v)
            for p in parents[v]:
                if p in in_fast and p not in is_output and last_use[p] == v:
                    ops.append(['delete', p])
                    in_fast.discard(p)
        for o in outputs:
            if o in in_fast:
                ops.append(['store', o])
                in_fast.discard(o)
        if _validate_ops(ops, parents, outputs, fast_memory_budget):
            return ops
    except Exception:
        pass
    return []

# EVOLVE-BLOCK-END
