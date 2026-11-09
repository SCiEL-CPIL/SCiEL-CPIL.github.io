"""Rewritten pebbling program: structural change to an exact state-space shortest-path
search (Dijkstra) over red-pebble configurations, with a fast greedy fallback and an
explicit deadline guard. Replaces the baseline call with a self-contained implementation."""
# EVOLVE-BLOCK-START
import time
import heapq
import itertools

_OP_COST = {"load": 1, "store": 1, "compute": 0, "delete": 0}
_MAX_OPS = 20000
_DEADLINE_MARGIN = 30.0  # seconds; evaluator kills at 90s, so finish by ~60s


def _topo_order(dag):
    """Return vertices in topological order (parents before children)."""
    vertices = list(dag.get("vertices", dag.get("vertex", [])))
    if not vertices:
        # Try alternate key names
        for k in ("vertices", "vertex", "nodes", "V"):
            if k in dag:
                vertices = list(dag[k])
                break
    n = len(vertices)
    # Build children map
    children = {v: [] for v in vertices}
    indeg = {v: 0 for v in vertices}
    # parents: for each vertex, its parents
    parents_map = {}
    for k in ("parents", "parent", "preds", "predecessors"):
        if k in dag:
            parents_map = dag[k]
            break
    for v in vertices:
        ps = parents_map.get(v, [])
        if ps is None:
            ps = []
        for p in ps:
            children[p].append(v)
            indeg[v] = indeg.get(v, 0) + 1
    # Kahn
    from collections import deque
    q = deque([v for v in vertices if indeg.get(v, 0) == 0])
    order = []
    indeg2 = dict(indeg)
    while q:
        v = q.popleft()
        order.append(v)
        for c in children[v]:
            indeg2[c] -= 1
            if indeg2[c] == 0:
                q.append(c)
    if len(order) != n:
        # Fallback: just use given order
        order = list(vertices)
    return order, parents_map, children, vertices


def _is_input(dag, v, parents_map):
    ps = parents_map.get(v, [])
    if ps is None:
        ps = []
    return len(list(ps)) == 0


def _is_output(dag, v, children):
    return len(children.get(v, [])) == 0


def _greedy_schedule(dag, fast_memory_budget, deadline):
    """Greedy topological schedule with load/store to respect budget.
    Returns list of [op, vertex]."""
    order, parents_map, children, vertices = _topo_order(dag)
    budget = fast_memory_budget
    ops = []
    # state: red set (in fast memory), blue set (computed & in slow memory)
    red = set()
    blue = set()
    # inputs are blue initially? "Inputs start blue" -> inputs are available in slow memory
    for v in order:
        if time.time() > deadline:
            break
        if _is_input(dag, v, parents_map):
            # input is blue in slow memory; to compute children we may need it red
            # We'll treat inputs as available (blue). They don't consume fast memory until loaded.
            blue.add(v)
            continue
        # Need all parents blue (available) to compute v.
        ps = list(parents_map.get(v, []) or [])
        # Ensure parents are blue: if a parent is red, it's already in fast memory (good for compute).
        # If a parent is neither red nor blue, we must have computed it earlier (should be blue).
        # If a parent is red, we can compute v without loading.
        # If a parent is blue (in slow), we need it red to compute? Actually compute reads parents.
        # Model: compute requires parents to be present (red) in fast memory.
        # So we must load any blue parent into red.
        for p in ps:
            if p not in red and p in blue:
                # load p into fast memory
                if len(red) >= budget:
                    # must delete something. Delete a red that is not an output and not needed soon.
                    # Simple: delete an arbitrary red that is not an output.
                    deleted = False
                    for r in list(red):
                        if not _is_output(dag, r, children):
                            ops.append(["delete", r])
                            red.discard(r)
                            deleted = True
                            break
                    if not deleted:
                        # delete any
                        if red:
                            r = next(iter(red))
                            ops.append(["delete", r])
                            red.discard(r)
                ops.append(["load", p])
                red.add(p)
                blue.discard(p)
        # Now compute v
        ops.append(["compute", v])
        # result v is red
        # if budget full and v not needed, maybe delete later
        red.add(v)
        # if v is an output, we must eventually store it (blue). Keep red until store.
        # If not output and budget tight, we could store now to free slot.
        if len(red) > budget:
            # shouldn't happen if we managed; but store v to free
            ops.append(["store", v])
            blue.add(v)
            red.discard(v)
    # Final: all outputs must be blue. Store any red outputs.
    for v in order:
        if time.time() > deadline:
            break
        if _is_output(dag, v, children):
            if v in red:
                ops.append(["store", v])
                blue.add(v)
                red.discard(v)
            # if v not blue and not red (shouldn't happen), load? It was computed.
    return ops


def _count_transfers(ops):
    return sum(1 for op, v in ops if op in ("load", "store"))


def _validate(ops, dag, fast_memory_budget):
    """Validate ops sequence. Returns True if valid."""
    order, parents_map, children, vertices = _topo_order(dag)
    red = set()
    blue = set()
    for v in vertices:
        if _is_input(dag, v, parents_map):
            blue.add(v)
    for op, v in ops:
        if op == "load":
            if v not in blue:
                return False
            if len(red) >= fast_memory_budget:
                return False
            red.add(v)
            blue.discard(v)
        elif op == "store":
            if v not in red:
                return False
            red.discard(v)
            blue.add(v)
        elif op == "compute":
            ps = list(parents_map.get(v, []) or [])
            for p in ps:
                if p not in red:
                    return False
            # result
            if v in red:
                return False
            red.add(v)
        elif op == "delete":
            if v not in red:
                return False
            red.discard(v)
        else:
            return False
    # all outputs blue
    for v in vertices:
        if _is_output(dag, v, children):
            if v not in blue:
                return False
    return True


def _dijkstra_exact(dag, fast_memory_budget, deadline, max_states=200000):
    """Exact shortest-path over red-pebble configurations.
    State: (frozenset(red), frozenset(blue_computed))
    We track red set and the set of computed vertices (blue or red).
    Cost = number of transfers (load/store).
    Goal: all outputs are blue (i.e., in blue set, not red).
    """
    order, parents_map, children, vertices = _topo_order(dag)
    outputs = [v for v in vertices if _is_output(dag, v, children)]
    inputs = [v for v in vertices if _is_input(dag, v, parents_map)]
    budget = fast_memory_budget
    if budget is None:
        budget = len(vertices)

    init_red = frozenset()
    init_blue = frozenset(inputs)
    init_state = (init_red, init_blue)

    def is_goal(red, blue):
        return all(o in blue for o in outputs)

    # Dijkstra
    # heap: (cost, counter, red, blue)
    counter = itertools.count()
    start_cost = 0
    heap = [(start_cost, next(counter), init_red, init_blue)]
    best = {init_state: start_cost}
    parent = {}
    goal_state = None
    states_explored = 0
    while heap:
        cost, _, red, blue = heapq.heappop(heap)
        state = (red, blue)
        if state in best and best[state] < cost:
            continue
        states_explored += 1
        if time.time() > deadline or states_explored > max_states:
            break
        if is_goal(red, blue):
            goal_state = state
            break
        # Generate successors
        # Operations:
        # 1. load v: v in blue, v not in red, len(red) < budget
        # 2. store v: v in red
        # 3. compute v: all parents in red, v not in (red|blue) [not computed yet]
        # 4. delete v: v in red
        # 1. load
        if len(red) < budget:
            for v in blue:
                if v not in red:
                    new_red = red | {v}
                    new_blue = blue - {v}
                    nstate = (new_red, new_blue)
                    ncost = cost + 1
                    if ncost < best.get(nstate, float("inf")):
                        best[nstate] = ncost
                        parent[nstate] = (state, ("load", v))
                        heapq.heappush(heap, (ncost, next(counter), new_red, new_blue))
        # 2. store
        for v in red:
            new_red = red - {v}
            new_blue = blue | {v}
            nstate = (new_red, new_blue)
            ncost = cost + 1
            if ncost < best.get(nstate, float("inf")):
                best[nstate] = ncost
                parent[nstate] = (state, ("store", v))
                heapq.heappush(heap, (ncost, next(counter), new_red, new_blue))
        # 3. compute
        computed = red | blue
        for v in vertices:
            if v in computed:
                continue
            ps = list(parents_map.get(v, []) or [])
            if all(p in red for p in ps):
                new_red = red | {v}
                new_blue = blue
                nstate = (new_red, new_blue)
                ncost = cost + 0
                if ncost < best.get(nstate, float("inf")):
                    best[nstate] = ncost
                    parent[nstate] = (state, ("compute", v))
                    heapq.heappush(heap, (ncost, next(counter), new_red, new_blue))
        # 4. delete
        for v in red:
            new_red = red - {v}
            new_blue = blue
            nstate = (new_red, new_blue)
            ncost = cost + 0
            if ncost < best.get(nstate, float("inf")):
                best[nstate] = ncost
                parent[nstate] = (state, ("delete", v))
                heapq.heappush(heap, (ncost, next(counter), new_red, new_blue))

    if goal_state is None:
        return None

    # Reconstruct path
    ops = []
    state = goal_state
    while state in parent:
        prev, (op, v) = parent[state]
        ops.append([op, v])
        state = prev
    ops.reverse()
    return ops


def pebble(dag, fast_memory_budget):
    start = time.time()
    deadline = start + _DEADLINE_MARGIN

    # Try exact Dijkstra if the DAG is small enough
    try:
        order, parents_map, children, vertices = _topo_order(dag)
        n = len(vertices)
        budget = fast_memory_budget if fast_memory_budget is not None else n
        # Exact search is feasible only for small n and small budget
        if n <= 10 and budget <= 4:
            ops = _dijkstra_exact(dag, fast_memory_budget, deadline, max_states=500000)
            if ops is not None and len(ops) <= _MAX_OPS and _validate(ops, dag, fast_memory_budget):
                return ops
    except Exception:
        pass

    # Fallback: greedy schedule
    ops = _greedy_schedule(dag, fast_memory_budget, deadline)
    if len(ops) > _MAX_OPS:
        ops = ops[:_MAX_OPS]
    return ops
# EVOLVE-BLOCK-END
