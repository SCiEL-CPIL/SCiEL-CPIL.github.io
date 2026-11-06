"""Communication-minimizing DAG pebbling."""
import time

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Parse dag structure. dag is expected to contain:
    #   - vertices (list or dict), topologically ordered
    #   - parents mapping: vertex -> list of parent vertices
    #   - inputs (start blue), outputs (must finish blue)
    # We handle several plausible shapes defensively.

    def _time_left():
        return DEADLINE - time.time()

    # Try to extract fields
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('V', None))
        parents = dag.get('parents', dag.get('edges', dag.get('pred', None)))
        inputs = dag.get('inputs', dag.get('input', dag.get('sources', None)))
        outputs = dag.get('outputs', dag.get('output', dag.get('sinks', None)))
    else:
        # maybe an object with attributes
        vertices = getattr(dag, 'vertices', None)
        parents = getattr(dag, 'parents', None)
        inputs = getattr(dag, 'inputs', None)
        outputs = getattr(dag, 'outputs', None)

    # Fallback: if vertices is a dict of adjacency, etc.
    if vertices is None and parents is not None:
        # try to infer vertex set from parents keys
        vertices = list(parents.keys())

    # Normalize vertices to list
    if vertices is None:
        vertices = []
    elif isinstance(vertices, dict):
        vertices = list(vertices.keys())

    # Normalize parents to dict
    if parents is None:
        parents = {}
    elif isinstance(parents, list):
        # list of (parent, child) edges
        pd = {}
        for e in parents:
            if isinstance(e, (tuple, list)) and len(e) == 2:
                p, c = e
                pd.setdefault(c, []).append(p)
        parents = pd

    # Normalize inputs/outputs to sets
    def _as_set(x):
        if x is None:
            return set()
        if isinstance(x, (set, frozenset)):
            return set(x)
        if isinstance(x, (list, tuple)):
            return set(x)
        return {x}

    if inputs is None:
        # inputs = vertices with no parents
        inputs = set(v for v in vertices if not parents.get(v))
    inputs = _as_set(inputs)

    if outputs is None:
        # outputs = vertices with no children
        child_set = set()
        for c, ps in parents.items():
            for p in ps:
                child_set.add(c)
        outputs = set(v for v in vertices if v not in child_set)
    outputs = _as_set(outputs)

    # Build child adjacency
    children = {}
    for c, ps in parents.items():
        for p in ps:
            children.setdefault(p, []).append(c)

    # Topological order (assume vertices already topologically ordered; verify)
    topo = list(vertices)
    # If parents given but vertices not ordered, do a topo sort
    try:
        indeg = {v: len(parents.get(v, [])) for v in topo}
        from collections import deque
        q = deque(v for v in topo if indeg[v] == 0)
        order = []
        indeg2 = dict(indeg)
        while q:
            u = q.popleft()
            order.append(u)
            for c in children.get(u, []):
                indeg2[c] -= 1
                if indeg2[c] == 0:
                    q.append(c)
        if len(order) == len(topo):
            topo = order
    except Exception:
        pass

    # Budget
    B = int(fast_memory_budget)
    if B < 1:
        B = 1

    # Strategy: top-down pebbling.
    # We process vertices in topological order. A vertex can be computed
    # when all its parents are available (blue) in fast memory (or it's an input,
    # which we load on demand). We keep a fast-memory set of currently blue
    # vertices. We delete (store) vertices that are no longer needed (no
    # remaining children) to free space, respecting the budget.

    ops = []

    def state():
        # not needed for output but for budget checks
        pass

    # We'll simulate.
    # fast: set of vertex ids currently in fast memory (blue)
    # remaining_children: count of children not yet computed
    remaining = {}
    for v in topo:
        remaining[v] = len(children.get(v, []))

    fast = set()

    # Track which vertices have been computed (so we don't recompute needlessly
    # if they're still in fast memory). But we may recompute if evicted.
    computed_in_fast = set()  # subset of fast that are computed values (not just loaded inputs)

    def load(v):
        ops.append(['load', v])
        fast.add(v)

    def store(v):
        ops.append(['store', v])
        fast.discard(v)
        computed_in_fast.discard(v)

    def compute(v):
        ops.append(['compute', v])
        fast.add(v)
        computed_in_fast.add(v)

    # We process in topological order. For each vertex:
    #  - if it's an input: ensure it's in fast (load if not)
    #  - else: ensure all parents are in fast (load them if evicted), then compute
    #  - after computing, optionally store parents that are no longer needed
    #    to keep |fast| <= B.

    # We need to be careful: a parent might be needed by multiple children, so
    # only store it when remaining[parent] == 0.

    # To respect budget, before computing a new vertex we may need to free space.
    # The new computed vertex takes one slot. So we need |fast| + 1 <= B,
    # i.e., we may need to store some vertices. But we can only safely store
    # vertices that are no longer needed (remaining == 0) OR vertices we are
    # willing to recompute later (inputs can be reloaded; computed values would
    # need recomputation). To keep it simple and valid, we store only
    # no-longer-needed vertices. If that's not enough, we may need to evict
    # a needed vertex and recompute it later — but that complicates things.

    # Given layered DAGs, the frontier width is usually <= B, so storing
    # no-longer-needed vertices should suffice. We'll add a fallback: if we
    # still exceed budget, evict the oldest computed non-output vertex and
    # mark it for recomputation (we'll recompute when needed).

    # Recomputation bookkeeping: a vertex may be evicted before all children
    # are computed. We'll track a "dirty" set: vertices that were computed but
    # are no longer in fast memory and still have remaining children. When a
    # child needs it, we recompute it (which requires its parents, recursively).

    # This can get complex. For layered DAGs with reasonable budget, the simple
    # top-down with store-when-done works. Let's implement that and add a
    # safety recompute path.

    # We'll implement a recursive "ensure_in_fast(v)" that loads/loads parents
    # as needed and recomputes if v was evicted.

    # To avoid infinite recursion and cycles, we use a visiting set.

    import sys
    sys.setrecursionlimit(100000)

    # We'll keep a memo: value_cache not needed; we recompute on the fly.
    # But recomputing a vertex requires its parents in fast, which may require
    # recomputing them, etc. This is fine for DAGs.

    # However, recomputation must not violate budget. We'll manage by storing
    # no-longer-needed vertices aggressively.

    def ensure(v, visiting=None):
        # Ensure v is in fast memory (blue).
        if v in fast:
            return
        if visiting is None:
            visiting = set()
        if v in visiting:
            return  # cycle guard (shouldn't happen in DAG)
        visiting.add(v)

        if v in inputs:
            # Load input
            if len(fast) >= B:
                # Need to free space: store any no-longer-needed vertex
                freed = _free_space(1, visiting)
                if not freed and len(fast) >= B:
                    # Evict a computed non-output vertex (will recompute later)
                    _evict_one(visiting)
            load(v)
            return

        # v is computed: need all parents in fast
        ps = parents.get(v, [])
        for p in ps:
            ensure(p, visiting)
        # Now compute v
        if len(fast) >= B:
            freed = _free_space(1, visiting)
            if not freed and len(fast) >= B:
                _evict_one(visiting)
        # v may already be in fast (if it was a parent loaded), but it's not
        # computed as a value necessarily. For non-input vertices, we compute.
        # If v is already in fast (e.g., it was an input? no, inputs handled
        # above). For computed vertices, if v in fast it means it was loaded
        # as input or is a parent that is also a value — but in a DAG a vertex
        # is either input or computed, not both. So v not in fast here.
        compute(v)
        visiting.discard(v)

    def _free_space(n, visiting):
        # Store vertices with remaining == 0 that are in fast (and not outputs
        # that must finish blue — but outputs we keep until the end; actually
        # outputs must finish blue, so we should NOT store them).
        freed = 0
        to_store = []
        for v in list(fast):
            if v in visiting:
                continue
            if v in outputs:
                continue  # must finish blue
            if remaining.get(v, 0) == 0:
                to_store.append(v)
        for v in to_store:
            if freed >= n:
                break
            store(v)
            freed += 1
        return freed > 0

    def _evict_one(visiting):
        # Evict a computed (non-input, non-output) vertex with remaining > 0.
        # It will be recomputed when needed. Prefer one with fewest remaining
        # children to minimize recompute cost.
        candidates = [v for v in fast
                      if v not in visiting
                      and v not in inputs
                      and v not in outputs
                      and v in computed_in_fast]
        if not candidates:
            # Evict any non-output, non-input
            candidates = [v for v in fast
                          if v not in visiting
                          and v not in inputs
                          and v not in outputs]
        if not candidates:
            return
        candidates.sort(key=lambda v: remaining.get(v, 0))
        v = candidates[0]
        store(v)

    # Process vertices in topological order. For each, ensure it's in fast
    # (which computes it if needed), then decrement remaining of its parents.
    for v in topo:
        if _time_left() < 1.0:
            break
        ensure(v)
        # Now v is in fast. Decrement remaining of its parents.
        for p in parents.get(v, []):
            remaining[p] = remaining.get(p, 0) - 1
            # If parent no longer needed, store it to free space
            if remaining[p] == 0 and p not in outputs and p in fast:
                store(p)
        # Also, v itself may become unneeded after its children are processed,
        # but we handle that when children are processed.

    # Ensure all outputs are in fast at the end (they must finish blue).
    for o in outputs:
        if o not in fast:
            ensure(o)

    # Truncate to 20000 ops if needed (shouldn't happen for reasonable DAGs)
    if len(ops) > 20000:
        ops = ops[:20000]

    return ops
# EVOLVE-BLOCK-END
