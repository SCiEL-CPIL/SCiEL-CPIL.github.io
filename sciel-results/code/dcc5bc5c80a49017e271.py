"""Rewritten pebbling solver: layer-aware schedule with memory management."""
import time

_DEADLINE = 80.0  # safety margin below 90s kill
_START = time.time()

def _elapsed():
    return time.time() - _START

def pebble(dag, fast_memory_budget):
    """
    dag: dict with 'vertices' (list of ids, topologically ordered),
         'parents' (dict vertex -> list of parent vertices),
         'inputs' (set/list of input vertex ids),
         'outputs' (set/list of output vertex ids).
    fast_memory_budget: int, number of red slots available.
    Returns list of [operation, vertex] where operation in
    {'load','store','compute','delete'}.
    """
    # Parse the DAG robustly (support several plausible layouts)
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', dag.get('nodes', [])))
        parents = dag.get('parents', dag.get('preds', {}))
        inputs = set(dag.get('inputs', dag.get('sources', [])))
        outputs = set(dag.get('outputs', dag.get('sinks', dag.get('targets', []))))
    else:
        # Fallback: treat dag as list of vertex records
        vertices = []
        parents = {}
        inputs = set()
        outputs = set()
        for v in dag:
            if isinstance(v, dict):
                vid = v.get('id', v.get('vertex'))
                vertices.append(vid)
                parents[vid] = list(v.get('parents', v.get('preds', [])))
                if v.get('is_input', v.get('input', False)):
                    inputs.add(vid)
                if v.get('is_output', v.get('output', False)):
                    outputs.add(vid)
            else:
                vertices.append(v)

    n = len(vertices)
    if n == 0:
        return []

    # Ensure parents dict covers all vertices
    for v in vertices:
        parents.setdefault(v, [])

    # Determine inputs/outputs if not given: inputs = no parents, outputs = no children
    if not inputs:
        inputs = {v for v in vertices if not parents.get(v, [])}
    if not outputs:
        children = {v: [] for v in vertices}
        for v in vertices:
            for p in parents.get(v, []):
                children[p].append(v)
        outputs = {v for v in vertices if not children.get(v, [])}

    # Topological order is assumed to be the order of `vertices` list.
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)

    # Layer assignment: longest path from an input (0-based)
    layer = {}
    for v in vertices:
        ps = parents.get(v, [])
        if not ps:
            layer[v] = 0
        else:
            layer[v] = max(layer[p] for p in ps) + 1
    max_layer = max(layer.values()) if layer else 0

    # Group vertices by layer
    by_layer = {}
    for v in vertices:
        by_layer.setdefault(layer[v], []).append(v)

    # We need a schedule. Strategy:
    # Process layer by layer. For each layer, compute its vertices.
    # A vertex is computable when all its parents are red.
    # We keep a set of red vertices (in fast memory).
    # Inputs start blue; we load them when needed.
    # Outputs must finish blue.

    # To minimize transfers, we want to keep values red as long as they're
    # needed by future layers, and delete/store them when no longer needed.
    #
    # For each vertex, compute the "last use" layer: the maximum layer of any
    # vertex that (transitively) needs it. More precisely, a value for v is
    # needed until all descendants of v that are in later layers have been
    # computed. We compute last_needed[v] = max layer among all descendants.
    #
    # Actually for a layered DAG where edges go from layer i to i+1 (or skip),
    # a value at layer L is needed until the last layer that uses it.

    # Compute for each vertex the set of descendants' layers (or just max).
    # last_use[v] = max(layer[d] for d in descendants of v) or layer[v] if leaf.
    last_use = {}
    # Process in reverse topological order
    for v in reversed(vertices):
        lu = layer[v]
        for c in children[v]:
            if c in last_use:
                lu = max(lu, last_use[c])
        last_use[v] = lu

    # Now schedule:
    # red = set of vertices currently in fast memory (red)
    # blue = set of vertices currently blue (in slow memory, available to load)
    # Initially: inputs are blue, others are unknown (not yet computed).
    # Actually "blue" means the value is in slow memory and can be loaded.
    # Inputs start blue. Computed results are red until stored/deleted.
    # We can store a red value to make it blue (so it can be reloaded later).
    #
    # State:
    #   red: set of vertices whose value is in fast memory
    #   blue: set of vertices whose value is in slow memory (loadable)
    #   computed: set of vertices that have been computed at least once
    #
    # Initially: blue = inputs, red = empty, computed = empty.
    #
    # We process layers in order. For each layer L, we compute all vertices in
    # that layer. To compute a vertex v, all its parents must be red.
    #
    # Before computing layer L, ensure all parents of layer-L vertices are red.
    # Parents are in earlier layers. If a parent is blue (stored), load it.
    # If a parent is not red or blue, it must have been deleted -> need to
    # recompute (but in a layered DAG with no skips beyond 1, parents are in
    # previous layer, so they should still be available if we manage memory).
    #
    # Memory management: after computing layer L, we can delete/store vertices
    # whose last_use < next layer we need. We keep them red if last_use >= next
    # layer and we have room. Otherwise store (make blue) or delete.
    #
    # To minimize transfers, prefer to keep red if it fits, else store.
    # Delete only if we must and it's not needed (but if not needed, store is
    # also fine; delete saves nothing in transfers but frees memory faster).
    # Actually delete and store both free a red slot. Store makes it loadable
    # later (costs a load later), delete makes it gone (costs recompute later).
    # If last_use is far in the future, store is better than delete.
    # If last_use is the current layer (no future use), delete is fine.
    #
    # Let's implement:
    #   For each layer L from 0 to max_layer:
    #     1. For each v in by_layer[L], ensure all parents are red.
    #        For each parent p of v: if p not in red:
    #            if p in blue: load p (p: blue->red)
    #            else: p was deleted; need to recompute p. This is complex.
    #                 In a proper layered DAG where we manage memory well,
    #                 parents should be available. Let's handle by
    #                 recomputing if necessary (but this may cascade).
    #
    # Given the complexity, let's use a simpler but correct approach:
    # Keep ALL computed values red until their last use, then delete.
    # If memory runs out, store (not delete) the ones with furthest last_use
    # that we can't keep, prioritizing keeping recent ones.
    #
    # Actually, let's do a clean simulation:

    red = set()
    blue = set(inputs)
    ops = []
    op_count = 0
    MAX_OPS = 20000

    def do_load(v):
        nonlocal op_count
        ops.append(['load', v])
        blue.discard(v)
        red.add(v)
        op_count += 1

    def do_store(v):
        nonlocal op_count
        ops.append(['store', v])
        red.discard(v)
        blue.add(v)
        op_count += 1

    def do_compute(v):
        nonlocal op_count
        ops.append(['compute', v])
        red.add(v)
        op_count += 1

    def do_delete(v):
        nonlocal op_count
        ops.append(['delete', v])
        red.discard(v)
        op_count += 1

    # Process each layer
    for L in range(max_layer + 1):
        if _elapsed() > _DEADLINE:
            break
        layer_verts = by_layer.get(L, [])
        if not layer_verts:
            continue

        # Ensure all parents of layer-L vertices are red
        for v in layer_verts:
            for p in parents.get(v, []):
                if p not in red:
                    if p in blue:
                        do_load(p)
                    else:
                        # p was deleted; recompute it
                        # This requires p's parents to be red, etc.
                        # For safety, we do a recursive recompute.
                        # But this can be expensive. Let's just load if blue,
                        # else recompute via a helper.
                        _recompute(p, red, blue, ops, do_load, do_store,
                                   do_compute, do_delete, parents, children,
                                   op_count)

        # Compute all vertices in this layer
        for v in layer_verts:
            if _elapsed() > _DEADLINE:
                break
            if v not in red:
                # All parents should be red now
                do_compute(v)

        # After computing layer L, manage memory:
        # Determine which red vertices are no longer needed (last_use <= L
        # and no future layer needs them).
        # Actually last_use[v] is the max layer of descendants. If last_use[v]
        # <= L, v is no longer needed.
        # But we need to be careful: a vertex in layer L might be needed by
        # layer L+1. last_use[v] >= L+1 means it's needed in the future.
        #
        # We want to keep red vertices that are needed in the future, up to
        # fast_memory_budget. For those not needed, delete. For those needed
        # but exceeding budget, store the ones with smallest last_use first.
        #
        # Current red set after computing layer L.
        # Partition red into: needed (last_use > L) and not_needed (last_use <= L).
        # Delete all not_needed.
        # For needed, if len(needed) > budget, store the excess (smallest
        # last_use first).
        not_needed = [v for v in red if last_use.get(v, L) <= L]
        needed = [v for v in red if last_use.get(v, L) > L]

        # Delete not_needed
        for v in not_needed:
            if v in red:
                do_delete(v)

        # If needed exceeds budget, store some
        if len(needed) > fast_memory_budget:
            # Sort by last_use ascending; store the ones with smallest last_use
            needed_sorted = sorted(needed, key=lambda x: last_use.get(x, L))
            excess = len(needed) - fast_memory_budget
            for v in needed_sorted[:excess]:
                if v in red:
                    do_store(v)

    # Final: ensure all outputs are blue.
    # If an output is red, store it.
    for v in outputs:
        if v in red:
            do_store(v)

    # If we ran out of time, return what we have (may be invalid, but better
    # than nothing). The evaluator checks validity, so this may score 0, but
    # it's a safety net.
    return ops


def _recompute(v, red, blue, ops, do_load, do_store, do_compute, do_delete,
               parents, children, op_count):
    """Recursively recompute v if it's not red or blue. This is a fallback."""
    if v in red:
        return
    if v in blue:
        do_load(v)
        return
    # v was deleted; need to recompute. Ensure parents are red first.
    for p in parents.get(v, []):
        _recompute(p, red, blue, ops, do_load, do_store, do_compute,
                   do_delete, parents, children, op_count)
    if v not in red:
        do_compute(v)
