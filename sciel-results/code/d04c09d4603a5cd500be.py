import time
import math

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    DEADLINE = 80.0

    # Parse DAG
    vertices = list(dag.get('vertices', []))
    if not vertices:
        parents = dag.get('parents', {})
        vertices = list(parents.keys())
    if not vertices:
        return []

    parents = dict(dag.get('parents', {}))
    for v in vertices:
        if v not in parents:
            parents[v] = []

    inputs = list(dag.get('inputs', dag.get('input_vertices', [])))
    outputs = list(dag.get('outputs', dag.get('output_vertices', [])))

    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents[v]:
            if p in children:
                children[p].append(v)

    # Compute depth (longest path from source)
    depth = {}
    for v in vertices:
        if not parents[v]:
            depth[v] = 0
        else:
            depth[v] = 1 + max(depth.get(p, 0) for p in parents[v])

    # Group by depth
    layers = {}
    for v in vertices:
        d = depth[v]
        if d not in layers:
            layers[d] = []
        layers[d].append(v)

    max_depth = max(layers.keys()) if layers else 0
    sorted_depths = sorted(layers.keys())

    B = int(fast_memory_budget) if fast_memory_budget is not None else 0
    if B < 0:
        B = 0

    ops = []
    op_count = 0
    MAX_OPS = 20000

    # Layer-based pebbling:
    # For each layer d:
    #   - Load all parents of vertices in layer d from blue to red.
    #   - Compute all vertices in layer d.
    #   - Store all vertices in layer d to blue (except outputs, which
    #     we keep red until the end).
    #   - Free all red vertices from previous layers.

    # Actually, a better approach: process each vertex in topological order,
    # and use a smart eviction policy.

    # Let me implement a clean topological processing with eviction.

    blue = set(inputs)
    red = set()

    # remaining_children[p] = number of children of p not yet processed
    remaining_children = {}
    for v in vertices:
        remaining_children[v] = len(children[v])

    # Process in topological order
    for v in vertices:
        if op_count >= MAX_OPS:
            break
        if time.time() - t0 > DEADLINE:
            break

        if v in blue:
            # Already blue, decrement remaining_children for parents
            for p in parents[v]:
                if p in remaining_children:
                    remaining_children[p] -= 1
            continue

        # Ensure all parents are red
        for p in parents[v]:
            if p in red:
                continue
            if p in blue:
                ops.append(['load', p])
                op_count += 1
                red.add(p)
                blue.discard(p)
            else:
                # Should not happen, but handle gracefully
                pass

        # Compute v
        ops.append(['compute', v])
        op_count += 1
        red.add(v)

        # Decrement remaining_children for v's parents
        for p in parents[v]:
            if p in remaining_children:
                remaining_children[p] -= 1

        # Evict if fast memory is full
        while len(red) > B and red:
            # Find best vertex to evict
            best_to_evict = None
            best_score = float('inf')
            for r in red:
                if r == v:
                    continue
                score = 0
                if r in outputs:
                    score += 1000
                if remaining_children.get(r, 0) > 0:
                    score += 500
                score += depth.get(r, 0) * 10
                score += len(children[r]) * 5
                if score < best_score:
                    best_score = score
                    best_to_evict = r
            if best_to_evict is None:
                break
            ops.append(['store', best_to_evict])
            op_count += 1
            red.discard(best_to_evict)
            blue.add(best_to_evict)

    # Ensure all outputs are blue
    for o in outputs:
        if o in red:
            ops.append(['store', o])
            op_count += 1
            red.discard(o)
            blue.add(o)

    return ops
