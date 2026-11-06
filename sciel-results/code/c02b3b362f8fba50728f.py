"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import heapq

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store) operations for DAG pebbling.
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    deadline = time.time() + 45  # safety margin before 60s limit

    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        parents = dag.get('parents', dag.get('edges', dag.get('pred', None)))
        inputs = dag.get('inputs', dag.get('sources', dag.get('input_nodes', None)))
        outputs = dag.get('outputs', dag.get('sinks', dag.get('output_nodes', None)))
    else:
        vertices = None
        parents = None
        inputs = None
        outputs = None

    # If we can't parse, fall back to baseline
    if vertices is None or parents is None:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    # Normalize to sets/lists
    if isinstance(vertices, (list, tuple, range)):
        vertex_list = list(vertices)
    elif isinstance(vertices, int):
        vertex_list = list(range(vertices))
    else:
        vertex_list = list(vertices)

    num_vertices = len(vertex_list)

    # Normalize parents to dict
    if isinstance(parents, dict):
        parent_dict = dict(parents)
    elif isinstance(parents, (list, tuple)):
        parent_dict = {i: list(p) for i, p in enumerate(parents)}
    else:
        parent_dict = {}

    for v in vertex_list:
        if v not in parent_dict:
            parent_dict[v] = []

    # Normalize inputs and outputs
    if inputs is None:
        inputs = set(v for v in vertex_list if not parent_dict.get(v, []))
    elif isinstance(inputs, (set, list, tuple)):
        inputs = set(inputs)
    else:
        inputs = set(inputs)

    if outputs is None:
        children = {v: [] for v in vertex_list}
        for v in vertex_list:
            for p in parent_dict.get(v, []):
                children[p].append(v)
        outputs = set(v for v in vertex_list if not children[v])
    elif isinstance(outputs, (set, list, tuple)):
        outputs = set(outputs)
    else:
        outputs = set(outputs)

    # Build children map
    children = {v: [] for v in vertex_list}
    for v in vertex_list:
        for p in parent_dict.get(v, []):
            children[p].append(v)

    # Determine needed vertices (ancestors of outputs)
    needed = set(outputs)
    for v in reversed(vertex_list):
        if v in needed:
            for p in parent_dict.get(v, []):
                needed.add(p)

    needed_vertices = [v for v in vertex_list if v in needed]
    needed_set = set(needed_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    # Compute last use index for each vertex
    last_child_idx = {}
    for v in needed_vertices:
        last_child_idx[v] = -1
        for child in children.get(v, []):
            if child in needed_set:
                last_child_idx[v] = max(last_child_idx[v], vertex_to_idx.get(child, -1))

    # Compute first use index (first child that needs it)
    first_child_idx = {}
    for v in needed_vertices:
        first_child_idx[v] = num_vertices
        for child in children.get(v, []):
            if child in needed_set:
                first_child_idx[v] = min(first_child_idx[v], vertex_to_idx.get(child, num_vertices))

    # Greedy pebbling with smart eviction
    in_fast = set()
    computed = set()
    operations = []
    total_comm = 0

    def load(v):
        nonlocal total_comm
        operations.append(['load', v])
        in_fast.add(v)
        total_comm += 1

    def store(v):
        nonlocal total_comm
        operations.append(['store', v])
        in_fast.discard(v)
        total_comm += 1

    def compute(v):
        operations.append(['compute', v])
        computed.add(v)
        in_fast.add(v)

    def delete(v):
        operations.append(['delete', v])
        in_fast.discard(v)

    # Count remaining uses: for each vertex, how many children still need it
    # Build a "remaining_uses" counter that we decrement as we process children
    remaining_uses = {}
    for v in needed_vertices:
        remaining_uses[v] = 0
    for v in needed_vertices:
        for p in parent_dict.get(v, []):
            if p in needed_set:
                remaining_uses[p] = remaining_uses.get(p, 0) + 1

    # Process vertices in topological order
    for v in needed_vertices:
        if time.time() > deadline:
            break

        parents = [p for p in parent_dict.get(v, []) if p in needed_set]
        v_idx = vertex_to_idx.get(v, -1)

        # Proactively evict vertices whose last use was before current position
        # (i.e., no more children need them). This frees slots before we need them.
        if len(in_fast) >= fast_memory_budget:
            for u in list(in_fast):
                if u == v:
                    continue
                if last_child_idx.get(u, -1) < v_idx:
                    store(u)

        # Ensure parents are in fast memory
        for p in parents:
            if p in in_fast:
                continue
            # Need to load parent
            if len(in_fast) >= fast_memory_budget:
                # Evict: prefer vertices with most remaining uses AFTER current
                # (i.e., least useful going forward)
                evicted = False
                # First try to evict vertices that are no longer needed
                for u in list(in_fast):
                    if u == v or u in parents:
                        continue
                    if last_child_idx.get(u, -1) < v_idx:
                        store(u)
                        evicted = True
                        break
                if not evicted:
                    # Evict vertex with latest last_use (least useful now)
                    best_evict = None
                    best_last = -1
                    for u in list(in_fast):
                        if u == v or u in parents:
                            continue
                        u_last = last_child_idx.get(u, -1)
                        if u_last > best_last:
                            best_last = u_last
                            best_evict = u
                    if best_evict is not None:
                        store(best_evict)
                    else:
                        # Emergency: delete any non-parent
                        for u in list(in_fast):
                            if u not in parents and u != v:
                                delete(u)
                                break
            load(p)

        # Compute v
        if v not in in_fast:
            compute(v)

        # Decrement remaining uses for parents (v is now processed)
        for p in parents:
            remaining_uses[p] = remaining_uses.get(p, 0) - 1

        # Clean up: store vertices no longer needed (last use passed)
        for u in list(in_fast):
            if u == v:
                continue
            if last_child_idx.get(u, -1) < v_idx:
                store(u)

    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                for u in list(in_fast):
                    if u not in outputs:
                        store(u)
                        break
            load(v)

    return operations
# EVOLVE-BLOCK-END
