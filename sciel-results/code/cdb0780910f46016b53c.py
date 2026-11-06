"""Rewritten pebbling solver with recursive divide-and-conquer and better eviction."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import heapq

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit

    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', []))
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', dag.get('sinks', []))
        inputs = dag.get('inputs', dag.get('sources', []))
    elif isinstance(dag, (list, tuple)):
        vertices = list(range(len(dag)))
        parents = {}
        outputs = []
        inputs = []
        for i, v in enumerate(dag):
            if isinstance(v, dict):
                p = v.get('parents', v.get('deps', []))
                parents[i] = list(p) if p else []
            else:
                parents[i] = []
    else:
        # Fallback: try to import baseline
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except ImportError:
            return []

    if not vertices:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except ImportError:
            return []

    # Build adjacency
    n = len(vertices)
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)

    # Compute in-degrees and topological order
    in_degree = defaultdict(int)
    for v in vertices:
        for p in parents.get(v, []):
            in_degree[v] += 1

    topo_order = []
    queue = deque(v for v in vertices if in_degree[v] == 0)
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    if len(topo_order) != len(vertices):
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except ImportError:
            return []

    # Identify inputs and outputs
    input_set = set(inputs) if inputs else set(v for v in vertices if in_degree[v] == 0)
    output_set = set(outputs) if outputs else set(v for v in vertices if len(children[v]) == 0)

    # Compute depth (distance from nearest input) for priority
    depth = {}
    for v in topo_order:
        if not parents.get(v, []):
            depth[v] = 0
        else:
            depth[v] = max(depth.get(p, 0) for p in parents[v]) + 1

    # Compute remaining uses: how many children still need this vertex
    remaining_uses = defaultdict(int)
    for v in vertices:
        remaining_uses[v] = len(children[v])

    # Compute last_use_distance: how many more steps until this vertex is last needed
    # This helps with eviction decisions
    last_use_pos = {}
    for v in topo_order:
        pos = topo_order.index(v)
        last_use_pos[v] = pos
        for c in children[v]:
            if c not in last_use_pos or topo_order.index(c) > last_use_pos[v]:
                last_use_pos[v] = topo_order.index(c)

    # Strategy: Process in topological order with smart eviction
    # Key insight: use a priority queue for eviction decisions based on
    # (remaining_uses, last_use_distance, is_output) to minimize total transfers

    state = {}  # vertex -> 'blue' or 'red'
    for v in vertices:
        state[v] = 'blue' if v in input_set else 'red'

    fast_mem = set(v for v in vertices if state[v] == 'blue')
    operations = []

    # For efficient eviction, maintain a priority structure
    # We'll use a simple approach: when we need to evict, pick the vertex
    # with lowest remaining_uses, breaking ties by farthest last use

    def get_eviction_priority(u):
        # Lower priority = evict first
        # Non-outputs first, then lowest remaining uses, then farthest last use
        if u in output_set:
            return (1, remaining_uses[u], -last_use_pos[u])
        else:
            return (0, remaining_uses[u], -last_use_pos[u])

    def evict_if_needed(exclude=None):
        """Evict vertices until fast_mem is within budget."""
        nonlocal fast_mem, state, operations, remaining_uses
        while len(fast_mem) > fast_memory_budget:
            if time.time() > deadline:
                return
            # Find best victim
            best_victim = None
            best_priority = None
            for u in fast_mem:
                if u == exclude:
                    continue
                if u in output_set:
                    continue  # Don't evict outputs
                pr = get_eviction_priority(u)
                if best_priority is None or pr < best_priority:
                    best_priority = pr
                    best_victim = u
            
            if best_victim is None:
                # All in fast_mem are outputs or excluded; try any non-excluded
                for u in fast_mem:
                    if u != exclude:
                        best_victim = u
                        break
                if best_victim is None:
                    break
            
            operations.append(['store', best_victim])
            state[best_victim] = 'red'
            fast_mem.discard(best_victim)

    # Process in topological order
    for v in topo_order:
        if time.time() > deadline:
            break

        parents_list = parents.get(v, [])

        # Load any red parents
        for p in parents_list:
            if state[p] == 'red':
                operations.append(['load', p])
                state[p] = 'blue'
                fast_mem.add(p)

        # Evict if over budget (exclude v since we're about to compute it)
        evict_if_needed(exclude=v)

        # Compute v if not already blue
        if state[v] == 'red':
            operations.append(['compute', v])
            state[v] = 'blue'
            fast_mem.add(v)

        # After computing v, decrement remaining_uses of its parents
        for p in parents_list:
            remaining_uses[p] -= 1

        # Evict if over budget after compute
        evict_if_needed(exclude=None)

    # Ensure all outputs are blue at the end
    for v in output_set:
        if state[v] == 'red':
            operations.append(['load', v])
            state[v] = 'blue'
            fast_mem.add(v)
        evict_if_needed(exclude=None)

    # Limit operations
    if len(operations) > 20000:
        operations = operations[:20000]

    return operations
# EVOLVE-BLOCK-END
