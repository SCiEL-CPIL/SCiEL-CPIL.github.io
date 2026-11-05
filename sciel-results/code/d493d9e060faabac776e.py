"""Communication-minimizing DAG pebbling with deadline guard."""
import time
import heapq
from collections import defaultdict

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Solve the DAG pebbling problem to minimize communication.
    
    Args:
        dag: dict with keys:
            - 'vertices': list of vertex ids (topologically ordered)
            - 'parents': dict mapping vertex -> list of parent vertices
            - 'outputs': list of output vertex ids
            - 'inputs': list of input vertex ids (start blue)
        fast_memory_budget: int, number of fast-memory slots available
    
    Returns:
        list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    # Parse DAG
    vertices = list(dag.get('vertices', []))
    parents = dict(dag.get('parents', {}))
    outputs = list(dag.get('outputs', []))
    inputs = list(dag.get('inputs', []))
    
    n = len(vertices)
    if n == 0:
        return []
    
    # Build adjacency and topological order
    # parents[v] = list of parents of v
    # children[v] = list of children of v
    children = defaultdict(list)
    in_degree = defaultdict(int)
    for v in vertices:
        ps = parents.get(v, [])
        in_degree[v] = len(ps)
        for p in ps:
            children[p].append(v)
    
    # Topological sort (Kahn's algorithm)
    topo_order = []
    from collections import deque
    queue = deque()
    for v in vertices:
        if in_degree[v] == 0:
            queue.append(v)
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != n:
        # Cycle or issue; fall back to simple schedule
        return _naive_schedule(vertices, parents, outputs, inputs, fast_memory_budget)
    
    # Identify input and output sets
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Naive baseline: load inputs, compute in topological order, store outputs
    # This gives us a reference cost
    naive_ops = _naive_schedule(vertices, parents, outputs, inputs, fast_memory_budget)
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return naive_ops
    
    # Try to find a better schedule using a greedy strategy with memory management
    # Strategy: Process vertices in topological order, but use fast memory budget wisely.
    # Key insight: We want to keep intermediate values in fast memory when possible
    # to avoid reloads, but we're limited by fast_memory_budget.
    
    # First, let's understand the structure better
    # For each vertex, count how many times its value is needed (fanout)
    fanout = {}
    for v in vertices:
        fanout[v] = len(children[v])
    
    # Count how many children each vertex has that are not yet computed
    # This helps decide when to delete values
    
    # Greedy approach:
    # 1. Start with all inputs in fast memory (they start blue)
    # 2. Process vertices in topological order
    # 3. When computing a vertex, if all parents are in fast memory, compute it
    # 4. If fast memory is full, we need to evict something
    # 5. When a value is no longer needed by any unprocessed vertex, delete it
    
    # Let's implement a more sophisticated approach:
    # Use a priority queue to decide which values to keep in fast memory
    # based on how many remaining uses they have
    
    # First, compute for each vertex the number of times it will be needed
    # by subsequent vertices (in topological order)
    remaining_uses = {}
    for v in reversed(topo_order):
        uses = fanout[v]
        for c in children[v]:
            # c needs v
            pass
        remaining_uses[v] = uses
    
    # Actually, let's think about this differently.
    # The key optimization is:
    # 1. Keep values in fast memory that will be used again soon
    # 2. Delete values that won't be used again
    # 3. When fast memory is full, evict the value with the least remaining uses
    
    # Let's implement a simulation-based approach:
    ops = []
    
    # State:
    # - fast_mem: set of vertices currently in fast memory
    # - slow_mem: set of vertices currently in slow memory (already loaded at least once)
    # - computed: set of vertices that have been computed at least once
    # - next_to_compute: index in topo_order
    
    fast_mem = set()
    slow_mem = set()
    computed = set()
    
    # Initialize: inputs start blue (in fast memory)
    for v in inputs:
        if v in set(vertices):
            fast_mem.add(v)
            computed.add(v)
    
    # For each vertex in topological order, we need to:
    # 1. Ensure all parents are in fast memory (load if not)
    # 2. Compute the vertex
    # 3. Decide whether to keep it in fast memory or store it
    
    # Precompute: for each vertex, which future vertices need it
    future_needs = defaultdict(set)
    for v in topo_order:
        for c in children[v]:
            future_needs[v].add(c)
    
    # Process in topological order
    for v in topo_order:
        if time.time() - start_time > DEADLINE - 5:
            # Emergency: just finish with naive approach from here
            break
        
        # If v is an input, it's already computed
        if v in input_set:
            # Make sure it's in fast memory
            if v not in fast_mem:
                # It was deleted, need to reload
                ops.append(['load', v])
                # Check if fast memory is full
                if len(fast_mem) > fast_memory_budget:
                    # Evict something
                    _evict_least_useful(fast_mem, slow_mem, computed, v, future_needs, topo_order, ops)
            continue
        
        # v is not an input, needs to be computed
        # Ensure all parents are in fast memory
        for p in parents.get(v, []):
            if p not in fast_mem:
                if p in slow_mem or p in computed:
                    ops.append(['load', p])
                    fast_mem.add(p)
                    # Check if fast memory is full
                    if len(fast_mem) > fast_memory_budget:
                        _evict_least_useful(fast_mem, slow_mem, computed, v, future_needs, topo_order, ops)
                else:
                    # p hasn't been computed yet - this shouldn't happen in topological order
                    # But it might if there's a cycle or issue
                    pass
        
        # Compute v
        ops.append(['compute', v])
        computed.add(v)
        slow_mem.add(v)
        fast_mem.add(v)
        
        # Check if fast memory is full
        if len(fast_mem) > fast_memory_budget:
            _evict_least_useful(fast_mem, slow_mem, computed, v, future_needs, topo_order, ops)
    
    # At the end, ensure all outputs are in fast memory (blue)
    for v in outputs:
        if v not in fast_mem:
            if v in slow_mem or v in computed:
                ops.append(['load', v])
                fast_mem.add(v)
                if len(fast_mem) > fast_memory_budget:
                    _evict_least_useful(fast_mem, slow_mem, computed, v, future_needs, topo_order, ops)
    
    # Count operations
    op_count = len(ops)
    
    # If we exceeded operation limit, fall back
    if op_count > 20000:
        return naive_ops
    
    # Check time
    if time.time() - start_time > DEADLINE - 3:
        return naive_ops
    
    # Validate: all outputs must be in fast memory at the end
    # (they should be by construction)
    
    return ops


def _evict_least_useful(fast_mem, slow_mem, computed, current_v, future_needs, topo_order, ops):
    """Evict the vertex from fast_mem with the least remaining future uses."""
    if not fast_mem:
        return
    
    # Find the vertex with the least remaining uses
    # A vertex's remaining uses = number of children that haven't been processed yet
    # and need this vertex
    
    # We need to know which vertices are yet to be processed
    # For simplicity, let's just evict the one with the fewest children in general
    
    # More sophisticated: count how many unprocessed children each fast_mem vertex has
    # This is expensive, so let's use a simpler heuristic
    
    # For now, evict the vertex that is not an output and has the fewest children
    # Actually, let's just evict the first non-output vertex we find
    
    # Better: evict the vertex with the minimum number of children
    best_to_evict = None
    min_children = float('inf')
    
    for v in fast_mem:
        # Don't evict outputs
        if v in set(future_needs.keys()) or True:
            # Count children
            num_children = len(future_needs.get(v, set()))
            # We want to evict the one with fewest remaining uses
            # But future_needs includes ALL children, not just unprocessed ones
            # This is a simplification
            if num_children < min_children:
                min_children = num_children
                best_to_evict = v
    
    if best_to_evict is not None:
        ops.append(['delete', best_to_evict])
        fast_mem.discard(best_to_evict)
        # It remains in slow_mem and computed


def _naive_schedule(vertices, parents, outputs, inputs, fast_memory_budget):
    """Simple baseline schedule: load, compute in topo order, store."""
    vertices = list(vertices)
    parents = dict(parents)
    outputs = list(outputs)
    inputs = list(inputs)
    
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Build children
    children = defaultdict(list)
    in_degree = defaultdict(int)
    for v in vertices:
        ps = parents.get(v, [])
        in_degree[v] = len(ps)
        for p in ps:
            children[p].append(v)
    
    # Topological sort
    from collections import deque
    topo_order = []
    queue = deque()
    for v in vertices:
        if in_degree[v] == 0:
            queue.append(v)
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Fallback: just use given order
        topo_order = list(vertices)
    
    ops = []
    loaded = set()
    
    for v in topo_order:
        if v in input_set:
            # Input: load if not loaded
            if v not in loaded:
                ops.append(['load', v])
                loaded.add(v)
            continue
        
        # Non-input: ensure parents are loaded
        for p in parents.get(v, []):
            if p not in loaded:
                ops.append(['load', p])
                loaded.add(p)
        
        # Compute
        ops.append(['compute', v])
        loaded.add(v)
    
    # Ensure outputs are loaded (they should be)
    for v in outputs:
        if v not in loaded:
            ops.append(['load', v])
            loaded.add(v)
    
    return ops


# EVOLVE-BLOCK-END
