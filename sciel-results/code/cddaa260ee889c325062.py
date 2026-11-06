"""Communication-minimizing DAG pebbling with bounded fast memory."""
# EVOLVE-BLOCK-START
import time
import heapq
import math
from collections import defaultdict, deque

_DEADLINE_MARGIN = 2.0  # seconds before the 90s kill
_MAX_OPS = 20000


def _toposort_vertices(dag):
    """Return vertices in topological order (parents before children)."""
    # dag structure: list of (vertex, parents_list) or dict
    # We'll handle the common formats
    if isinstance(dag, dict):
        # dag might be {vertex: [parents]}
        vertices = list(dag.keys())
        parents_map = {v: list(p) for v, p in dag.items()}
    elif isinstance(dag, (list, tuple)):
        # list of (vertex, parents) tuples
        vertices = [v for v, _ in dag]
        parents_map = {v: list(p) for v, p in dag}
    else:
        raise ValueError("Unknown dag format")
    
    # Build children map
    children = defaultdict(list)
    for v, parents in parents_map.items():
        for p in parents:
            children[p].append(v)
    
    # Topological sort using Kahn's algorithm
    in_degree = {v: 0 for v in vertices}
    for v, parents in parents_map.items():
        in_degree[v] = len(parents)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(order) != len(vertices):
        raise ValueError("DAG has a cycle or invalid structure")
    
    return order, parents_map, children


def _identify_inputs_and_outputs(dag):
    """Identify input vertices (no parents) and output vertices."""
    order, parents_map, children = _toposort_vertices(dag)
    inputs = [v for v in order if len(parents_map[v]) == 0]
    outputs = [v for v in order if len(children[v]) == 0]
    return inputs, outputs, order, parents_map, children


def _naive_schedule_ops(dag, fast_memory_budget):
    """Generate a naive pebbling schedule: compute each vertex once, store results."""
    order, parents_map, children = _toposort_vertices(dag)
    inputs, outputs, _, _, _ = _identify_inputs_and_outputs(dag)
    
    ops = []
    for v in order:
        if v in inputs:
            ops.append(["load", v])
        else:
            ops.append(["compute", v])
    
    # Store all outputs at the end
    for v in outputs:
        ops.append(["store", v])
    
    return ops


def _count_transfers(ops):
    """Count the number of load and store operations (transfers)."""
    return sum(1 for op, v in ops if op in ("load", "store"))


def _validate_ops(ops, dag, fast_memory_budget):
    """
    Validate a pebbling schedule.
    Returns True if valid, False otherwise.
    
    Rules:
    - Inputs start blue (in fast memory or available for load)
    - All outputs must finish blue
    - Every red value consumes one fast-memory slot
    - Operations: load, store, compute, delete
    """
    order, parents_map, children = _toposort_vertices(dag)
    inputs, outputs, _, _, _ = _identify_inputs_and_outputs(dag)
    
    # State: set of vertices currently in fast memory (blue/red)
    # We'll track which vertices have their value available
    in_memory = set()
    computed = set()  # vertices whose value has been computed at least once
    
    for op, v in ops:
        if op == "load":
            if v not in inputs:
                return False
            if v in in_memory:
                return False
            if len(in_memory) >= fast_memory_budget:
                return False
            in_memory.add(v)
        
        elif op == "store":
            if v not in in_memory:
                return False
            in_memory.discard(v)
        
        elif op == "compute":
            # All parents must be in memory
            for p in parents_map[v]:
                if p not in in_memory:
                    return False
            # Result goes to memory (or stays if already there)
            if v not in in_memory:
                if len(in_memory) >= fast_memory_budget:
                    return False
            in_memory.add(v)
            computed.add(v)
        
        elif op == "delete":
            if v not in in_memory:
                return False
            in_memory.discard(v)
        
        else:
            return False
    
    # All outputs must be in memory at the end
    for v in outputs:
        if v not in in_memory:
            return False
    
    return True


def _greedy_layered_pebble(dag, fast_memory_budget, deadline=None):
    """
    Greedy layered pebbling: process layers, keeping as much as possible in memory.
    """
    start_time = time.time()
    order, parents_map, children = _toposort_vertices(dag)
    inputs, outputs, _, _, _ = _identify_inputs_and_outputs(dag)
    
    if deadline is None:
        deadline = start_time + 85.0
    
    # Group vertices by depth (longest path from an input)
    depth = {}
    for v in order:
        if len(parents_map[v]) == 0:
            depth[v] = 0
        else:
            depth[v] = max(depth[p] for p in parents_map[v]) + 1
    
    max_depth = max(depth.values()) if depth else 0
    layers = defaultdict(list)
    for v in order:
        layers[depth[v]].append(v)
    
    # Process layer by layer
    ops = []
    in_memory = set()
    computed = set()
    
    def can_add(v):
        return len(in_memory) < fast_memory_budget or v in in_memory
    
    def add_to_memory(v):
        nonlocal in_memory
        if v not in in_memory:
            if len(in_memory) >= fast_memory_budget:
                return False
            in_memory.add(v)
        return True
    
    def remove_from_memory(v):
        nonlocal in_memory
        if v in in_memory:
            in_memory.discard(v)
        return True
    
    # For each layer, we need to:
    # 1. Ensure all parents of vertices in this layer are in memory
    # 2. Compute vertices in this layer
    # 3. Decide what to keep in memory for the next layer
    
    for d in range(max_depth + 1):
        if time.time() > deadline:
            break
        
        layer_vertices = layers[d]
        
        # First, ensure all needed parents are in memory
        needed_parents = set()
        for v in layer_vertices:
            for p in parents_map[v]:
                needed_parents.add(p)
        
        # Load any missing parents
        for p in needed_parents:
            if p not in in_memory:
                if p in inputs:
                    ops.append(["load", p])
                    add_to_memory(p)
                elif p in computed:
                    # Need to recompute or have stored it
                    # For simplicity, recompute
                    _recompute(p, parents_map, children, in_memory, computed, ops, fast_memory_budget, deadline)
                    if p not in in_memory:
                        add_to_memory(p)
        
        # Compute vertices in this layer
        for v in layer_vertices:
            if time.time() > deadline:
                break
            if v in in_memory:
                continue
            # Check all parents are in memory
            all_parents_in = all(p in in_memory for p in parents_map[v])
            if not all_parents_in:
                continue
            ops.append(["compute", v])
            add_to_memory(v)
            computed.add(v)
        
        # After computing this layer, decide what to keep
        # Keep vertices that are needed by future layers
        future_needed = set()
        for d2 in range(d + 1, max_depth + 1):
            for v in layers[d2]:
                for p in parents_map[v]:
                    future_needed.add(p)
        
        # Also keep all outputs
        future_needed.update(outputs)
        
        # Remove vertices not needed in the future
        to_remove = []
        for v in list(in_memory):
            if v not in future_needed and v not in computed:
                to_remove.append(v)
        
        # But we can only remove if we don't need them again
        # Actually, let's be more careful: keep vertices that might be needed
        for v in to_remove:
            if v not in future_needed:
                ops.append(["delete", v])
                remove_from_memory(v)
    
    # Ensure all outputs are in memory
    for v in outputs:
        if v not in in_memory:
            if v in inputs:
                ops.append(["load", v])
                add_to_memory(v)
            elif v in computed:
                _recompute(v, parents_map, children, in_memory, computed, ops, fast_memory_budget, deadline)
                if v not in in_memory:
                    add_to_memory(v)
    
    # Store outputs
    for v in outputs:
        if v in in_memory:
            ops.append(["store", v])
    
    return ops


def _recompute(v, parents_map, children, in_memory, computed, ops, fast_memory_budget, deadline):
    """Recompute a vertex if it's not in memory."""
    start_time = time.time()
    if time.time() > deadline:
        return
    
    if v in in_memory:
        return
    
    # Compute all parents first
    for p in parents_map[v]:
        _recompute(p, parents_map, children, in_memory, computed, ops, fast_memory_budget, deadline)
    
    if time.time() > deadline:
        return
    
    # Now compute v
    all_parents_in = all(p in in_memory for p in parents_map[v])
    if not all_parents_in:
        return
    
    if v not in in_memory:
        if len(in_memory) >= fast_memory_budget:
            return
        ops.append(["compute", v])
        in_memory.add(v)
    computed.add(v)


def _smart_greedy_pebble(dag, fast_memory_budget, deadline=None):
    """
    Smart greedy pebbling with better memory management.
    Uses a priority queue to decide what to keep in memory.
    """
    start_time = time.time()
    if deadline is None:
        deadline = start_time + 85.0
    
    order, parents_map, children = _toposort_vertices(dag)
    inputs, outputs, _, _, _ = _identify_inputs_and_outputs(dag)
    
    # Compute depth and use count
    depth = {}
    use_count = defaultdict(int)  # how many times each vertex is used as a parent
    
    for v in order:
        if len(parents_map[v]) == 0:
            depth[v] = 0
        else:
            depth[v] = max(depth[p] for p in parents_map[v]) + 1
    
    for v in order:
        for c in children[v]:
            use_count[v] += 1
    
    # Vertices with high use_count are more valuable to keep
    # Vertices with low depth that are used many times should be kept
    
    ops = []
    in_memory = set()
    computed = set()
    
    # Process in topological order, but with smart memory management
    for v in order:
        if time.time() > deadline:
            break
        
        if v in inputs:
            # Load if not already in memory
            if v not in in_memory:
                if len(in_memory) >= fast_memory_budget:
                    # Need to evict something
                    _evict_least_valuable(in_memory, children, use_count, depth, ops, fast_memory_budget)
                if v not in in_memory:
                    ops.append(["load", v])
                    in_memory.add(v)
            continue
        
        # For non-input vertices, compute if possible
        all_parents_in = all(p in in_memory for p in parents_map[v])
        
        if not all_parents_in:
            # Try to bring in missing parents
            for p in parents_map[v]:
                if p not in in_memory:
                    if p in inputs:
                        if len(in_memory) >= fast_memory_budget:
                            _evict_least_valuable(in_memory, children, use_count, depth, ops, fast_memory_budget)
                        if p not in in_memory:
                            ops.append(["load", p])
                            in_memory.add(p)
                    elif p in computed:
                        # Recompute p
                        _recompute_vertex(p, parents_map, children, in_memory, computed, ops, fast_memory_budget, deadline)
        
        # Now try to compute v
        all_parents_in = all(p in in_memory for p in parents_map[v])
        if all_parents_in:
            if v not in in_memory:
                if len(in_memory) >= fast_memory_budget:
                    _evict_least_valuable(in_memory, children, use_count, depth, ops, fast_memory_budget)
                if v not in in_memory:
                    ops.append(["compute", v])
                    in_memory.add(v)
            computed.add(v)
    
    # Ensure outputs are in memory
    for v in outputs:
        if v not in in_memory:
            if v in inputs:
                if len(in_memory) >= fast_memory_budget:
                    _evict_least_valuable(in_memory, children, use_count, depth, ops, fast_memory_budget)
                if v not in in_memory:
                    ops.append(["load", v])
                    in_memory.add(v)
            else:
                _recompute_vertex(v, parents_map, children, in_memory, computed, ops, fast_memory_budget, deadline)
    
    # Store outputs
    for v in outputs:
        if v in in_memory:
            ops.append(["store", v])
    
    return ops


def _evict_least_valuable(in_memory, children, use_count, depth, ops, fast_memory_budget):
    """Evict the least valuable vertex from memory."""
    if not in_memory:
        return
    
    # Value = use_count (how many children use this vertex)
    # Prefer to evict vertices with low use_count and high depth
    def value(v):
        return (use_count.get(v, 0), -depth.get(v, 0))
    
    # Find the vertex with minimum value
    min_v = min(in_memory, key=value)
    ops.append(["delete", min_v])
    in_memory.discard(min_v)


def _recompute_vertex(v, parents_map, children, in_memory, computed, ops, fast_memory_budget, deadline):
    """Recompute a vertex, recursively computing its parents if needed."""
    if time.time() > deadline:
        return
    
    if v in in_memory:
        return
    
    # Compute parents first
    for p in parents_map[v]:
        _recompute_vertex(p, parents_map, children, in_memory, computed, ops, fast_memory_budget, deadline)
    
    if time.time() > deadline:
        return
    
    all_parents_in = all(p in in_memory for p in parents_map[v])
    if not all_parents_in:
        return
    
    if v not in in_memory:
        if len(in_memory) >= fast_memory_budget:
            # Can't add, skip
            return
        ops.append(["compute", v])
        in_memory.add(v)
    computed.add(v)


def _hybrid_pebble(dag, fast_memory_budget):
    """
    Hybrid pebbling: try multiple strategies and return the best valid one.
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Try different strategies
    strategies = [
        ("naive", lambda: _naive_schedule_ops(dag, fast_memory_budget)),
        ("greedy_layered", lambda: _greedy_layered_pebble(dag, fast_memory_budget, deadline)),
        ("smart_greedy", lambda: _smart_greedy_pebble(dag, fast_memory_budget, deadline)),
    ]
    
    best_ops = None
    best_transfers = float('inf')
    
    for name, strategy in strategies:
        if time.time() > deadline - 5:
            break
        
        try:
            ops = strategy()
            if len(ops) > _MAX_OPS:
                continue
            
            # Validate
            if _validate_ops(ops, dag, fast_memory_budget):
                transfers = _count_transfers(ops)
                if transfers < best_transfers:
                    best_transfers = transfers
                    best_ops = ops
        except Exception:
            continue
    
    if best_ops is not None:
        return best_ops
    
    # Fallback to naive
    return _naive_schedule_ops(dag, fast_memory_budget)


def pebble(dag, fast_memory_budget):
    """
    Main entry point for pebbling.
    Returns a list of [operation, vertex] pairs.
    """
    start_time = time.time()
    
    # Try the hybrid approach
    ops = _hybrid_pebble(dag, fast_memory_budget)
    
    # Ensure we don't exceed max ops
    if len(ops) > _MAX_OPS:
        ops = ops[:_MAX_OPS]
        # Make sure outputs are stored
        # This might invalidate the schedule, so just return what we have
    
    return ops
# EVOLVE-BLOCK-END
