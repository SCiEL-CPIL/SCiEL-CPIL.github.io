# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque
import math

def _topo_sort(vertices, parents_map):
    """Compute topological order given vertices and parents."""
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children_map[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    return order

def _get_children_map(vertices, parents_map):
    """Build children map."""
    children_map = defaultdict(list)
    for v in vertices:
        for p in parents_map.get(v, []):
            children_map[p].append(v)
    return children_map

def _compute_layer_depth(vertices, parents_map):
    """Compute the layer/depth of each vertex for layered DAGs."""
    depth = {}
    for v in vertices:
        parents = parents_map.get(v, [])
        if not parents:
            depth[v] = 0
        else:
            depth[v] = 1 + max(depth.get(p, 0) for p in parents)
    return depth

def _compute_future_needs(vertices, parents_map, children_map):
    """
    For each vertex, compute:
    - total_uses: how many times it's needed (number of children that need it)
    - future_needs: for each vertex, the set of vertices that will need it in the future
    """
    total_uses = defaultdict(int)
    for v in vertices:
        for p in parents_map.get(v, []):
            total_uses[p] += 1
    
    # For layered DAGs, we can be smarter:
    # A vertex at layer L is needed by vertices at layer L+1
    # Once all vertices at layer L+1 are processed, the vertex at layer L is no longer needed
    return total_uses

def _compute_recompute_cost(vertices, parents_map, input_set):
    """
    Compute the cost to recompute each node.
    For inputs: cost = 1 (just load)
    For computed nodes: cost = 1 (store) + sum of costs to get parents
    But we use a simpler proxy based on the DAG structure.
    """
    recompute_cost = {}
    for v in vertices:
        if v in input_set:
            recompute_cost[v] = 1  # Just a load
        else:
            # Cost to recompute: 1 (store) + number of parents (each might need a load)
            # For layered DAGs, this is typically small
            recompute_cost[v] = 1 + len(parents_map.get(v, []))
    return recompute_cost

def _greedy_pebble_optimized(dag, fast_memory_budget, deadline):
    """
    Optimized greedy pebbling for layered DAGs.
    
    Key improvements:
    1. Process in topological order (layer by layer for layered DAGs)
    2. Keep vertices in memory only as long as they're needed
    3. Use a cost model that accounts for recompute vs load costs
    4. For layered DAGs, we can predict exactly when a vertex is no longer needed
    """
    vertices = list(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [])
    if not outputs:
        children_map = _get_children_map(vertices, parents_map)
        outputs = [v for v in vertices if not children_map[v]]
    
    input_vertices = [v for v in vertices if not parents_map.get(v, [])]
    input_set = set(input_vertices)
    output_set = set(outputs)
    
    # Pre-compute useful information
    children_map = _get_children_map(vertices, parents_map)
    depth = _compute_layer_depth(vertices, parents_map)
    total_uses = _compute_future_needs(vertices, parents_map, children_map)
    recompute_cost = _compute_recompute_cost(vertices, parents_map, input_set)
    
    # For each vertex, track when it's last needed
    # In a layered DAG, a vertex at layer L is needed until all vertices at layer L+1 are processed
    last_needed_layer = {}
    for v in vertices:
        if v in output_set:
            last_needed_layer[v] = float('inf')  # Outputs are always needed until the end
        else:
            # Find the maximum layer of vertices that need v
            max_child_layer = -1
            for c in children_map.get(v, []):
                max_child_layer = max(max_child_layer, depth.get(c, 0))
            last_needed_layer[v] = max_child_layer
    
    fast_memory = set()
    operations = []
    
    def get_eviction_priority(v):
        """
        Priority for eviction: lower priority = better to evict.
        
        We want to evict vertices that:
        1. Are no longer needed (last_needed_layer < current layer)
        2. Are cheap to re-fetch (low recompute_cost)
        3. Have few remaining uses
        """
        # If the vertex is no longer needed, evict it immediately
        current_max_layer = max(depth.values()) if depth else 0
        
        # Priority based on:
        # - Is it still needed? (lower priority if not)
        # - Recompute cost (lower priority if cheap)
        # - Remaining uses (lower priority if few)
        
        still_needed = last_needed_layer.get(v, -1) >= 0
        rc = recompute_cost[v]
        uses = total_uses.get(v, 0)
        
        # We want to keep vertices that are still needed and expensive to recompute
        # So eviction priority is low for those
        if not still_needed:
            return (0, 0, v)  # Evict immediately
        else:
            # Priority = recompute_cost * uses (higher = more valuable to keep)
            # We want to evict the one with the lowest priority
            return (1, -rc * uses, v)
    
    def evict_one():
        if not fast_memory:
            return
        # Find the vertex with the lowest eviction priority (best to evict)
        victim = min(fast_memory, key=get_eviction_priority)
        if victim in input_set:
            operations.append(['delete', victim])
        else:
            operations.append(['store', victim])
        fast_memory.discard(victim)
    
    def ensure_in_memory(v, current_depth):
        if v in fast_memory:
            return
        
        # Make space if needed
        while len(fast_memory) >= fast_memory_budget:
            evict_one()
        
        if v in input_set:
            operations.append(['load', v])
            fast_memory.add(v)
        else:
            # Need to compute v
            # Ensure parents are in memory
            for p in parents_map.get(v, []):
                ensure_in_memory(p, current_depth)
            operations.append(['compute', v])
            fast_memory.add(v)
    
    order = _topo_sort(vertices, parents_map)
    
    for v in order:
        if time.time() > deadline:
            break
            
        current_depth = depth.get(v, 0)
        ensure_in_memory(v, current_depth)
        
        # After using v, check if any of its parents are no longer needed
        # and can be evicted to free space
        for p in parents_map.get(v, []):
            # Check if p is no longer needed
            # p is no longer needed if all its children have been processed
            # For now, we'll handle this in the eviction logic
            pass
        
        # Evict vertices that are no longer needed
        # This is a simplification: we evict based on the priority function
        # which considers whether the vertex is still needed
        while len(fast_memory) > fast_memory_budget:
            evict_one()
    
    # Ensure all outputs are in fast memory at the end
    for o in outputs:
        if o not in fast_memory:
            ensure_in_memory(o, depth.get(o, 0))
    
    return operations

def _count_transfers(operations):
    """Count the number of load/store operations (transfers)."""
    return sum(1 for op, v in operations if op in ('load', 'store'))

def _verify_operations(operations, dag, fast_memory_budget):
    """
    Verify the operations are valid.
    Returns True if valid, False otherwise.
    """
    if not operations:
        return True
    
    vertices = set(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [])
    if not outputs:
        children_map = _get_children_map(vertices, parents_map)
        outputs = [v for v in vertices if not children_map[v]]
    output_set = set(outputs)
    input_set = set(v for v in vertices if not parents_map.get(v, []))
    
    in_fast = set()
    computed_stored = set()
    
    for op, v in operations:
        if op == 'load':
            # Can only load inputs or previously computed/stored vertices
            if v not in input_set and v not in computed_stored:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.discard(v)
            computed_stored.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.discard(v)
        elif op == 'compute':
            # All parents must be in fast memory
            for p in parents_map.get(v, []):
                if p not in in_fast:
                    return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
            computed_stored.add(v)
    
    # Check all outputs are in fast memory at the end
    for o in outputs:
        if o not in in_fast:
            return False
    
    return True

def _optimize_operations(operations, dag, fast_memory_budget):
    """
    Try to optimize the operations by removing redundant ones.
    Returns the optimized operations.
    """
    if not operations:
        return operations
    
    vertices = set(dag.keys())
    parents_map = {v: dag[v] for v in vertices}
    outputs = dag.get('_outputs', [])
    if not outputs:
        children_map = _get_children_map(vertices, parents_map)
        outputs = [v for v in vertices if not children_map[v]]
    output_set = set(outputs)
    input_set = set(v for v in vertices if not parents_map.get(v, []))
    
    # Simulate and track state to find redundant operations
    in_fast = set()
    computed_stored = set()
    optimized = []
    
    for op, v in operations:
        if op == 'load':
            # Skip if v is already in fast memory
            if v in in_fast:
                continue
            if v not in input_set and v not in computed_stored:
                # Invalid, keep as is (will fail verification)
                optimized.append([op, v])
                in_fast.add(v)
                continue
            if len(in_fast) >= fast_memory_budget:
                # Need to evict, but we can't do that in optimization
                # Keep the operation
                optimized.append([op, v])
                in_fast.add(v)
                continue
            optimized.append([op, v])
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                # Invalid, skip
                continue
            optimized.append([op, v])
            in_fast.discard(v)
            computed_stored.add(v)
        elif op == 'delete':
            if v not in in_fast:
                # Invalid, skip
                continue
            optimized.append([op, v])
            in_fast.discard(v)
        elif op == 'compute':
            # Check if parents are in fast memory
            parents_ok = all(p in in_fast for p in parents_map.get(v, []))
            if not parents_ok:
                # Invalid, keep as is
                optimized.append([op, v])
                in_fast.add(v)
                computed_stored.add(v)
                continue
            if len(in_fast) >= fast_memory_budget:
                # Need space, but we can't evict in optimization
                # Keep the operation
                optimized.append([op, v])
                in_fast.add(v)
                computed_stored.add(v)
                continue
            optimized.append([op, v])
            in_fast.add(v)
            computed_stored.add(v)
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in in_fast:
            # Need to add load/compute for output
            # This is complex, so we'll just return the original if this happens
            return operations
    
    return optimized

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    try:
        # Try the optimized greedy approach
        result = _greedy_pebble_optimized(dag, fast_memory_budget, deadline)
        
        # Verify the result is valid
        if result and len(result) <= 20000:
            if _verify_operations(result, dag, fast_memory_budget):
                # Try to optimize
                optimized = _optimize_operations(result, dag, fast_memory_budget)
                if optimized and len(optimized) <= 20000:
                    if _verify_operations(optimized, dag, fast_memory_budget):
                        return optimized
                return result
            else:
                # Result is invalid, fall back to a simpler approach
                pass
        
        # Fallback: simple topological order with immediate eviction
        vertices = list(dag.keys())
        parents_map = {v: dag[v] for v in vertices}
        outputs = dag.get('_outputs', [])
        if not outputs:
            children_map = _get_children_map(vertices, parents_map)
            outputs = [v for v in vertices if not children_map[v]]
        
        input_vertices = [v for v in vertices if not parents_map.get(v, [])]
        input_set = set(input_vertices)
        
        fast_memory = set()
        operations = []
        
        def ensure_in_memory_simple(v):
            if v in fast_memory:
                return
            while len(fast_memory) >= fast_memory_budget:
                # Evict the first non-output vertex
                for fm in list(fast_memory):
                    if fm not in set(outputs):
                        if fm in input_set:
                            operations.append(['delete', fm])
                        else:
                            operations.append(['store', fm])
                        fast_memory.discard(fm)
                        break
            if v in input_set:
                operations.append(['load', v])
            else:
                for p in parents_map.get(v, []):
                    ensure_in_memory_simple(p)
                operations.append(['compute', v])
            fast_memory.add(v)
        
        order = _topo_sort(vertices, parents_map)
        for v in order:
            if time.time() > deadline:
                break
            ensure_in_memory_simple(v)
        
        # Ensure outputs are in memory
        for o in outputs:
            if o not in fast_memory:
                ensure_in_memory_simple(o)
        
        if operations and len(operations) <= 20000:
            return operations
        
        return []
    except Exception as e:
        # Last resort: return empty
        return []
# EVOLVE-BLOCK-END
