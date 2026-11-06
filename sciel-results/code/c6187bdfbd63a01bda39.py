"""Communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Solve DAG pebbling to minimize communication (transfers).
    
    dag: dict with keys:
        - 'n': number of vertices
        - 'parents': list of lists, parents[i] = list of parent indices for vertex i
        - 'inputs': list of input vertex indices
        - 'outputs': list of output vertex indices
    
    fast_memory_budget: max number of vertices that can be in fast memory at once
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    n = dag['n']
    parents = dag['parents']
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    # Build children mapping
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Find topological order (should already be given, but verify)
    # Compute in-degree for topological sort
    in_degree = [len(parents[v]) for v in range(n)]
    topo_order = []
    from collections import deque
    queue = deque()
    for v in range(n):
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
        # Cycle detected or issue, fall back to simple approach
        return _simple_schedule(dag, fast_memory_budget)
    
    # Identify layers based on longest path from any input
    # layer[v] = max distance from any input to v
    layer = [-1] * n
    for v in topo_order:
        if v in inputs:
            layer[v] = 0
        else:
            if parents[v]:
                layer[v] = max(layer[p] for p in parents[v]) + 1
            else:
                layer[v] = 0  # no parents, treat as layer 0
    
    # Group vertices by layer
    max_layer = max(layer) if n > 0 else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in range(n):
        layers[layer[v]].append(v)
    
    # Strategy: process layer by layer
    # For each layer, we need to:
    # 1. Ensure all parents (previous layer) are available
    # 2. Compute current layer vertices
    # 3. Decide which to keep in fast memory vs store to slow
    # 4. Delete vertices that are no longer needed
    
    # Determine which vertices are "needed" (have children that haven't been processed yet)
    # A vertex v is needed if it has at least one child in a future layer
    
    # Track which vertices are in fast memory
    in_fast = set()
    # Track which vertices are in slow memory (stored)
    in_slow = set()
    # Track which vertices have been computed
    computed = set()
    
    # Initialize: inputs are blue (in fast memory)
    for v in inputs:
        in_fast.add(v)
        computed.add(v)
    
    operations = []
    
    # Process each layer
    for L in range(max_layer + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        current_layer = layers[L]
        if not current_layer:
            continue
        
        # For each vertex in current layer, ensure parents are in fast memory
        for v in current_layer:
            if time.time() - start_time > DEADLINE:
                break
            
            # Load parents from slow memory if needed
            for p in parents[v]:
                if p not in in_fast and p in in_slow:
                    # Need to load parent from slow to fast
                    # Check if we have space
                    if len(in_fast) >= fast_memory_budget:
                        # Need to delete something
                        # Delete a vertex that is no longer needed
                        deleted = _find_deletable_vertex(in_fast, computed, children, layer, L, n, inputs, outputs)
                        if deleted is not None:
                            operations.append(['delete', deleted])
                            in_fast.discard(deleted)
                            if deleted in outputs:
                                # If output, must store before delete
                                # Actually, outputs must finish blue, so we can't delete them
                                # This case shouldn't happen if we handle outputs properly
                                pass
                    operations.append(['load', p])
                    in_fast.add(p)
                    in_slow.discard(p)
            
            # Compute vertex v
            if v not in computed:
                operations.append(['compute', v])
                computed.add(v)
                in_fast.add(v)
            # v is now in fast memory (just computed)
        
        # After computing current layer, decide what to keep in fast memory
        # We need to keep:
        # 1. Vertices that are outputs (must finish blue)
        # 2. Vertices that have children in future layers (will be needed)
        # 3. Inputs that still have children (but inputs are already handled)
        
        # Determine which vertices in current layer will be needed in future
        needed_in_future = set()
        for v in current_layer:
            for c in children[v]:
                if layer[c] > L:
                    needed_in_future.add(v)
                    break
        
        # Store vertices that won't be needed and aren't outputs
        for v in current_layer:
            if v in in_fast:
                if v not in needed_in_future and v not in outputs:
                    # Store to slow memory
                    operations.append(['store', v])
                    in_fast.discard(v)
                    in_slow.add(v)
                elif v in needed_in_future and len(in_fast) > fast_memory_budget:
                    # Over budget, need to store some
                    # Store vertices that won't be needed first
                    pass
        
        # If over budget, delete or store excess vertices
        while len(in_fast) > fast_memory_budget:
            # Find a vertex to remove
            removed = False
            for v in list(in_fast):
                if v not in outputs:
                    # Can we delete or store?
                    if v in in_slow:
                        # Already in slow, just delete from fast
                        operations.append(['delete', v])
                        in_fast.discard(v)
                        removed = True
                        break
                    else:
                        # Store to slow
                        operations.append(['store', v])
                        in_fast.discard(v)
                        in_slow.add(v)
                        removed = True
                        break
            if not removed:
                break
        
        # Delete vertices from previous layers that are no longer needed
        for v in list(in_fast):
            if v in inputs:
                continue  # Keep inputs
            if v in outputs:
                continue  # Keep outputs
            # Check if v has any children in future layers
            has_future_child = False
            for c in children[v]:
                if layer[c] > L:
                    has_future_child = True
                    break
            if not has_future_child:
                # No longer needed
                if v in in_slow:
                    operations.append(['delete', v])
                    in_fast.discard(v)
                else:
                    operations.append(['store', v])
                    in_fast.discard(v)
                    in_slow.add(v)
    
    # Final cleanup: ensure all outputs are in fast memory
    for v in outputs:
        if v not in in_fast:
            if v in in_slow:
                operations.append(['load', v])
                in_fast.add(v)
                in_slow.discard(v)
            # If not computed, compute it
            elif v not in computed:
                # Need to compute
                for p in parents[v]:
                    if p not in in_fast and p in in_slow:
                        if len(in_fast) >= fast_memory_budget:
                            # Delete something
                            for d in list(in_fast):
                                if d not in outputs:
                                    operations.append(['delete', d])
                                    in_fast.discard(d)
                                    break
                        operations.append(['load', p])
                        in_fast.add(p)
                        in_slow.discard(p)
                operations.append(['compute', v])
                in_fast.add(v)
                computed.add(v)
    
    # Verify all outputs are in fast memory
    for v in outputs:
        if v not in in_fast:
            # Emergency: load or compute
            if v in in_slow:
                operations.append(['load', v])
                in_fast.add(v)
            elif v not in computed:
                # This shouldn't happen, but handle it
                pass
    
    return operations


def _find_deletable_vertex(in_fast, computed, children, layer, current_layer_idx, n, inputs, outputs):
    """Find a vertex in fast memory that can be deleted (no future children, not an output)."""
    for v in in_fast:
        if v in outputs:
            continue
        if v in inputs:
            continue
        has_future_child = False
        for c in children[v]:
            if layer[c] > current_layer_idx:
                has_future_child = True
                break
        if not has_future_child:
            return v
    return None


def _simple_schedule(dag, fast_memory_budget):
    """Fallback simple schedule: load all, compute in topo order, store outputs."""
    n = dag['n']
    parents = dag['parents']
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    operations = []
    in_fast = set()
    computed = set()
    
    # Initialize inputs
    for v in inputs:
        in_fast.add(v)
        computed.add(v)
    
    # Topological order
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    in_degree = [len(parents[v]) for v in range(n)]
    from collections import deque
    queue = deque()
    for v in range(n):
        if in_degree[v] == 0:
            queue.append(v)
    
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # Process in topological order
    for v in topo_order:
        if v in inputs:
            continue
        
        # Load parents if needed
        for p in parents[v]:
            if p not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    # Delete a non-output vertex
                    for d in list(in_fast):
                        if d not in outputs:
                            operations.append(['delete', d])
                            in_fast.discard(d)
                            break
                operations.append(['load', p])
                in_fast.add(p)
        
        # Compute
        operations.append(['compute', v])
        in_fast.add(v)
        computed.add(v)
        
        # If not an output and no future children, store
        has_future_child = False
        for c in children[v]:
            if c not in computed:
                has_future_child = True
                break
        
        if v not in outputs and not has_future_child:
            operations.append(['store', v])
            in_fast.discard(v)
    
    # Ensure outputs are in fast memory
    for v in outputs:
        if v not in in_fast:
            operations.append(['load', v])
            in_fast.add(v)
    
    return operations
# EVOLVE-BLOCK-END
