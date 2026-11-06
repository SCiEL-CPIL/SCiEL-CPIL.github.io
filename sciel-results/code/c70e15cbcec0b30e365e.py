# EVOLVE-BLOCK-START
import time
import heapq

def pebble(dag, fast_memory_budget):
    """
    DAG pebbling to minimize memory transfers.
    
    dag: dict with keys:
      - 'vertices': list of vertex names
      - 'edges': list of (parent, child) tuples
      - 'inputs': list of input vertex names
      - 'outputs': list of output vertex names
      - 'topo_order': list of vertices in topological order
    fast_memory_budget: int, number of slots in fast memory
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s
    
    # Extract DAG structure
    vertices = dag.get('vertices', [])
    edges = dag.get('edges', [])
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    topo_order = dag.get('topo_order', [])
    
    if not topo_order:
        # Try to build topological order
        if vertices:
            adj = {v: [] for v in vertices}
            in_deg = {v: 0 for v in vertices}
            for parent, child in edges:
                adj[parent].append(child)
                in_deg[child] = in_deg.get(child, 0) + 1
            queue = [v for v in vertices if in_deg[v] == 0]
            topo_order = []
            while queue:
                v = queue.pop(0)
                topo_order.append(v)
                for child in adj[v]:
                    in_deg[child] -= 1
                    if in_deg[child] == 0:
                        queue.append(child)
    
    if not topo_order:
        return []
    
    # Build adjacency and in-degree
    adj = {v: [] for v in vertices}
    in_deg = {v: 0 for v in vertices}
    parents = {v: [] for v in vertices}
    for parent, child in edges:
        adj[parent].append(child)
        in_deg[child] = in_deg.get(child, 0) + 1
        parents[child].append(parent)
    
    # Count fan-out (number of children) for each vertex
    fan_out = {v: len(adj[v]) for v in vertices}
    
    # Track state
    fast_mem = set()  # vertices currently in fast memory
    slow_mem = set(inputs)  # vertices that have been stored to slow memory (inputs start here)
    computed = set()  # vertices that have been computed at least once
    
    # For each vertex, track how many of its children still need it
    # This is used to decide if we can delete from fast memory without storing
    remaining_children = {v: len(adj[v]) for v in vertices}
    
    operations = []
    
    # Priority function: higher priority = keep in fast memory longer
    # We want to keep nodes that are used many times (high fan_out) or are outputs
    def priority(v):
        if v in outputs:
            return 1000000 + fan_out[v]
        return fan_out[v] * 10 + remaining_children[v]
        
    def make_space():
        """Evict a vertex from fast memory if necessary to make room."""
        nonlocal fast_mem, operations
        if len(fast_mem) < fast_memory_budget:
            return True
        # Find lowest priority non-output vertex to evict
        # We must keep outputs in fast memory? No, we can store them and reload if needed, 
        # but the task says "all outputs must finish blue". 
        # Usually "blue" means in fast memory. 
        # However, if we store an output, it's in slow. To finish blue, we must load it back at the end.
        # The error "all outputs must be stored" suggests we need to write them to slow memory at some point.
        # Let's assume we can evict outputs if we store them first.
        
        candidates = [x for x in fast_mem if x not in outputs]
        if not candidates:
            # All are outputs. We must evict one.
            # This is bad. We should have kept non-outputs if possible.
            # If we are forced to evict an output, we store it.
            candidates = list(fast_mem)
            
        if not candidates:
            return False
            
        min_p = min(candidates, key=priority)
        
        # If we are evicting a vertex that has not been stored yet, we MUST store it
        if min_p not in slow_mem:
            operations.append(['store', min_p])
            slow_mem.add(min_p)
            
        operations.append(['delete', min_p])
        fast_mem.discard(min_p)
        return True

    # Process in topological order
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        # Check if all parents are available
        # A parent is available if it is in fast_mem OR in slow_mem
        parent_ready = True
        for p in parents[v]:
            if p not in fast_mem and p not in slow_mem:
                parent_ready = False
                break
        
        if not parent_ready:
            # This should not happen if we store parents before deleting them
            # and inputs are in slow_mem.
            # If it happens, we might need to recompute, but for now we skip to avoid infinite loops/errors
            # In a correct implementation, this shouldn't occur.
            continue
        
        # If v is not in fast memory, we need to bring it in
        if v not in fast_mem:
            # Make space
            if not make_space():
                continue
                
            if v in inputs:
                # Load input from slow memory
                operations.append(['load', v])
                fast_mem.add(v)
            elif v in slow_mem:
                # Load from slow memory
                operations.append(['load', v])
                fast_mem.add(v)
            else:
                # Need to compute v
                # Ensure parents are in fast memory? 
                # The problem statement says "Inputs start blue". 
                # "Compute" usually requires parents to be in fast memory.
                # Let's ensure parents are in fast memory before computing.
                for p in parents[v]:
                    if p not in fast_mem:
                        # Load parent into fast memory
                        if not make_space():
                            break
                        operations.append(['load', p])
                        fast_mem.add(p)
                else:
                    operations.append(['compute', v])
                    fast_mem.add(v)
                    computed.add(v)
                    continue
                # If we broke out, skip v
                continue
            
        else:
            # v is already in fast memory, just compute if needed
            if v not in computed:
                # Ensure parents are in fast memory
                parents_in_fast = all(p in fast_mem for p in parents[v])
                if not parents_in_fast:
                    for p in parents[v]:
                        if p not in fast_mem:
                            if not make_space():
                                break
                            operations.append(['load', p])
                            fast_mem.add(p)
                    else:
                        operations.append(['compute', v])
                        computed.add(v)
                    continue
                operations.append(['compute', v])
                computed.add(v)
        
        # Update remaining children for parents
        for p in parents[v]:
            if p in remaining_children:
                remaining_children[p] = max(0, remaining_children[p] - 1)
        
        # Store to slow memory if it will be needed again or is an output
        if v in outputs:
            # Must store outputs
            if v not in slow_mem:
                operations.append(['store', v])
                slow_mem.add(v)
        elif remaining_children[v] > 0:
            # Will be needed by other children, store it
            if v not in slow_mem:
                operations.append(['store', v])
                slow_mem.add(v)
        
        # Delete from fast memory if no longer needed and not an output?
        # Actually, we can delete non-outputs if remaining_children is 0.
        # For outputs, we keep them in fast memory until the end?
        # "all outputs must finish blue" -> They must be in fast memory at the end.
        # So we should NOT delete outputs from fast memory.
        if v not in outputs and remaining_children[v] == 0:
            if v in fast_mem:
                # If it's not in slow_mem, we lose it. But if remaining_children is 0, no one needs it.
                # So we can just delete.
                operations.append(['delete', v])
                fast_mem.discard(v)
    
    # Ensure all outputs are in fast memory at the end (blue)
    for v in outputs:
        if v not in fast_mem:
            if v in slow_mem:
                if not make_space():
                    continue
                operations.append(['load', v])
                fast_mem.add(v)
            else:
                # Should not happen if we stored all outputs
                pass
    
    return operations
# EVOLVE-BLOCK-END
