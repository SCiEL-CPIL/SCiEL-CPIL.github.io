"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    DAG pebbling to minimize transfers.
    
    dag structure:
    - dag['vertices']: list of vertex names
    - dag['parents']: dict vertex -> list of parent vertices (topologically ordered)
    - dag['inputs']: list of input vertices (start blue)
    - dag['outputs']: list of output vertices (must finish blue)
    
    fast_memory_budget: number of red values that can be stored simultaneously
    
    Returns list of [operation, vertex] where operation is load/store/compute/delete
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', []))
        parents = dict(dag.get('parents', {}))
        inputs = list(dag.get('inputs', []))
        outputs = list(dag.get('outputs', []))
    else:
        # Try to handle other formats
        vertices = list(dag.get('vertices', [])) if hasattr(dag, 'get') else []
        parents = dict(dag.get('parents', {})) if hasattr(dag, 'get') else {}
        inputs = list(dag.get('inputs', [])) if hasattr(dag, 'get') else []
        outputs = list(dag.get('outputs', [])) if hasattr(dag, 'get') else []
    
    if not vertices:
        return []
    
    # Build adjacency: children of each vertex
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Compute topological order (should already be given, but let's verify)
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    topo_order = []
    queue = [v for v in vertices if in_degree[v] == 0]
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Cycles or issues, just use given order
        topo_order = vertices
    
    # Identify layers: group vertices by their longest path from inputs
    # For layered DAGs, this gives natural layers
    layer_of = {}
    for v in topo_order:
        if v in inputs:
            layer_of[v] = 0
        else:
            max_parent_layer = -1
            for p in parents.get(v, []):
                if p in layer_of:
                    max_parent_layer = max(max_parent_layer, layer_of[p])
            layer_of[v] = max_parent_layer + 1 if max_parent_layer >= 0 else 0
    
    # Group by layer
    layers = defaultdict(list)
    for v in topo_order:
        layers[layer_of[v]].append(v)
    
    max_layer = max(layers.keys()) if layers else 0
    
    # Count how many times each vertex's value is needed by children
    use_count = {}
    for v in vertices:
        use_count[v] = len(children.get(v, []))
    
    # Determine which vertices to keep in fast memory
    # Strategy: keep values in fast memory if they will be reused AND budget allows
    # For each layer, decide what to keep from previous layers
    
    operations = []
    fast_memory = set()  # vertices currently in fast memory (red)
    blue_set = set(inputs)  # vertices currently blue (available without load)
    
    def can_load(v):
        """Check if we can load v into fast memory"""
        if v in fast_memory:
            return True
        if len(fast_memory) < fast_memory_budget:
            return True
        return False
    
    def load_into_fast(v):
        """Load vertex v into fast memory"""
        if v not in fast_memory:
            if len(fast_memory) >= fast_memory_budget:
                # Need to evict something - evict least useful
                # Simple LRU-like: evict one that won't be used soon
                evict_candidates = [x for x in fast_memory if x != v]
                if evict_candidates:
                    # Evict the one with fewest remaining uses
                    min_uses = float('inf')
                    evict_v = None
                    for x in evict_candidates:
                        # Count future uses
                        remaining = 0
                        for c in children.get(x, []):
                            if c not in blue_set and c not in fast_memory:
                                remaining += 1
                        if remaining < min_uses:
                            min_uses = remaining
                            evict_v = x
                    if evict_v:
                        operations.append(['delete', evict_v])
                        fast_memory.discard(evict_v)
            operations.append(['load', v])
            fast_memory.add(v)
    
    def store_from_fast(v):
        """Store vertex v from fast memory to slow (make it blue)"""
        if v in fast_memory:
            operations.append(['store', v])
            fast_memory.discard(v)
            blue_set.add(v)
    
    # Process layer by layer
    for layer in range(max_layer + 1):
        if layer not in layers:
            continue
        
        layer_vertices = layers[layer]
        
        # Determine which vertices from previous layers we should keep in fast memory
        # A vertex from layer L-1 should be kept if it's used by multiple vertices in layer L
        # and the budget allows
        
        # First, figure out which previous-layer vertices are needed by this layer
        needed_from_prev = set()
        for v in layer_vertices:
            for p in parents.get(v, []):
                if layer_of.get(p, -1) == layer - 1:
                    needed_from_prev.add(p)
        
        # Decide what to keep in fast memory
        # Keep vertices that are used by multiple children in this layer
        keep_in_fast = set()
        if fast_memory_budget > 0:
            for p in needed_from_prev:
                # Count how many children in this layer use p
                num_uses_in_layer = sum(1 for v in layer_vertices if p in parents.get(v, []))
                if num_uses_in_layer > 1:
                    keep_in_fast.add(p)
        
        # Evict vertices not in keep_in_fast and not in current layer
        to_evict = fast_memory - keep_in_fast
        # Also evict vertices that are inputs (they're always available)
        to_evict = to_evict - set(inputs)
        
        # Limit evictions to stay within budget
        for v in to_evict:
            if v in fast_memory:
                operations.append(['delete', v])
                fast_memory.discard(v)
        
        # Process vertices in this layer
        for v in layer_vertices:
            if time.time() > DEADLINE:
                break
            
            # Check if all parents are available
            parent_available = all(
                (p in blue_set) or (p in fast_memory) or (p in inputs)
                for p in parents.get(v, [])
            )
            
            if not parent_available:
                # Need to load some parents
                for p in parents.get(v, []):
                    if p not in blue_set and p not in fast_memory and p not in inputs:
                        # Try to load p
                        if p in keep_in_fast or use_count.get(p, 0) > 1:
                            load_into_fast(p)
                    elif p not in fast_memory and p not in blue_set and p not in inputs:
                        # Load temporarily
                        load_into_fast(p)
            
            # Now compute v
            # First, make sure all parents are in fast memory or blue
            for p in parents.get(v, []):
                if p not in fast_memory and p not in blue_set and p not in inputs:
                    load_into_fast(p)
            
            # Compute v
            operations.append(['compute', v])
            
            # Decide whether to put v in fast memory
            if use_count.get(v, 0) > 1 and fast_memory_budget > 0:
                # v will be used multiple times, keep in fast memory
                if v not in fast_memory:
                    load_into_fast(v)
                # Don't make it blue yet - keep it red in fast memory
            else:
                # v is used once or not at all, make it blue immediately
                if v in fast_memory:
                    store_from_fast(v)
                else:
                    blue_set.add(v)
        
        # After processing layer, clean up fast memory
        # Remove vertices that are no longer needed
        for v in list(fast_memory):
            # Check if v is still needed by any future vertex
            still_needed = False
            for c in children.get(v, []):
                if layer_of.get(c, -1) > layer:
                    still_needed = True
                    break
            if not still_needed:
                operations.append(['delete', v])
                fast_memory.discard(v)
                if v not in blue_set and v not in inputs:
                    # Make it blue since we're deleting from fast
                    blue_set.add(v)
    
    # Ensure all outputs are blue
    for v in outputs:
        if v in fast_memory:
            operations.append(['store', v])
            fast_memory.discard(v)
            blue_set.add(v)
        elif v not in blue_set and v not in inputs:
            # Need to load and store
            if v not in fast_memory:
                load_into_fast(v)
            store_from_fast(v)
    
    # Truncate if too many operations
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations

# EVOLVE-BLOCK-END
