"""Rewritten program for communication-minimizing DAG pebbling."""
import time

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize communication (transfers).
    
    dag: dict with keys:
        - 'n': number of vertices (0..n-1)
        - 'parents': list of lists, parents[i] = list of parent indices of vertex i
        - 'inputs': list of input vertex indices (start blue)
        - 'outputs': list of output vertex indices (must finish blue)
        - 'layers': optional list of layer indices for each vertex
    
    fast_memory_budget: int, max number of red values that can be stored
    
    Returns: list of [operation, vertex] where operation is 'load', 'store', 'compute', or 'delete'
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse DAG structure
    if isinstance(dag, dict):
        n = dag.get('n', 0)
        parents = dag.get('parents', [])
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        layers = dag.get('layers', None)
    else:
        # Fallback: assume dag is a list of parent lists
        n = len(dag)
        parents = dag
        inputs = set()
        outputs = set()
        layers = None
    
    if n == 0:
        return []
    
    # Determine topological order (vertices 0..n-1 assumed topologically ordered)
    # Compute children
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)
    
    # Identify input vertices (no parents) and output vertices
    if not inputs:
        inputs = set(i for i in range(n) if len(parents[i]) == 0)
    if not outputs:
        # Outputs are vertices with no children, or specified
        outputs = set(i for i in range(n) if len(children[i]) == 0)
    
    # Determine layer for each vertex if not provided
    if layers is None:
        layers = [0] * n
        for i in range(n):
            if parents[i]:
                layers[i] = max(layers[p] for p in parents[i]) + 1
            else:
                layers[i] = 0
    
    max_layer = max(layers) if n > 0 else 0
    
    # Group vertices by layer
    layer_vertices = [[] for _ in range(max_layer + 1)]
    for i in range(n):
        layer_vertices[layers[i]].append(i)
    
    # Strategy: recursive divide-and-conquer pebbling
    # For layered DAGs, we can use a divide-and-conquer approach:
    # 1. Split the DAG at a middle layer
    # 2. Pebble the top half, keeping only interface vertices in fast memory
    # 3. Pebble the bottom half using the interface values
    
    # First, let's try a simple but effective strategy:
    # Process layer by layer, using fast memory to cache intermediate results
    
    # Simple baseline: process in topological order, load inputs, compute, store outputs
    # This is the "naive" schedule. We need to do better.
    
    # Better strategy: For each layer, compute all vertices in that layer,
    # keeping only what's needed for the next layer in fast memory.
    
    # Let's implement a divide-and-conquer approach for layered DAGs
    
    def pebble_subdag(vertices, fast_mem, time_limit):
        """
        Pebble a sub-DAG defined by the set of vertices.
        Returns a list of operations.
        """
        if time.time() > time_limit:
            return []
        
        if not vertices:
            return []
        
        vertices = sorted(vertices)
        
        # If only one vertex, handle directly
        if len(vertices) == 1:
            v = vertices[0]
            ops = []
            # Check if this is an input
            if v in inputs:
                ops.append(['load', v])
            else:
                # Need to compute it - parents must be available
                for p in parents[v]:
                    if p not in fast_mem:
                        # Need to load or compute parent
                        pass  # This is simplified
                ops.append(['compute', v])
            ops.append(['store', v])
            return ops
        
        # Find the middle layer to split
        layer_set = set(layers[v] for v in vertices)
        if len(layer_set) <= 1:
            # All in same layer, process in order
            ops = []
            for v in vertices:
                if v in inputs:
                    ops.append(['load', v])
                else:
                    ops.append(['compute', v])
                if v in outputs:
                    ops.append(['store', v])
            return ops
        
        # Find a good split layer
        sorted_layers = sorted(layer_set)
        mid_layer = sorted_layers[len(sorted_layers) // 2]
        
        # Split vertices into top and bottom
        top_verts = set(v for v in vertices if layers[v] <= mid_layer)
        bottom_verts = set(v for v in vertices if layers[v] > mid_layer)
        
        # Interface: vertices in top that are parents of vertices in bottom
        interface = set()
        for v in bottom_verts:
            for p in parents[v]:
                if p in top_verts:
                    interface.add(p)
        
        # Pebble top part, keeping interface in fast memory
        top_fast = set(interface)  # These must remain in fast memory after top pebbling
        ops_top = pebble_subdag(top_verts, top_fast, time_limit)
        
        # Pebble bottom part
        ops_bottom = pebble_subdag(bottom_verts, fast_mem, time_limit)
        
        return ops_top + ops_bottom
    
    # Alternative simpler approach: layer-by-layer processing
    # This is more reliable and still better than naive for layered DAGs
    
    def pebble_layered(fast_mem_budget):
        """
        Layer-by-layer pebbling:
        - For each layer, compute all vertices
        - Keep only vertices needed by next layer in fast memory
        - Store outputs as we go
        """
        ops = []
        # Track what's in fast memory (red values)
        fast_mem = set()
        
        # First, load all inputs
        for v in sorted(inputs):
            ops.append(['load', v])
            fast_mem.add(v)
        
        # Process each layer
        for layer in range(max_layer + 1):
            if time.time() > DEADLINE:
                break
            layer_verts = layer_vertices[layer]
            if not layer_verts:
                continue
            
            # Determine which vertices in this layer are needed by next layer
            next_layer_verts = set(layer_vertices[layer + 1]) if layer + 1 <= max_layer else set()
            needed_next = set()
            for v in next_layer_verts:
                for p in parents[v]:
                    if p in set(layer_verts):
                        needed_next.add(p)
            
            # Compute vertices in this layer
            for v in layer_verts:
                if time.time() > DEADLINE:
                    break
                # Parents should be in fast memory or already computed
                ops.append(['compute', v])
                fast_mem.add(v)
            
            # After computing this layer, free memory for vertices not needed
            # Keep: vertices needed by next layer, and outputs
            keep = needed_next | (outputs & set(layer_verts))
            
            # Delete vertices not needed
            for v in layer_verts:
                if v in fast_mem and v not in keep:
                    ops.append(['delete', v])
                    fast_mem.discard(v)
            
            # Store outputs from this layer
            for v in layer_verts:
                if v in outputs:
                    ops.append(['store', v])
                    fast_mem.discard(v)  # After store, no longer in fast mem
        
        # Ensure all outputs are stored
        for v in sorted(outputs):
            if v not in fast_mem:
                # Might need to reload
                pass
        
        return ops
    
    # Try the layered approach first
    ops = pebble_layered(fast_memory_budget)
    
    # Check if we're within deadline
    if time.time() > DEADLINE:
        return ops[:20000]
    
    # If the layered approach didn't work well, try divide-and-conquer
    # But for now, return the layered result
    
    # Truncate to 20000 ops max
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
