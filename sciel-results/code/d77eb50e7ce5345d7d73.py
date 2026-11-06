"""Pebbling scheduler for layered DAGs - minimizes communication via layer-wise processing."""
# EVOLVE-BLOCK-START

def pebble(dag, fast_memory_budget):
    """
    Schedule pebbling operations for a layered DAG.
    
    Uses layer-by-layer processing to minimize transfers.
    """
    vertices = dag['vertices']
    parents = dag.get('parents', {})
    outputs = set(dag['outputs'])
    inputs = set(dag.get('inputs', []))
    
    n = len(vertices)
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Identify layers: layer[v] = max(parent layers) + 1, inputs are layer 0
    layer = {}
    for v in vertices:
        if v in inputs:
            layer[v] = 0
        else:
            ps = parents.get(v, [])
            if ps:
                layer[v] = max(layer[p] for p in ps) + 1
            else:
                layer[v] = 0
    
    max_layer = max(layer.values()) if layer else 0
    
    # Group vertices by layer
    layers = {}
    for v in vertices:
        l = layer[v]
        if l not in layers:
            layers[l] = []
        layers[l].append(v)
    
    for l in layers:
        layers[l].sort()
    
    ops = []
    fast_mem = set()
    computed = set()
    
    # Load all inputs
    for v in sorted(inputs):
        ops.append(['load', v])
        fast_mem.add(v)
        computed.add(v)
    
    # Process layers 1 through max_layer
    for l in range(1, max_layer + 1):
        layer_vertices = layers.get(l, [])
        
        # Compute all vertices in this layer
        for v in layer_vertices:
            # Ensure all parents are in fast memory
            for p in parents.get(v, []):
                if p not in fast_mem:
                    ops.append(['load', p])
                    fast_mem.add(p)
            ops.append(['compute', v])
            fast_mem.add(v)
            computed.add(v)
        
        # After computing layer l, clean up: delete vertices from earlier layers
        # that are no longer needed (all their children are computed)
        for l2 in range(0, l):
            if l2 not in layers:
                continue
            for v in layers[l2]:
                if v in fast_mem and v not in outputs:
                    # Check if all children of v are computed
                    all_children_done = all(c in computed for c in children.get(v, []))
                    if all_children_done:
                        ops.append(['delete', v])
                        fast_mem.discard(v)
    
    # Store all outputs
    for v in sorted(outputs):
        if v in fast_mem:
            ops.append(['store', v])
            fast_mem.discard(v)
    
    # Clean up any remaining red values
    for v in sorted(fast_mem):
        ops.append(['delete', v])
        fast_mem.discard(v)
    
    return ops

# EVOLVE-BLOCK-END
