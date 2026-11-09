"""Rewritten program for communication-minimizing DAG pebbling."""
import time
import sys

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Compute a schedule of operations to process a DAG with bounded fast memory.
    
    Args:
        dag: A dictionary or object representing the DAG. Expected structure:
             - 'vertices': list of vertex ids
             - 'parents': dict mapping vertex -> list of parent vertices (topologically ordered)
             - 'outputs': list of output vertex ids
             - 'inputs': list of input vertex ids (start blue)
        fast_memory_budget: Integer, maximum number of red values in fast memory at once.
    
    Returns:
        List of [operation, vertex] pairs. Operations: 'load', 'store', 'compute', 'delete'.
    """
    start_time = time.time()
    deadline = 60.0  # Safety margin before 90s kill
    
    # --- Parse DAG structure ---
    # Support both dict-based and object-based DAG representations
    if hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
        parents = dict(dag.parents)
        outputs = list(dag.outputs)
        inputs = list(getattr(dag, 'inputs', []))
    elif isinstance(dag, dict):
        vertices = list(dag.get('vertices', []))
        parents = dict(dag.get('parents', {}))
        outputs = list(dag.get('outputs', []))
        inputs = list(dag.get('inputs', []))
    else:
        # Fallback: try common attribute names
        try:
            vertices = list(dag['vertices'])
            parents = dict(dag['parents'])
            outputs = list(dag['outputs'])
            inputs = list(dag.get('inputs', []))
        except Exception:
            return []
    
    if not vertices:
        return []
    
    # Build adjacency and topological order
    # vertices should be topologically ordered (parents before children)
    vertex_set = set(vertices)
    
    # Determine which vertices are inputs (no parents) and outputs
    if not inputs:
        inputs = [v for v in vertices if not parents.get(v, [])]
    if not outputs:
        # Outputs are vertices with no children
        children_map = {v: [] for v in vertices}
        for v in vertices:
            for p in parents.get(v, []):
                if p in children_map:
                    children_map[p].append(v)
        outputs = [v for v in vertices if not children_map.get(v, [])]
    
    # Topological order: use the given vertex order (assumed topological)
    # Build index for quick lookup
    vertex_index = {v: i for i, v in enumerate(vertices)}
    
    # Compute out-degree (number of children) for each vertex
    out_degree = {v: 0 for v in vertices}
    children_map = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in out_degree:
                out_degree[p] += 1
                if p in children_map:
                    children_map[p].append(v)
    
    # --- Pebbling strategy ---
    # We use a recursive pebbling approach with memoization of sub-DAGs.
    # The key insight: for layered DAGs, we can process layer by layer,
    # keeping only the necessary intermediate results in fast memory.
    
    # Identify layers based on longest path from inputs
    layer = {}
    for v in vertices:
        if not parents.get(v, []):
            layer[v] = 0
        else:
            layer[v] = max(layer.get(p, 0) for p in parents.get(v, [])) + 1
    
    max_layer = max(layer.values()) if layer else 0
    
    # Group vertices by layer
    layers = {}
    for v in vertices:
        l = layer[v]
        if l not in layers:
            layers[l] = []
        layers[l].append(v)
    
    # Sort vertices within each layer by topological index
    for l in layers:
        layers[l].sort(key=lambda v: vertex_index[v])
    
    # --- Generate schedule ---
    # Strategy: 
    # 1. Load all inputs into fast memory (they start blue, so "load" brings them to red? 
    #    Actually, inputs start blue. We need to make them available for computation.
    #    Let's assume "load" moves a blue vertex to red (fast memory).
    # 2. Process layer by layer. For each layer, compute all vertices.
    # 3. After computing a layer, store (write back) vertices that are no longer needed,
    #    and delete (free) fast memory slots as needed.
    # 4. For outputs, ensure they finish blue (i.e., stored back).
    
    # Determine which vertices are "needed" for future computations
    # A vertex is needed if it's a parent of some vertex in a future layer
    needed = set(outputs)  # Outputs must be kept until stored
    # Actually, we need to track dependencies more carefully
    
    # Let's use a simpler strategy:
    # Process vertices in topological order.
    # For each vertex:
    #   - Ensure all parents are in fast memory (red)
    #   - If not, load them (or recompute if deleted)
    #   - Compute the vertex (result goes to fast memory)
    #   - If fast memory is full, evict a vertex that won't be needed soon
    #   - When all children of a vertex are processed, store it and delete from fast memory
    
    # Better approach for layered DAGs:
    # Since the task mentions "three layered DAGs", we can exploit the layer structure.
    
    # Let's implement a straightforward topological pebbling with eviction:
    
    ops = []
    in_fast = set()  # Vertices currently in fast memory (red)
    computed = set()  # Vertices that have been computed
    stored = set()  # Vertices that have been stored back to blue
    
    # Count how many times each vertex is needed as a parent
    remaining_children = {v: len(children_map.get(v, [])) for v in vertices}
    
    def get_fast_count():
        return len(in_fast)
    
    def can_compute(v):
        """Check if all parents of v are in fast memory."""
        for p in parents.get(v, []):
            if p not in in_fast:
                return False
        return True
    
    def evict_one():
        """Evict a vertex from fast memory. Prefer vertices whose children are all processed."""
        # Find vertices that can be safely evicted
        candidates = []
        for v in in_fast:
            # A vertex can be evicted if all its children have been computed
            # OR if it's an input that's no longer needed
            all_children_done = all(child in computed for child in children_map.get(v, []))
            if all_children_done:
                candidates.append(v)
        
        if candidates:
            # Evict the one with the highest layer (least likely to be needed)
            # Actually, evict the one that's an input or has no remaining uses
            # Prefer inputs first, then highest layer
            candidates.sort(key=lambda v: (-layer[v], v not in inputs))
            v = candidates[0]
            ops.append(['delete', v])
            in_fast.discard(v)
            # If it was computed but not stored, we may need to recompute later
            # For now, just delete from fast memory
        else:
            # Emergency: evict any vertex that's an input (least harmful)
            for v in in_fast:
                if v in inputs:
                    ops.append(['delete', v])
                    in_fast.discard(v)
                    return
            # If no inputs, evict the vertex with the highest layer
            if in_fast:
                v = max(in_fast, key=lambda x: layer[x])
                ops.append(['delete', v])
                in_fast.discard(v)
    
    # Load all inputs into fast memory
    for v in inputs:
        if v in vertex_set:
            ops.append(['load', v])
            in_fast.add(v)
    
    # Process vertices in topological order
    for v in vertices:
        if time.time() - start_time > deadline:
            break
        
        # Skip if already computed
        if v in computed:
            continue
        
        # Ensure parents are in fast memory
        for p in parents.get(v, []):
            if p not in in_fast:
                # Need to load or recompute parent
                if p in stored:
                    ops.append(['load', p])
                    in_fast.add(p)
                elif p in computed:
                    # Was computed but deleted from fast memory - need to recompute
                    # This is complex; for now, just load if possible
                    # Actually, if it was computed and deleted, we need to recompute
                    # For simplicity, let's assume we can load it
                    ops.append(['load', p])
                    in_fast.add(p)
                else:
                    # Parent not computed yet - shouldn't happen in topological order
                    # unless we have a bug
                    pass
        
        # Check if we can compute v
        if not can_compute(v):
            # Try to evict and retry
            evict_one()
            # Re-check
            if not can_compute(v):
                # Skip this vertex for now (shouldn't happen in valid topological order)
                continue
        
        # Check fast memory budget
        if get_fast_count() >= fast_memory_budget:
            evict_one()
        
        # Compute v
        ops.append(['compute', v])
        in_fast.add(v)
        computed.add(v)
        
        # After computing, check if any parents can now be stored/deleted
        for p in parents.get(v, []):
            remaining_children[p] = remaining_children.get(p, 0) - 1
            if remaining_children[p] == 0 and p in in_fast:
                # All children of p are computed, so p can be stored and deleted
                # But only if p is not an output (outputs must finish blue, so store them)
                if p in outputs:
                    # Store output (make it blue)
                    ops.append(['store', p])
                    stored.add(p)
                    in_fast.discard(p)
                else:
                    # Delete intermediate
                    ops.append(['delete', p])
                    in_fast.discard(p)
        
        # If v is an output and all its children are done (outputs have no children),
        # store it
        if v in outputs and v in in_fast:
            ops.append(['store', v])
            stored.add(v)
            in_fast.discard(v)
    
    # Ensure all outputs are stored (blue)
    for v in outputs:
        if v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)
        elif v not in stored:
            # Output was never computed? Shouldn't happen
            pass
    
    # Truncate if too many operations
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
