"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

def pebble(dag, fast_memory_budget):
    """
    DAG pebbling with bounded fast memory.
    
    dag structure (expected):
    - dag['vertices']: list of vertex ids
    - dag['parents']: dict vertex -> list of parent vertices (topologically ordered)
    - dag['inputs']: list of input vertex ids (start blue)
    - dag['outputs']: list of output vertex ids (must finish blue)
    - dag['layers']: optional layer info for layered DAGs
    
    Returns: list of [operation, vertex]
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse the DAG structure - handle multiple possible formats
    if isinstance(dag, dict):
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        inputs = set(dag.get('inputs', []))
        outputs = list(dag.get('outputs', []))
        layers = dag.get('layers', None)
    elif isinstance(dag, (list, tuple)):
        # Maybe dag is (vertices, parents, inputs, outputs)
        if len(dag) >= 4:
            vertices = dag[0]
            parents = dag[1]
            inputs = set(dag[2])
            outputs = list(dag[3])
            layers = dag[4] if len(dag) > 4 else None
        else:
            vertices = list(range(len(dag)))
            parents = {}
            inputs = set()
            outputs = []
            layers = None
    else:
        # Fallback
        vertices = []
        parents = {}
        inputs = set()
        outputs = []
        layers = None
    
    if not vertices:
        return []
    
    # Build topological order
    # parents[v] = list of parents of v (v depends on parents[v])
    # We need to compute in topological order: parents before children
    
    # Build adjacency: for each vertex, its children
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)
    
    # Topological sort (Kahn's algorithm)
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo_order) != len(vertices):
        # Cycle detected, fallback to original order
        topo_order = list(vertices)
    
    # Identify which vertices are inputs, outputs, internal
    input_set = set(inputs)
    output_set = set(outputs)
    all_vertices_set = set(vertices)
    
    # Vertices that are neither inputs nor outputs are computed
    computed_vertices = [v for v in topo_order if v not in input_set and v not in output_set]
    # Outputs that aren't inputs need to be computed too
    output_computed = [v for v in topo_order if v in output_set and v not in input_set]
    
    # Strategy: Layered pebbling
    # For layered DAGs, process layer by layer
    # Each layer: load inputs, compute, store outputs, delete intermediates
    
    # Determine layers if not provided
    if layers is None:
        # Compute layer for each vertex
        layer_of = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                layer_of[v] = 0
            else:
                layer_of[v] = max(layer_of.get(p, 0) for p in ps) + 1
    else:
        if isinstance(layers, dict):
            layer_of = layers
        elif isinstance(layers, (list, tuple)):
            layer_of = {v: l for v, l in zip(vertices, layers)}
        else:
            layer_of = {}
            for v in topo_order:
                ps = parents.get(v, [])
                if not ps:
                    layer_of[v] = 0
                else:
                    layer_of[v] = max(layer_of.get(p, 0) for p in ps) + 1
    
    # Group vertices by layer
    max_layer = max(layer_of.values()) if layer_of else 0
    layer_vertices = {}
    for v in vertices:
        l = layer_of.get(v, 0)
        if l not in layer_vertices:
            layer_vertices[l] = []
        layer_vertices[l].append(v)
    
    # Sort within each layer by topological order
    topo_index = {v: i for i, v in enumerate(topo_order)}
    for l in layer_vertices:
        layer_vertices[l].sort(key=lambda v: topo_index.get(v, 0))
    
    # Pebbling strategy:
    # We maintain a fast memory (set of vertices currently in fast memory)
    # Operations:
    #   load v: bring v from slow to fast (counts as transfer)
    #   store v: move v from fast to slow (counts as transfer)
    #   compute v: compute v in fast memory (v must have all parents in fast)
    #   delete v: remove v from fast memory (no transfer cost, but frees slot)
    
    # Key insight: minimize transfers by keeping frequently-used values in fast memory
    # and only storing/loading when memory is full
    
    # Simple optimal strategy for layered DAGs:
    # Process layer by layer. For each layer:
    # 1. Ensure all parents (from previous layer) that are needed are in fast memory
    # 2. Compute all vertices in current layer
    # 3. Store vertices that won't be needed again
    # 4. Load vertices from previous layers that are needed again
    
    # Track usage: how many times each vertex is needed as a parent
    parent_count = {v: 0 for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            parent_count[p] += 1
    
    # A vertex's last use: the last child in topological order that uses it
    last_use = {}
    for v in topo_order:
        for p in parents.get(v, []):
            last_use[p] = v  # last child that uses p
    
    # Pebbling algorithm
    fast_mem = set()  # vertices currently in fast memory
    ops = []
    
    # Track which vertices have been computed and stored
    computed_and_stored = set()
    loaded = set()  # vertices currently loaded from slow memory
    
    def is_in_fast(v):
        return v in fast_mem
    
    def do_load(v):
        ops.append(['load', v])
        fast_mem.add(v)
        loaded.add(v)
    
    def do_store(v):
        ops.append(['store', v])
        fast_mem.discard(v)
        loaded.discard(v)
        computed_and_stored.add(v)
    
    def do_compute(v):
        ops.append(['compute', v])
        fast_mem.add(v)
    
    def do_delete(v):
        ops.append(['delete', v])
        fast_mem.discard(v)
        loaded.discard(v)
    
    # Process vertices in topological order
    # Maintain fast memory with budget constraint
    
    # For each vertex in topological order:
    # - If it's an input: it starts blue, may need to be in fast memory
    # - If it's computed: ensure parents are in fast memory, then compute
    # - Manage memory: if budget exceeded, store/delete least-useful vertices
    
    # Determine which vertices need to be in fast memory at each step
    # Process in topological order
    
    # Pre-compute: for each vertex, which of its parents are also needed later
    # (i.e., have children after the current vertex in topo order)
    
    for v in topo_order:
        if time.time() - start_time > DEADLINE:
            break
        
        if v in input_set:
            # Input starts blue - it's available in slow memory
            # We'll load it when needed
            continue
        
        # This vertex needs to be computed
        # Ensure all parents are in fast memory
        ps = parents.get(v, [])
        for p in ps:
            if p not in fast_mem:
                # Need to load p
                if len(fast_mem) >= fast_memory_budget:
                    # Need to free space
                    # Find a vertex in fast memory that won't be needed soon
                    # Priority: store vertices that are no longer needed (no more children)
                    # or vertices that can be recomputed
                    freed = False
                    # First try to delete/store vertices that are no longer needed
                    for fm in list(fast_mem):
                        if fm == v:
                            continue
                        # Check if fm is still needed (has children not yet processed)
                        # fm is needed if it has a child in the remaining topo_order
                        still_needed = False
                        for c in children.get(fm, []):
                            if topo_index.get(c, 0) > topo_index.get(v, 0):
                                still_needed = True
                                break
                        if not still_needed:
                            if fm in loaded:
                                do_store(fm)
                            else:
                                do_delete(fm)
                            freed = True
                            if len(fast_mem) < fast_memory_budget:
                                break
                    
                    if len(fast_mem) >= fast_memory_budget:
                        # Still need space - store any vertex
                        for fm in list(fast_mem):
                            if fm != v:
                                if fm in loaded:
                                    do_store(fm)
                                else:
                                    do_delete(fm)
                                if len(fast_mem) < fast_memory_budget:
                                    break
                    
                    if len(fast_mem) >= fast_memory_budget:
                        # Emergency: can't proceed, skip
                        continue
                
                do_load(p)
        
        # Compute v
        if len(fast_mem) >= fast_memory_budget:
            # Need space for v
            for fm in list(fast_mem):
                if fm != v:
                    if fm in loaded:
                        do_store(fm)
                    else:
                        do_delete(fm)
                    if len(fast_mem) < fast_memory_budget:
                        break
        
        do_compute(v)
        
        # If v is an output, it must finish blue - keep it or store it
        # If v is not an output and won't be needed again, we can delete it
        if v not in output_set:
            # Check if v will be needed again
            v_last_use = last_use.get(v)
            if v_last_use is None or topo_index.get(v_last_use, 0) <= topo_index.get(v, 0):
                # v won't be needed again - delete it to free space
                if len(fast_mem) > fast_memory_budget // 2:
                    do_delete(v)
    
    # Ensure all outputs are blue (in fast memory or stored)
    # Outputs must finish blue - they should be in fast memory at the end
    for v in outputs:
        if v not in fast_mem:
            if len(fast_mem) >= fast_memory_budget:
                # Free space
                for fm in list(fast_mem):
                    if fm not in output_set:
                        if fm in loaded:
                            do_store(fm)
                        else:
                            do_delete(fm)
                        if len(fast_mem) < fast_memory_budget:
                            break
            do_load(v)
    
    # Limit operations to 20000
    if len(ops) > 20000:
        ops = ops[:20000]
    
    return ops
# EVOLVE-BLOCK-END
