"""Communication-minimizing DAG pebbling with bounded fast memory."""
# EVOLVE-BLOCK-START
import time
import heapq
import math
import sys

def _extract_dag_info(dag):
    """Extract n, parents, outputs from the dag object."""
    def _get_attr(dag, name, default=None):
        if hasattr(dag, name):
            return getattr(dag, name)
        if isinstance(dag, dict):
            return dag.get(name, default)
        return default

    n = _get_attr(dag, 'n', None)
    parents = _get_attr(dag, 'parents', None)
    outputs = _get_attr(dag, 'outputs', None)

    if n is None:
        if parents is not None:
            n = len(parents)
        else:
            vertices = _get_attr(dag, 'vertices', None)
            if vertices is not None:
                n = len(vertices)
            else:
                n = 0

    if parents is None:
        parents = [[] for _ in range(n)]

    if outputs is None:
        outputs = []

    return n, parents, list(outputs)

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG with bounded fast memory, minimizing transfers.
    
    Returns list of [operation, vertex].
    """
    deadline = time.time() + 85.0
    
    n, parents, outputs = _extract_dag_info(dag)
    
    if n == 0:
        return []
    
    inputs = [v for v in range(n) if len(parents[v]) == 0]
    input_set = set(inputs)
    output_set = set(outputs)
    
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Compute depth (longest path from input)
    depth = [0] * n
    for v in range(n):
        if parents[v]:
            depth[v] = max(depth[p] for p in parents[v]) + 1
    max_depth = max(depth) if n > 0 else 0
    
    # Check if layered
    is_layered = True
    for v in range(n):
        for p in parents[v]:
            if depth[p] != depth[v] - 1:
                is_layered = False
                break
        if not is_layered:
            break
    
    # Build layers
    layers = [[] for _ in range(max_depth + 1)]
    for v in range(n):
        layers[depth[v]].append(v)
    
    # Strategy: For layered DAGs, process layer by layer.
    # Key insight: to compute layer i, we need layer i-1 in fast memory.
    # After computing layer i, we can free layer i-1.
    # 
    # If each layer fits in fast_memory_budget, this is optimal.
    # If not, we need sub-chunk processing.
    
    # For the general case, use a worklist-based pebbling algorithm.
    
    ops = []
    
    # Try layer-by-layer first if it's layered and layers fit.
    max_layer_size = max((len(l) for l in layers), default=0)
    
    if is_layered and max_layer_size <= fast_memory_budget:
        # Clean layer-by-layer processing
        for v in layers[0]:
            ops.append(['load', v])
        
        for i in range(1, max_depth + 1):
            for v in layers[i]:
                ops.append(['compute', v])
            for v in layers[i-1]:
                ops.append(['delete', v])
        
        for v in outputs:
            ops.append(['store', v])
        
        return ops
    
    # General case: use a priority-based pebbling algorithm.
    # We process vertices in topological order.
    # We maintain fast memory and delete vertices when over budget.
    
    # Use a greedy approach:
    # - Process vertices in topological order.
    # - When we need to compute v, ensure parents are in fast memory.
    # - If a parent is not in fast, load it (if input) or recompute it.
    # - When fast is over budget, delete the vertex least likely to be needed.
    
    # To handle recomputation correctly, we use a stack-based approach.
    
    fast = set()
    
    def ensure_available(v):
        """Ensure vertex v's value is in fast memory."""
        if v in fast:
            return
        if v in input_set:
            ops.append(['load', v])
            fast.add(v)
        else:
            # Need to recompute v. Ensure parents are available first.
            for p in parents[v]:
                ensure_available(p)
            ops.append(['compute', v])
            fast.add(v)
    
    # Process in topological order
    for v in range(n):
        # Ensure all parents are in fast memory
        for p in parents[v]:
            ensure_available(p)
        
        # Compute v if not an input
        if v not in input_set:
            ops.append(['compute', v])
            fast.add(v)
        else:
            fast.add(v)
        
        # Store if output
        if v in output_set:
            ops.append(['store', v])
        
        # Manage fast memory: delete if over budget
        while len(fast) > fast_memory_budget:
            # Delete vertex with fewest remaining children (least needed)
            best_victim = None
            best_score = float('inf')
            for u in fast:
                remaining = sum(1 for c in children[u] if c > v)
                if remaining < best_score:
                    best_score = remaining
                    best_victim = u
            if best_victim is not None:
                ops.append(['delete', best_victim])
                fast.discard(best_victim)
            else:
                break
    
    return ops

# EVOLVE-BLOCK-END
