"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def _parse_dag(dag):
    """Parse various possible DAG formats into a unified structure."""
    if isinstance(dag, dict):
        parents = dag.get('parents', None)
        outputs = dag.get('outputs', None)
        inputs = dag.get('inputs', None)
        vertices = dag.get('vertices', None)
        n = dag.get('n', None)
        
        if parents is None:
            if isinstance(dag, list):
                parents = [list(p) for p in dag]
                n = len(dag)
                if outputs is None:
                    outputs = [n - 1]
                if inputs is None:
                    inputs = [i for i in range(n) if not parents[i]]
            else:
                return None, None, None, None, None
        else:
            if isinstance(parents, list):
                parents = [list(p) for p in parents]
                n = len(parents)
            elif isinstance(parents, dict):
                n = max(parents.keys()) + 1 if parents else 0
                parents_list = [list(parents.get(i, [])) for i in range(n)]
                parents = parents_list
            
            if outputs is None:
                children = [[] for _ in range(n)]
                for v in range(n):
                    for p in parents[v]:
                        children[p].append(v)
                outputs = [v for v in range(n) if not children[v]]
            
            if inputs is None:
                inputs = [v for v in range(n) if not parents[v]]
        
        return parents, outputs, inputs, vertices, n
    
    elif isinstance(dag, list):
        parents = [list(p) if isinstance(p, (list, tuple)) else [] for p in dag]
        n = len(dag)
        outputs = [n - 1] if n > 0 else []
        inputs = [i for i in range(n) if not parents[i]]
        return parents, outputs, inputs, None, n
    
    return None, None, None, None, None


def _build_children(parents, n):
    """Build children list."""
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    return children


def _compute_depth(parents, n):
    """Compute depth (longest path from any input) for each vertex."""
    depth = [0] * n
    # Topological order is 0..n-1
    for v in range(n):
        if parents[v]:
            depth[v] = max(depth[p] for p in parents[v]) + 1
    return depth


def _layered_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Optimized layer-by-layer pebbling with memory tracking.
    Only transfers vertices that are actually needed and not already in fast memory.
    """
    n = len(parents)
    if n == 0:
        return []
    
    children = _build_children(parents, n)
    depth = _compute_depth(parents, n)
    is_output = set(outputs)
    
    # Group vertices by depth (layer)
    max_depth = max(depth) if depth else 0
    layers = [[] for _ in range(max_depth + 1)]
    for v in range(n):
        layers[depth[v]].append(v)
    
    ops = []
    in_fast = set()
    
    # Precompute last use for each vertex
    last_use = [-1] * n
    for v in range(n):
        if children[v]:
            last_use[v] = max(children[v])
    
    # Process layer by layer
    for d in range(max_depth + 1):
        if time.time() > deadline:
            return None
        
        layer = layers[d]
        if not layer:
            continue
        
        if d == 0:
            # Layer 0: inputs - load them
            for v in layer:
                if v not in in_fast:
                    ops.append(['load', v])
                    in_fast.add(v)
        else:
            # Collect all unique parents of vertices in this layer
            needed_parents = set()
            for v in layer:
                for p in parents[v]:
                    needed_parents.add(p)
            
            # Load only those parents not already in fast memory
            for p in sorted(needed_parents):
                if p not in in_fast:
                    ops.append(['load', p])
                    in_fast.add(p)
            
            # Compute vertices in this layer
            for v in layer:
                if v not in in_fast:
                    ops.append(['compute', v])
                    in_fast.add(v)
            
            # Store vertices in this layer that are not final outputs
            for v in layer:
                if v not in is_output:
                    ops.append(['store', v])
                    in_fast.discard(v)
        
        # Delete vertices from previous layers that are no longer needed
        if d > 0:
            for prev_d in range(d):
                prev_layer = layers[prev_d]
                for v in prev_layer:
                    if v in in_fast and v not in is_output and last_use[v] < d:
                        ops.append(['delete', v])
                        in_fast.discard(v)
    
    # Store any remaining outputs that haven't been stored
    for v in range(n):
        if v in is_output and v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)
    
    # Clean up any remaining non-output vertices
    for v in range(n):
        if v in in_fast and v not in is_output:
            ops.append(['delete', v])
            in_fast.discard(v)
    
    return ops


def _greedy_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Optimized greedy pebbling with better eviction strategy.
    Uses reference counting and evicts based on next-use distance.
    """
    n = len(parents)
    if n == 0:
        return []
    
    children = _build_children(parents, n)
    is_output = set(outputs)
    
    # Precompute next use for each vertex (distance to next child)
    next_use = [float('inf')] * n
    for v in range(n):
        if children[v]:
            next_use[v] = min(children[v])
    
    ops = []
    in_fast = set()
    
    def try_evict():
        """Try to evict a vertex from fast memory. Returns True if successful."""
        if not in_fast:
            return False
        # Find vertex with largest next_use (least urgent) that is not an output
        best = None
        best_nu = -1
        for u in in_fast:
            if u not in is_output:
                nu = next_use[u]
                if nu > best_nu:
                    best_nu = nu
                    best = u
        if best is not None:
            ops.append(['delete', best])
            in_fast.discard(best)
            return True
        # If all are outputs, evict any
        if in_fast:
            u = next(iter(in_fast))
            ops.append(['delete', u])
            in_fast.discard(u)
            return True
        return False
    
    for v in range(n):
        if time.time() > deadline:
            return None
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in in_fast:
                while len(in_fast) >= fast_memory_budget:
                    if not try_evict():
                        break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['load', p])
                    in_fast.add(p)
        
        # Compute v if it has parents
        if parents[v]:
            if v not in in_fast:
                while len(in_fast) >= fast_memory_budget:
                    if not try_evict():
                        break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['compute', v])
                    in_fast.add(v)
        
        # Handle input-output vertices
        if not parents[v] and v in is_output:
            if v not in in_fast:
                while len(in_fast) >= fast_memory_budget:
                    if not try_evict():
                        break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['load', v])
                    in_fast.add(v)
            if v in in_fast:
                ops.append(['store', v])
                in_fast.discard(v)
        
        # Store outputs
        if v in is_output and parents[v] and v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)
        
        # Clean up parents that are no longer needed
        for p in parents[v]:
            if p in in_fast and p not in is_output:
                # Check if p has any remaining children
                remaining_children = [c for c in children[p] if c > v]
                if not remaining_children:
                    ops.append(['delete', p])
                    in_fast.discard(p)
    
    return ops


def _validate_ops(ops, parents, outputs, fast_memory_budget):
    """Validate that operations are legal and complete."""
    n = len(parents)
    in_fast = set()
    stored = set()
    
    for op, v in ops:
        if op == 'load':
            if v in in_fast:
                continue
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.discard(v)
            stored.add(v)
        elif op == 'compute':
            for p in parents[v]:
                if p not in in_fast:
                    return False
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.discard(v)
        else:
            return False
    
    for o in outputs:
        if o not in stored:
            return False
    
    return True


def _count_transfers(ops):
    """Count the number of transfers (load + store operations)."""
    return sum(1 for op, v in ops if op in ('load', 'store'))


def _hierarchical_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Hierarchical pebbling with actual recursive decomposition.
    Splits the DAG at the best cut and processes each part separately.
    """
    n = len(parents)
    if n == 0:
        return []
    
    children = _build_children(parents, n)
    is_output = set(outputs)
    depth = _compute_depth(parents, n)
    max_depth = max(depth) if depth else 0
    
    # Find the best split point minimizing crossing vertices
    best_split = -1
    best_crossing = float('inf')
    
    for d in range(1, max_depth):
        prev_layer = [v for v in range(n) if depth[v] == d - 1]
        curr_layer = [v for v in range(n) if depth[v] == d]
        
        # Count how many prev_layer vertices are needed by curr_layer
        needed = set()
        for v in curr_layer:
            for p in parents[v]:
                if depth[p] < d:
                    needed.add(p)
        
        crossing = len(needed)
        if crossing < best_crossing:
            best_crossing = crossing
            best_split = d
    
    # If no good split found, use greedy
    if best_split <= 0 or best_split >= max_depth:
        return _greedy_pebbling(parents, outputs, fast_memory_budget, deadline)
    
    # Split the DAG at best_split
    part1_vertices = [v for v in range(n) if depth[v] < best_split]
    part2_vertices = [v for v in range(n) if depth[v] >= best_split]
    
    # Vertices in part1 that are needed by part2
    crossing_vertices = set()
    for v in part2_vertices:
        for p in parents[v]:
            if depth[p] < best_split:
                crossing_vertices.add(p)
    
    # Outputs in part1 that need to be stored
    part1_outputs = [v for v in outputs if depth[v] < best_split]
    
    # Process part1: compute all vertices, store crossing vertices and part1 outputs
    ops1 = []
    in_fast1 = set()
    
    for v in part1_vertices:
        if time.time() > deadline:
            return None
        
        for p in parents[v]:
            if p not in in_fast1:
                ops1.append(['load', p])
                in_fast1.add(p)
        
        if parents[v]:
            ops1.append(['compute', v])
            in_fast1.add(v)
        
        # Store if crossing vertex or output
        if v in crossing_vertices or v in part1_outputs:
            ops1.append(['store', v])
            in_fast1.discard(v)
        elif v not in in_fast1:
            pass
        else:
            # Check if still needed
            still_needed = any(c in part1_vertices and depth[c] >= depth[v] for c in children[v])
            if not still_needed and v not in crossing_vertices:
                ops1.append(['delete', v])
                in_fast1.discard(v)
    
    # Process part2: load crossing vertices, compute rest
    ops2 = []
    in_fast2 = set()
    
    for v in part2_vertices:
        if time.time() > deadline:
            return None
        
        for p in parents[v]:
            if p not in in_fast2:
                ops2.append(['load', p])
                in_fast2.add(p)
        
        if parents[v]:
            ops2.append(['compute', v])
            in_fast2.add(v)
        
        if v in is_output:
            ops2.append(['store', v])
            in_fast2.discard(v)
        elif v not in in_fast2:
            pass
        else:
            still_needed = any(c in part2_vertices and depth[c] >= depth[v] for c in children[v])
            if not still_needed:
                ops2.append(['delete', v])
                in_fast2.discard(v)
    
    return ops1 + ops2


def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin before 90s kill
    
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        result = pebble_baseline(dag, fast_memory_budget)
        if result is not None:
            return result
    except Exception:
        pass
    
    parents, outputs, inputs, vertices, n = _parse_dag(dag)
    
    if parents is None or n is None:
        return []
    
    if outputs is None:
        outputs = [n - 1]
    
    if fast_memory_budget is None:
        fast_memory_budget = n
    if fast_memory_budget <= 0:
        fast_memory_budget = 1
    
    best_ops = None
    best_transfers = float('inf')
    
    # Try layered pebbling first (optimized for layered DAGs)
    try:
        ops = _layered_pebbling(parents, outputs, fast_memory_budget, deadline)
        if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
            transfers = _count_transfers(ops)
            if transfers < best_transfers:
                best_transfers = transfers
                best_ops = ops
    except Exception:
        pass
    
    # Try greedy pebbling
    try:
        ops = _greedy_pebbling(parents, outputs, fast_memory_budget, deadline)
        if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
            transfers = _count_transfers(ops)
            if transfers < best_transfers:
                best_transfers = transfers
                best_ops = ops
    except Exception:
        pass
    
    # Try hierarchical pebbling
    try:
        ops = _hierarchical_pebbling(parents, outputs, fast_memory_budget, deadline)
        if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
            transfers = _count_transfers(ops)
            if transfers < best_transfers:
                best_transfers = transfers
                best_ops = ops
    except Exception:
        pass
    
    if best_ops is not None:
        return best_ops
    
    # Last resort: very simple approach
    try:
        ops = []
        in_fast = set()
        children = _build_children(parents, n)
        is_output = set(outputs)
        
        for v in range(n):
            if time.time() > deadline:
                break
            
            for p in parents[v]:
                if p not in in_fast:
                    while len(in_fast) >= fast_memory_budget:
                        evicted = False
                        for u in list(in_fast):
                            if u not in is_output:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                evicted = True
                                break
                        if not evicted:
                            break
                    if len(in_fast) < fast_memory_budget:
                        ops.append(['load', p])
                        in_fast.add(p)
            
            if parents[v]:
                while len(in_fast) >= fast_memory_budget:
                    evicted = False
                    for u in list(in_fast):
                        if u not in is_output:
                            ops.append(['delete', u])
                            in_fast.discard(u)
                            evicted = True
                            break
                    if not evicted:
                        break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['compute', v])
                    in_fast.add(v)
            
            if v in is_output:
                if v not in in_fast and not parents[v]:
                    while len(in_fast) >= fast_memory_budget:
                        evicted = False
                        for u in list(in_fast):
                            if u not in is_output:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                evicted = True
                                break
                        if not evicted:
                            break
                    if len(in_fast) < fast_memory_budget:
                        ops.append(['load', v])
                        in_fast.add(v)
                
                if v in in_fast:
                    ops.append(['store', v])
                    in_fast.discard(v)
            
            for p in parents[v]:
                if p in in_fast and p not in is_output:
                    # Check if p is still needed
                    still_needed = False
                    for c in children[p]:
                        if c > v:
                            still_needed = True
                            break
                    if not still_needed:
                        ops.append(['delete', p])
                        in_fast.discard(p)
        
        if _validate_ops(ops, parents, outputs, fast_memory_budget):
            return ops
    except Exception:
        pass
    
    return []

# EVOLVE-BLOCK-END
