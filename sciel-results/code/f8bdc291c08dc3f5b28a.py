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
    for v in range(n):
        if parents[v]:
            depth[v] = max(depth[p] for p in parents[v]) + 1
    return depth


def _compute_last_use(parents, n):
    """Compute the last vertex index that needs each vertex as a parent."""
    children = _build_children(parents, n)
    last_use = [-1] * n
    for v in range(n):
        if children[v]:
            last_use[v] = max(children[v])
    return last_use, children


def _optimized_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Optimized pebbling using reference counting and strategic eviction.
    Key insight: Keep vertices in fast memory until they're no longer needed,
    and only evict when we must make room. Prioritize evicting vertices
    that won't be needed again (or will be needed far in the future).
    """
    n = len(parents)
    if n == 0:
        return []
    
    children = _build_children(parents, n)
    is_output = set(outputs)
    last_use, children = _compute_last_use(parents, n)
    
    ops = []
    in_fast = set()
    
    def can_evict(u):
        """Check if vertex u can be safely evicted."""
        # Can always evict if it's an output (already stored) or no longer needed
        if u in is_output:
            return True
        return last_use[u] < 0  # Never used as parent
    
    def find_best_evict(current_index):
        """Find the best vertex to evict based on future usage."""
        best = None
        best_score = -1
        for u in in_fast:
            if u in is_output:
                # Outputs should be stored, not evicted
                continue
            lu = last_use[u]
            if lu < 0:
                # Never used again - perfect candidate
                score = float('inf')
            else:
                # Score based on distance to next use
                score = lu - current_index
            if score > best_score:
                best_score = score
                best = u
        if best is not None:
            return best
        # Fallback: evict any non-output
        for u in in_fast:
            if u not in is_output:
                return u
        return None
    
    for v in range(n):
        if time.time() > deadline:
            return None
        
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in in_fast:
                # Need to make room if budget exceeded
                while len(in_fast) >= fast_memory_budget:
                    evict_v = find_best_evict(v)
                    if evict_v is None:
                        break
                    # Store if output, delete otherwise
                    if evict_v in is_output:
                        ops.append(['store', evict_v])
                    else:
                        ops.append(['delete', evict_v])
                    in_fast.discard(evict_v)
                
                if len(in_fast) < fast_memory_budget:
                    ops.append(['load', p])
                    in_fast.add(p)
        
        # Compute v if it has parents
        if parents[v]:
            if v not in in_fast:
                while len(in_fast) >= fast_memory_budget:
                    evict_v = find_best_evict(v)
                    if evict_v is None:
                        break
                    if evict_v in is_output:
                        ops.append(['store', evict_v])
                    else:
                        ops.append(['delete', evict_v])
                    in_fast.discard(evict_v)
                
                if len(in_fast) < fast_memory_budget:
                    ops.append(['compute', v])
                    in_fast.add(v)
        
        # Handle input that is also output
        if not parents[v] and v in is_output:
            if v not in in_fast:
                while len(in_fast) >= fast_memory_budget:
                    evict_v = find_best_evict(v)
                    if evict_v is None:
                        break
                    if evict_v in is_output:
                        ops.append(['store', evict_v])
                    else:
                        ops.append(['delete', evict_v])
                    in_fast.discard(evict_v)
                
                if len(in_fast) < fast_memory_budget:
                    ops.append(['load', v])
                    in_fast.add(v)
            
            if v in in_fast:
                ops.append(['store', v])
                in_fast.discard(v)
        
        # Store outputs that have been computed
        if v in is_output and parents[v] and v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)
        
        # Clean up vertices that are no longer needed
        for p in parents[v]:
            if p in in_fast and p not in is_output:
                if last_use[p] <= v:
                    ops.append(['delete', p])
                    in_fast.discard(p)
    
    # Store any remaining outputs
    for v in range(n):
        if v in is_output and v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)
    
    # Clean up remaining non-outputs
    for v in range(n):
        if v in in_fast and v not in is_output:
            ops.append(['delete', v])
            in_fast.discard(v)
    
    return ops


def _layer_aware_pebbling(parents, outputs, fast_memory_budget, deadline):
    """
    Layer-aware pebbling that processes by depth levels.
    For each layer, loads needed parents, computes the layer, and
    strategically stores/deletes based on future needs.
    """
    n = len(parents)
    if n == 0:
        return []
    
    children = _build_children(parents, n)
    depth = _compute_depth(parents, n)
    max_depth = max(depth) if depth else 0
    is_output = set(outputs)
    
    # Group by depth
    layers = [[] for _ in range(max_depth + 1)]
    for v in range(n):
        layers[depth[v]].append(v)
    
    # Precompute: for each vertex, which future layers need it
    # last_use_layer[v] = max depth of any child of v
    last_use_layer = [-1] * n
    for v in range(n):
        if children[v]:
            last_use_layer[v] = max(depth[c] for c in children[v])
    
    ops = []
    in_fast = set()
    
    for d in range(max_depth + 1):
        if time.time() > deadline:
            return None
        
        layer = layers[d]
        if not layer:
            continue
        
        if d == 0:
            # Inputs: load them
            for v in layer:
                if v not in in_fast:
                    ops.append(['load', v])
                    in_fast.add(v)
        else:
            # Collect unique parents needed for this layer
            needed_parents = set()
            for v in layer:
                for p in parents[v]:
                    needed_parents.add(p)
            
            # Load parents not already in fast memory
            for p in sorted(needed_parents):
                if p not in in_fast:
                    # Make room if needed
                    while len(in_fast) >= fast_memory_budget:
                        # Find vertex to evict: prefer those not needed in future layers
                        best_evict = None
                        best_score = -1
                        for u in in_fast:
                            if u in is_output:
                                continue
                            # Score: how far in the future is it needed?
                            lu = last_use_layer[u]
                            if lu < d:
                                score = float('inf')  # Not needed anymore
                            else:
                                score = lu - d
                            if score > best_score:
                                best_score = score
                                best_evict = u
                        if best_evict is None:
                            # All are outputs or no good candidate
                            for u in in_fast:
                                if u not in is_output:
                                    best_evict = u
                                    break
                        if best_evict is not None:
                            if best_evict in is_output:
                                ops.append(['store', best_evict])
                            else:
                                ops.append(['delete', best_evict])
                            in_fast.discard(best_evict)
                        else:
                            break
                    
                    if len(in_fast) < fast_memory_budget:
                        ops.append(['load', p])
                        in_fast.add(p)
            
            # Compute vertices in this layer
            for v in layer:
                if v not in in_fast:
                    while len(in_fast) >= fast_memory_budget:
                        best_evict = None
                        best_score = -1
                        for u in in_fast:
                            if u in is_output:
                                continue
                            lu = last_use_layer[u]
                            if lu < d:
                                score = float('inf')
                            else:
                                score = lu - d
                            if score > best_score:
                                best_score = score
                                best_evict = u
                        if best_evict is None:
                            for u in in_fast:
                                if u not in is_output:
                                    best_evict = u
                                    break
                        if best_evict is not None:
                            if best_evict in is_output:
                                ops.append(['store', best_evict])
                            else:
                                ops.append(['delete', best_evict])
                            in_fast.discard(best_evict)
                        else:
                            break
                    if len(in_fast) < fast_memory_budget:
                        ops.append(['compute', v])
                        in_fast.add(v)
            
            # Store outputs in this layer
            for v in layer:
                if v in is_output and v in in_fast:
                    ops.append(['store', v])
                    in_fast.discard(v)
            
            # Delete vertices no longer needed after this layer
            for v in list(in_fast):
                if v not in is_output and last_use_layer[v] < d + 1:
                    ops.append(['delete', v])
                    in_fast.discard(v)
    
    # Final cleanup
    for v in range(n):
        if v in is_output and v in in_fast:
            ops.append(['store', v])
            in_fast.discard(v)
    for v in range(n):
        if v in in_fast and v not in is_output:
            ops.append(['delete', v])
            in_fast.discard(v)
    
    return ops


def _validate_ops(ops, parents, outputs, fast_memory_budget):
    """Validate that operations are legal and complete."""
    n = len(parents)
    in_fast = set()
    stored = set()
    computed = set()
    
    for op, v in ops:
        if op == 'load':
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            # Can only load inputs or previously computed/stored vertices
            if not parents[v] or v in computed:
                in_fast.add(v)
            else:
                return False
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
            computed.add(v)
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
    
    # Try optimized pebbling first
    try:
        ops = _optimized_pebbling(parents, outputs, fast_memory_budget, deadline)
        if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
            transfers = _count_transfers(ops)
            if transfers < best_transfers:
                best_transfers = transfers
                best_ops = ops
    except Exception:
        pass
    
    # Try layer-aware pebbling
    try:
        ops = _layer_aware_pebbling(parents, outputs, fast_memory_budget, deadline)
        if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
            transfers = _count_transfers(ops)
            if transfers < best_transfers:
                best_transfers = transfers
                best_ops = ops
    except Exception:
        pass
    
    if best_ops is not None:
        return best_ops
    
    # Last resort: simple topological order with basic memory management
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
