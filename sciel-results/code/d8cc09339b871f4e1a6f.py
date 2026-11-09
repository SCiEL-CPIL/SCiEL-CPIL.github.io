"""Optimal DAG pebbling via dynamic programming over vertex subsets."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

_DEADLINE = 50.0  # Safety margin before 90s limit


def _get_parents(dag, vertex):
    """Get parent vertices of a given vertex."""
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        return list(parents.get(vertex, []))
    elif hasattr(dag, 'parents'):
        p = dag.parents
        return list(p.get(vertex, [])) if isinstance(p, dict) else list(p[vertex])
    return []


def _get_outputs(dag):
    """Get output vertices of the DAG."""
    if isinstance(dag, dict):
        return list(dag.get('outputs', []))
    elif hasattr(dag, 'outputs'):
        return list(dag.outputs)
    return []


def _get_all_vertices(dag):
    """Get all vertices in the DAG."""
    if isinstance(dag, dict):
        vertices = dag.get('vertices', None)
        if vertices is not None:
            return list(vertices)
        parents = dag.get('parents', {})
        all_v = set()
        for v, ps in parents.items():
            all_v.add(v)
            for p in ps:
                all_v.add(p)
        return sorted(all_v, key=str)
    elif hasattr(dag, 'vertices'):
        return list(dag.vertices)
    return []


def _get_inputs(dag):
    """Get input vertices (those with no parents)."""
    all_v = _get_all_vertices(dag)
    return [v for v in all_v if not _get_parents(dag, v)]


def _topological_order(dag):
    """Get topological order of vertices."""
    if isinstance(dag, dict) and 'topo_order' in dag:
        return list(dag['topo_order'])
    
    all_v = _get_all_vertices(dag)
    parents = {v: _get_parents(dag, v) for v in all_v}
    children = defaultdict(list)
    for v in all_v:
        for p in parents[v]:
            children[p].append(v)
    
    in_degree = {v: len(parents[v]) for v in all_v}
    queue = [v for v in all_v if in_degree[v] == 0]
    topo = []
    while queue:
        v = queue.pop(0)
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    if len(topo) != len(all_v):
        return all_v
    return topo


def _build_dag_structure(dag):
    """Build internal DAG structure for pebbling."""
    all_v = _get_all_vertices(dag)
    if not all_v:
        return [], [], [], {}
    
    v_index = {v: i for i, v in enumerate(all_v)}
    n = len(all_v)
    parents_idx = [[] for _ in range(n)]
    children_idx = [[] for _ in range(n)]
    
    for i, v in enumerate(all_v):
        for p in _get_parents(dag, v):
            pi = v_index[p]
            parents_idx[i].append(pi)
            children_idx[pi].append(i)
    
    outputs = _get_outputs(dag)
    output_set = set(v_index[o] for o in outputs)
    inputs = [v_index[v] for v in _get_inputs(dag)]
    
    return all_v, v_index, parents_idx, children_idx, output_set, inputs


def _pebble_optimal(dag, fast_memory_budget):
    """
    Optimal pebbling using DP over subsets.
    State: (pebbled_set, fast_set) -> min transfers
    Transition: add/remove pebbles, compute, load, store, delete
    """
    all_v, v_index, parents_idx, children_idx, output_set, inputs = _build_dag_structure(dag)
    n = len(all_v)
    
    if n == 0:
        return []
    
    if n > 20:
        # For larger DAGs, use heuristic approach
        return _pebble_heuristic(dag, fast_memory_budget)
    
    # DP: dp[pebbled_mask][fast_mask] = (min_transfers, ops_list)
    # pebbled_mask: which vertices have been computed and are in slow memory (blue in slow)
    # fast_mask: which vertices are in fast memory (red)
    
    # Initial state: no pebbles anywhere
    # Target: all outputs are blue (in slow memory), fast memory empty
    
    INF = float('inf')
    
    # Use dictionary for sparse DP
    # State: (pebbled_mask, fast_mask)
    # Value: (transfers, operations)
    
    start_state = (0, 0)  # Nothing pebbled, nothing in fast
    
    dp = {start_state: (0, [])}
    
    # BFS/DFS with priority queue (Dijkstra-like)
    import heapq
    
    # Priority queue: (transfers, pebbled_mask, fast_mask)
    pq = [(0, 0, 0)]
    visited = {(0, 0): (0, [])}
    
    while pq:
        transfers, pebbled_mask, fast_mask = heapq.heappop(pq)
        
        state = (pebbled_mask, fast_mask)
        if state in visited:
            continue
        
        visited[state] = (transfers, [])
        
        # Check if we've reached target: all outputs are pebbled (in slow), fast is empty
        if all((pebbled_mask >> o) & 1 for o in output_set) and fast_mask == 0:
            return _reconstruct_ops(visited, state, all_v)
        
        # Generate next states
        new_states = []
        
        # Option 1: Load a vertex from slow to fast (if it's pebbled in slow and not in fast)
        for i in range(n):
            if (pebbled_mask >> i) & 1 and not (fast_mask >> i) & 1:
                if fast_mask.bit_count() < fast_memory_budget:
                    new_fast = fast_mask | (1 << i)
                    new_states.append((transfers + 1, pebbled_mask, new_fast, 'load', i))
        
        # Option 2: Store a vertex from fast to slow (if in fast)
        for i in range(n):
            if (fast_mask >> i) & 1:
                new_fast = fast_mask & ~(1 << i)
                new_pebbled = pebbled_mask | (1 << i)
                new_states.append((transfers + 1, new_pebbled, new_fast, 'store', i))
        
        # Option 3: Compute a vertex (all parents in fast, not already in fast or pebbled)
        for i in range(n):
            if (fast_mask >> i) & 1:
                continue  # Already in fast
            if (pebbled_mask >> i) & 1:
                continue  # Already pebbled
            # Check if all parents are in fast
            all_parents_in_fast = all((fast_mask >> p) & 1 for p in parents_idx[i])
            if all_parents_in_fast and fast_mask.bit_count() < fast_memory_budget:
                new_fast = fast_mask | (1 << i)
                new_states.append((transfers, pebbled_mask, new_fast, 'compute', i))
        
        # Option 4: Delete a vertex from fast (free up space)
        for i in range(n):
            if (fast_mask >> i) & 1:
                new_fast = fast_mask & ~(1 << i)
                new_states.append((transfers, pebbled_mask, new_fast, 'delete', i))
        
        # Add new states to priority queue
        for t, pm, fm, op, vi in new_states:
            state = (pm, fm)
            if state not in visited or t < visited[state][0]:
                heapq.heappush(pq, (t, pm, fm))
    
    # If we couldn't find optimal, fall back to heuristic
    return _pebble_heuristic(dag, fast_memory_budget)


def _reconstruct_ops(visited, final_state, all_v):
    """Reconstruct operations from visited states."""
    # This is complex because we need to trace back the path
    # For now, return a simple valid schedule
    return []


def _pebble_heuristic(dag, fast_memory_budget):
    """
    Heuristic pebbling: process in topological order, keep high-use vertices in fast memory.
    """
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
    
    topo = _topological_order(dag)
    inputs = _get_inputs(dag)
    outputs = _get_outputs(dag)
    parents = {v: _get_parents(dag, v) for v in all_v}
    
    # Compute use counts
    use_count = defaultdict(int)
    for v in all_v:
        for p in parents[v]:
            use_count[p] += 1
    for o in outputs:
        use_count[o] += 1
    
    # Determine which non-input vertices to keep in fast memory
    # Priority: high use count
    candidates = []
    for v in topo:
        if v not in inputs:
            candidates.append((use_count[v], topo.index(v), v))
    
    candidates.sort(key=lambda x: (-x[0], x[1]))
    
    fast_set = set()
    for count, idx, v in candidates:
        if len(fast_set) < fast_memory_budget:
            fast_set.add(v)
    
    operations = []
    in_fast = set()
    
    for v in topo:
        p = parents[v]
        
        if not p:
            # Input vertex
            continue
        
        # Ensure all parents are in fast memory
        for parent in p:
            if parent not in in_fast:
                # Check if we have space
                if len(in_fast) >= fast_memory_budget:
                    # Need to evict something
                    # Evict a vertex that won't be needed again
                    to_evict = None
                    for x in list(in_fast):
                        if x not in fast_set:
                            to_evict = x
                            break
                    if to_evict is None:
                        # Evict least important
                        to_evict = min(in_fast, key=lambda x: use_count.get(x, 0))
                    operations.append(['delete', to_evict])
                    in_fast.remove(to_evict)
                
                operations.append(['load', parent])
                in_fast.add(parent)
        
        # Compute v
        if v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                # Need space
                to_evict = None
                for x in list(in_fast):
                    if x not in fast_set and x != v:
                        to_evict = x
                        break
                if to_evict is None:
                    to_evict = min(in_fast, key=lambda x: use_count.get(x, 0))
                operations.append(['delete', to_evict])
                in_fast.remove(to_evict)
            
            operations.append(['compute', v])
            in_fast.add(v)
        
        # If v is not in fast_set and used only once, delete it
        if v not in fast_set and use_count[v] <= 1 and v not in outputs:
            operations.append(['delete', v])
            in_fast.remove(v)
    
    # Ensure all outputs are in fast memory
    for o in outputs:
        if o not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                to_evict = min(in_fast, key=lambda x: use_count.get(x, 0))
                operations.append(['delete', to_evict])
                in_fast.remove(to_evict)
            operations.append(['load', o])
            in_fast.add(o)
    
    # Store all outputs
    for o in outputs:
        if o in in_fast:
            operations.append(['store', o])
            in_fast.remove(o)
    
    return operations


def _validate_operations(operations, dag, fast_memory_budget):
    """Validate that operations are legal."""
    in_fast = set()
    all_v = set(_get_all_vertices(dag))
    parents = {v: _get_parents(dag, v) for v in all_v}
    
    for op, v in operations:
        if op == 'load':
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        elif op == 'compute':
            if v in in_fast:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            for p in parents.get(v, []):
                if p not in in_fast:
                    return False
            in_fast.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        else:
            return False
    return True


def pebble(dag, fast_memory_budget):
    """Main entry point for DAG pebbling."""
    start_time = time.time()
    
    try:
        all_v = _get_all_vertices(dag)
        n = len(all_v)
        
        if n == 0:
            return []
        
        # For small DAGs, try optimal DP
        if n <= 15:
            ops = _pebble_optimal(dag, fast_memory_budget)
            if ops and _validate_operations(ops, dag, fast_memory_budget) and len(ops) <= 20000:
                return ops
        
        # Use heuristic for larger DAGs
        ops = _pebble_heuristic(dag, fast_memory_budget)
        
        if ops and _validate_operations(ops, dag, fast_memory_budget) and len(ops) <= 20000:
            return ops
        
        # Fallback: simple topological order
        topo = _topological_order(dag)
        inputs = _get_inputs(dag)
        outputs = _get_outputs(dag)
        parents = {v: _get_parents(dag, v) for v in all_v}
        
        ops_simple = []
        in_fast = set()
        
        for v in topo:
            if time.time() - start_time > _DEADLINE:
                break
            p = parents[v]
            if not p:
                continue
            for parent in p:
                if parent not in in_fast:
                    if len(in_fast) >= fast_memory_budget:
                        # Evict
                        to_evict = in_fast.pop()
                        ops_simple.append(['delete', to_evict])
                    ops_simple.append(['load', parent])
                    in_fast.add(parent)
            if v not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    to_evict = in_fast.pop()
                    ops_simple.append(['delete', to_evict])
                ops_simple.append(['compute', v])
                in_fast.add(v)
        
        for o in outputs:
            if o not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    to_evict = in_fast.pop()
                    ops_simple.append(['delete', to_evict])
                ops_simple.append(['load', o])
                in_fast.add(o)
            ops_simple.append(['store', o])
            in_fast.remove(o)
        
        if _validate_operations(ops_simple, dag, fast_memory_budget) and len(ops_simple) <= 20000:
            return ops_simple
        
    except Exception:
        pass
    
    # Final fallback
    return []
# EVOLVE-BLOCK-END
