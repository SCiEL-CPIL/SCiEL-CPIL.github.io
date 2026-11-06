"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def _parse_dag(dag):
    """Parse the DAG structure into a usable format.
    Expected format: dag = (n_vertices, parents_dict, output_vertices)
    or similar. We'll be defensive about the exact format.
    """
    # Try common formats
    if isinstance(dag, dict):
        n = dag.get('n', dag.get('num_vertices', len(dag.get('parents', {}))))
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', dag.get('output_vertices', []))
        return n, parents, outputs
    
    if isinstance(dag, (tuple, list)):
        # Could be (n, parents, outputs) or (n, parents)
        if len(dag) >= 3:
            return dag[0], dag[1], dag[2]
        elif len(dag) == 2:
            return dag[0], dag[1], []
    
    # Fallback: assume it's just a list of parents
    if isinstance(dag, list):
        n = len(dag)
        parents = {i: (list(dag[i]) if i < n else []) for i in range(n)}
        outputs = [n - 1] if n > 0 else []
        return n, parents, outputs
    
    raise ValueError(f"Unrecognized DAG format: {type(dag)}")

def _get_children(parents, n):
    """Build children list from parents dict."""
    children = [[] for _ in range(n)]
    for v, par_list in parents.items():
        for p in par_list:
            if p < n:
                children[p].append(v)
    return children

def _find_cut_layer(parents, n, outputs, children):
    """Find a good cut layer to split the DAG.
    We look for a layer where the number of active vertices (those that have
    been computed but not yet consumed, or that need to be kept) is small.
    
    Strategy: find a topological layer index where the "frontier" size is minimized.
    """
    # Compute topological levels
    level = [0] * n
    for v in range(n):
        par_list = parents.get(v, [])
        if par_list:
            level[v] = max(level[p] for p in par_list) + 1
    
    max_level = max(level) if n > 0 else 0
    
    # For each potential cut after layer L, count how many vertices are "in flight"
    # A vertex is in flight at cut L if level[v] <= L < level[v] + 1 ... 
    # Actually, let's think differently: at cut after processing all vertices with level <= L,
    # we need to keep alive any vertex with level <= L that is a parent of some vertex with level > L
    # AND is not yet consumed by all its children.
    
    best_cut = 0
    best_frontier = n  # fallback
    
    for L in range(max_level + 1):
        # Vertices completed: those with level <= L
        # Vertices not yet started: those with level > L
        # Frontier: completed vertices that have at least one child not yet started
        frontier = set()
        for v in range(n):
            if level[v] <= L:
                # Check if v has any child with level > L
                for c in children[v]:
                    if level[c] > L:
                        frontier.add(v)
                        break
        if len(frontier) < best_frontier:
            best_frontier = len(frontier)
            best_cut = L
    
    return best_cut, best_frontier, level

def _recursive_pebble(vertices, parents, n, fast_memory_budget, children, outputs_set, deadline):
    """Recursively pebble a subset of vertices.
    
    vertices: list of vertex indices to process
    Returns: list of operations
    """
    if time.time() > deadline:
        return []
    
    if not vertices:
        return []
    
    vset = set(vertices)
    
    # Base case: small sub-DAG, do linear pass
    if len(vertices) <= 4 or fast_memory_budget <= 1:
        ops = []
        # Topological sort of the sub-DAG
        sub_parents = {}
        for v in vertices:
            par_list = [p for p in parents.get(v, []) if p in vset]
            sub_parents[v] = par_list
        
        # Simple topological order
        in_degree = {v: len(sub_parents[v]) for v in vertices}
        from collections import deque
        queue = deque([v for v in vertices if in_degree[v] == 0])
        topo = []
        while queue:
            v = queue.popleft()
            topo.append(v)
            for c in children[v]:
                if c in vset:
                    in_degree[c] -= 1
                    if in_degree[c] == 0:
                        queue.append(c)
        
        # Linear pass: load parents, compute, store
        computed = set()
        for v in topo:
            par_list = sub_parents[v]
            # Load all parents that aren't already computed
            for p in par_list:
                if p not in computed:
                    ops.append(['load', p])
            ops.append(['compute', v])
            computed.add(v)
            # Store and delete if not needed later
            # For simplicity in base case, store all
            if v in outputs_set:
                ops.append(['store', v])
            else:
                # Check if v is parent of any remaining vertex
                needs_more = False
                for c in children[v]:
                    if c in vset and c not in computed:
                        needs_more = True
                        break
                if not needs_more:
                    ops.append(['delete', v])
                else:
                    ops.append(['store', v])
        
        return ops
    
    # Recursive case: find a cut
    # Build sub-children
    sub_children = [[] for _ in range(n)]
    for v in vertices:
        for c in children[v]:
            if c in vset:
                sub_children[v].append(c)
    
    # Compute levels within sub-DAG
    sub_level = {}
    for v in vertices:
        par_list = [p for p in parents.get(v, []) if p in vset]
        if par_list:
            sub_level[v] = max(sub_level[p] for p in par_list) + 1
        else:
            sub_level[v] = 0
    
    max_sub_level = max(sub_level.values()) if sub_level else 0
    
    # Find best cut
    best_cut_level = 0
    best_frontier_size = len(vertices)
    
    for L in range(max_sub_level + 1):
        frontier = set()
        for v in vertices:
            if sub_level[v] <= L:
                for c in sub_children[v]:
                    if sub_level[c] > L:
                        frontier.add(v)
                        break
        if len(frontier) < best_frontier_size:
            best_frontier_size = len(frontier)
            best_cut_level = L
    
    # Split vertices
    left_vertices = [v for v in vertices if sub_level[v] <= best_cut_level]
    right_vertices = [v for v in vertices if sub_level[v] > best_cut_level]
    frontier_vertices = [v for v in left_vertices if any(sub_level[c] > best_cut_level for c in sub_children[v])]
    
    # If the cut doesn't help (frontier too large), do linear
    if best_frontier_size > len(vertices) // 2 or not left_vertices or not right_vertices:
        # Fallback: linear pass on this sub-DAG
        ops = []
        sub_parents = {}
        for v in vertices:
            par_list = [p for p in parents.get(v, []) if p in vset]
            sub_parents[v] = par_list
        
        in_degree = {v: len(sub_parents[v]) for v in vertices}
        from collections import deque
        queue = deque([v for v in vertices if in_degree[v] == 0])
        topo = []
        while queue:
            v = queue.popleft()
            topo.append(v)
            for c in children[v]:
                if c in vset:
                    in_degree[c] -= 1
                    if in_degree[c] == 0:
                        queue.append(c)
        
        computed = set()
        for v in topo:
            par_list = sub_parents[v]
            for p in par_list:
                if p not in computed:
                    ops.append(['load', p])
            ops.append(['compute', v])
            computed.add(v)
            if v in outputs_set:
                ops.append(['store', v])
            else:
                needs_more = False
                for c in children[v]:
                    if c in vset and c not in computed:
                        needs_more = True
                        break
                if not needs_more:
                    ops.append(['delete', v])
                else:
                    ops.append(['store', v])
        return ops
    
    # Recursive pebbling
    # Left sub-DAG: pebble it, keeping frontier vertices in fast memory
    left_ops = _recursive_pebble(left_vertices, parents, n, fast_memory_budget, children, outputs_set, deadline)
    
    # Right sub-DAG: pebble it, but frontier vertices are already in fast memory
    right_ops = _recursive_pebble(right_vertices, parents, n, fast_memory_budget, children, outputs_set, deadline)
    
    # Combine: left ops, then right ops
    # The frontier vertices are stored by left, and loaded by right if needed
    # Actually, they should stay in fast memory if budget allows
    ops = left_ops + right_ops
    
    return ops

def _linear_pebble(vertices, parents, n, fast_memory_budget, children, outputs_set):
    """Simple linear topological pebbling."""
    if not vertices:
        return []
    
    vset = set(vertices)
    sub_parents = {}
    for v in vertices:
        par_list = [p for p in parents.get(v, []) if p in vset]
        sub_parents[v] = par_list
    
    from collections import deque
    in_degree = {v: len(sub_parents[v]) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            if c in vset:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
    
    ops = []
    computed = set()
    for v in topo:
        par_list = sub_parents[v]
        for p in par_list:
            if p not in computed:
                ops.append(['load', p])
        ops.append(['compute', v])
        computed.add(v)
        if v in outputs_set:
            ops.append(['store', v])
        else:
            needs_more = False
            for c in children[v]:
                if c in vset and c not in computed:
                    needs_more = True
                    break
            if not needs_more:
                ops.append(['delete', v])
            else:
                ops.append(['store', v])
    
    return ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    try:
        n, parents, outputs = _parse_dag(dag)
        outputs_set = set(outputs)
        children = _get_children(parents, n)
        
        all_vertices = list(range(n))
        
        # Try recursive approach
        ops = _recursive_pebble(all_vertices, parents, n, fast_memory_budget, children, outputs_set, deadline)
        
        if len(ops) == 0:
            # Fallback to linear
            ops = _linear_pebble(all_vertices, parents, n, fast_memory_budget, children, outputs_set)
        
        # Ensure we don't exceed operation limit
        if len(ops) > 20000:
            ops = ops[:20000]
        
        return ops
    
    except Exception:
        # Ultimate fallback: use baseline if available
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            # Last resort: minimal valid schedule
            try:
                n, parents, outputs = _parse_dag(dag)
                children = _get_children(parents, n)
                ops = _linear_pebble(list(range(n)), parents, n, fast_memory_budget, children, set(outputs))
                return ops[:20000]
            except Exception:
                return []
# EVOLVE-BLOCK-END
