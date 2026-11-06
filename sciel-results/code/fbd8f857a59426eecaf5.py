import sys
import time
from collections import deque, defaultdict

def pebble(dag, fast_memory_budget):
    """
    Pebble the DAG to minimize transfers.
    
    Args:
        dag: A dictionary or object representing the DAG.
             Expected structure based on typical tasks:
             - 'vertices': list of vertices
             - 'edges': list of (parent, child) or similar
             - 'inputs': list of input vertices
             - 'outputs': list of output vertices
             Or a dict mapping vertex -> list of parents.
             The prompt says "dag has topologically ordered parents and output vertices".
             Let's assume a standard structure: 
             dag = {
                 'vertices': [...],
                 'parents': {v: [p1, p2, ...]}, # or children
                 'inputs': [...],
                 'outputs': [...]
             }
             However, the baseline uses `pebble_baseline`. I need to be careful about the input format.
             Usually, in these generated tasks, `dag` is a dict with keys like 'vertices', 'edges', 'inputs', 'outputs'.
             Let's write a parser that tries to extract topological order and dependencies.
             
             Common formats:
             1. dag['vertices'], dag['edges'] (list of (u,v) where u->v)
             2. dag['parents'] dict
             
             Let's assume a standard interface where we can get:
             - list of all vertices
             - for each vertex, its parents (predecessors)
             - list of inputs
             - list of outputs
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin
    
    # --- 1. Parse DAG ---
    # We need to robustly extract the graph structure.
    # Based on "topologically ordered parents", let's assume we can find parents.
    
    vertices = []
    parents_map = defaultdict(list)
    children_map = defaultdict(list)
    inputs = []
    outputs = []
    
    # Try to infer structure from common keys
    if isinstance(dag, dict):
        if 'vertices' in dag:
            vertices = dag['vertices']
        elif 'nodes' in dag:
            vertices = dag['nodes']
        
        if 'edges' in dag:
            # Edges are usually (parent, child)
            for u, v in dag['edges']:
                parents_map[v].append(u)
                children_map[u].append(v)
        elif 'parents' in dag:
            for v, ps in dag['parents'].items():
                parents_map[v] = ps
                for p in ps:
                    children_map[p].append(v)
        elif 'children' in dag:
            for u, cs in dag['children'].items():
                children_map[u] = cs
                for c in cs:
                    parents_map[c].append(u)
        
        if 'inputs' in dag:
            inputs = dag['inputs']
        elif 'sources' in dag:
            inputs = dag['sources']
            
        if 'outputs' in dag:
            outputs = dag['outputs']
        elif 'sinks' in dag:
            outputs = dag['sinks']
            
        # If inputs/outputs not explicitly given, derive them
        if not inputs:
            inputs = [v for v in vertices if not parents_map[v]]
        if not outputs:
            outputs = [v for v in vertices if not children_map[v]]
            
    else:
        # Fallback: maybe dag is a list of dicts? Unlikely. 
        # Let's assume the dict format is standard.
        raise ValueError("Unsupported DAG format")

    if not vertices:
        return []

    # Ensure all vertices are in the maps
    for v in vertices:
        if v not in parents_map:
            parents_map[v] = []
        if v not in children_map:
            children_map[v] = []

    # --- 2. Topological Sort ---
    in_degree = {v: len(parents_map[v]) for v in vertices}
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo_order = []
    
    # Copy in_degree for sorting
    in_deg = in_degree.copy()
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in children_map[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
                
    if len(topo_order) != len(vertices):
        # Cycle detected? Or disconnected?
        # If cycle, we can't compute. But DAGs are acyclic.
        # If disconnected, the remaining vertices might have dependencies not in 'vertices'?
        # Let's just process what we have.
        # For safety, append remaining if any (should not happen in valid DAG)
        pass

    # --- 3. Heuristic: Cache High Fan-Out Vertices ---
    
    # Calculate fan-out (number of children)
    fan_out = {v: len(children_map[v]) for v in vertices}
    
    # Calculate "last use" in topological order
    # last_use[v] = max(index of u in topo_order such that v is an ancestor of u)
    # Simplification: last_use[v] is the index of the last vertex in topo_order that directly or indirectly depends on v.
    # We can compute this by reverse topological sort.
    last_use = {v: i for i, v in enumerate(topo_order)}
    
    # Update last_use: for each vertex, its last use is at least its own index.
    # If v is a parent of w, last_use[v] = max(last_use[v], last_use[w])
    for v in reversed(topo_order):
        for w in children_map[v]:
            if last_use[w] > last_use[v]:
                last_use[v] = last_use[w]
                
    # Score for caching:
    # We want to cache vertices that are used many times (high fan_out) 
    # and are used "soon" or "long"? 
    # Actually, if a vertex is used many times, caching it saves (fan_out - 1) loads.
    # The cost of caching is 1 slot.
    # So we should cache vertices with highest fan_out.
    # However, if a vertex is only used at the very end, caching it early wastes memory.
    # But since we process in topological order, we can evict it when it's no longer needed.
    # So the "lifespan" in memory is from its computation to its last use.
    # We want to maximize (savings / lifespan) or just savings.
    # Given the constraint of small memory budgets, simple fan-out is a strong predictor.
    
    # Let's sort vertices by fan_out descending.
    # If tie, prefer those with earlier last_use (shorter lifespan, less memory pressure).
    sorted_vertices = sorted(vertices, key=lambda v: (-fan_out[v], last_use[v]))
    
    # Select top K vertices to "pin" (strongly prefer to keep)
    # But we can't pin more than budget.
    # Also, inputs might need to be kept if they have high fan-out.
    
    # Strategy:
    # 1. All outputs must be blue at the end.
    # 2. We maintain a set `in_fast` of vertices currently in fast memory.
    # 3. We maintain a set `pinned` of vertices we strongly want to keep.
    
    # Let's define the "Keep Set" as the top `fast_memory_budget` vertices by fan_out, 
    # excluding inputs (inputs start blue, so they are already "in" if we consider the initial state? 
    # No, the prompt says "Inputs start blue". 
    # "Every red value consumes one fast-memory slot". 
    # "Inputs start blue" -> They are in fast memory initially? 
    # Usually, "blue" means in fast memory (cache). "Red" means in slow memory (disk).
    # So inputs are already in cache.
    # This means our budget is occupied by inputs initially?
    # "fast_memory_budget" is the total capacity.
    # If inputs > budget, we have a problem. 
    # But typically inputs are few or budget is sufficient.
    # Let's assume inputs are in fast memory initially.
    
    # Initial state:
    in_fast = set(inputs)
    # If inputs exceed budget, we must evict some. 
    # But we can't compute without them. 
    # So we must keep them until they are used.
    
    operations = []
    
    # Track next use or just rely on last_use to evict.
    # We iterate through topo_order.
    
    # To allow eviction, we need to know which vertices are "essential" (outputs) 
    # and which are "pinned" (high value).
    
    # Let's refine the Pinned set.
    # We want to keep vertices that are used frequently.
    # Let's select the top `fast_memory_budget` vertices by fan_out as "Preferred".
    # But we must also keep parents of the current vertex.
    
    preferred_set = set(sorted_vertices[:fast_memory_budget])
    
    # If an output is not in preferred_set, it must be kept at the end.
    # We'll handle outputs specifically at the end.
    
    for v in topo_order:
        if time.time() > deadline:
            break
            
        # 1. Ensure parents are in fast memory
        for p in parents_map[v]:
            if p not in in_fast:
                # Load p
                operations.append(["load", p])
                in_fast.add(p)
                # Check memory pressure
                if len(in_fast) > fast_memory_budget:
                    # Evict
                    # We need to evict a vertex that is not v, not a parent of v, 
                    # and not an output (unless we are at the end).
                    # Heuristic: Evict the one with lowest fan_out among non-essential.
                    # Essential: parents of v, v itself (not computed yet), outputs.
                    candidates = []
                    for u in in_fast:
                        if u == v: continue
                        if u in parents_map[v]: continue
                        if u in outputs: continue
                        candidates.append(u)
                    
                    if candidates:
                        # Sort by fan_out asc, then last_use desc (least urgent)
                        candidates.sort(key=lambda u: (fan_out[u], -last_use[u]))
                        evict_v = candidates[0]
                        operations.append(["delete", evict_v])
                        in_fast.remove(evict_v)
        
        # 2. Compute v
        operations.append(["compute", v])
        
        # 3. Decide whether to keep v
        # v is now in fast memory (result of compute)
        in_fast.add(v)
        
        # Check memory pressure again
        if len(in_fast) > fast_memory_budget:
            # We need to evict something.
            # We just added v. We can evict v if it's not useful?
            # Or evict an old parent.
            
            candidates = []
            for u in in_fast:
                # Do not evict v (just computed, might be needed)
                if u == v: continue
                # Do not evict parents of v (already used, but might be needed for other children? 
                # Actually, if we are computing v, we just used p. 
                # If p has other children later, we might need it. 
                # But if we evict p, we'll load it again.
                # So we prefer to evict vertices whose last_use is in the past.
                
                # A vertex u is "dead" if last_use[u] < current_index (index of v)
                # But we don't have current_index easily here without a map.
                
                # Let's use a simpler heuristic:
                # Evict the vertex with the lowest fan_out.
                # If tie, evict the one with the earliest last_use (most "used up").
                
                if u in outputs:
                    # Don't evict outputs unless absolutely necessary (and we are at the end)
                    continue
                
                candidates.append(u)
                
            if candidates:
                candidates.sort(key=lambda u: (fan_out[u], last_use[u]))
                evict_v = candidates[0]
                operations.append(["delete", evict_v])
                in_fast.remove(evict_v)
        
    # 4. Finalize
    # All outputs must be blue.
    for v in outputs:
        if v not in in_fast:
            operations.append(["load", v])
            in_fast.add(v)
            if len(in_fast) > fast_memory_budget:
                # Evict non-output
                candidates = [u for u in in_fast if u not in outputs]
                if candidates:
                    candidates.sort(key=lambda u: (fan_out[u], last_use[u]))
                    evict_v = candidates[0]
                    operations.append(["delete", evict_v])
                    in_fast.remove(evict_v)

    # Return operations
    return operations

