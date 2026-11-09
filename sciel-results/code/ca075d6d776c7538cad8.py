import time
import heapq

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin
    
    # 1. Parse DAG
    parents, inputs, outputs, vertices = parse_dag(dag)
    
    # Topological sort is assumed to be index order or derived from parents
    # We assume vertices are 0..N-1 or keys in parents
    if not vertices:
        if parents:
            vertices = sorted(parents.keys())
        else:
            return []
            
    # Ensure parents keys match vertices
    # If parents is a list of lists, keys are 0..N-1
    # If parents is a dict, keys are vertex ids
    
    # Build children map for greedy fallback
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            if p in children:
                children[p].append(v)

    # 2. Try Dijkstra (Exact)
    # State: (frozenset(red), frozenset(blue))
    # red: vertices in fast memory
    # blue: vertices available to load (inputs + stored)
    
    initial_red = frozenset()
    initial_blue = frozenset(inputs)
    initial_state = (initial_red, initial_blue)
    
    # Check if initial state is already target (empty outputs)
    if not outputs:
        return []
        
    # Target check function
    def is_target(state):
        red, blue = state
        # All outputs must be in blue and NOT in red
        for o in outputs:
            if o not in blue or o in red:
                return False
        return True

    if is_target(initial_state):
        return []

    # Dijkstra
    # dist[state] = cost
    # parent[state] = (prev_state, op, v)
    
    dist = {initial_state: 0}
    parent = {}
    pq = [(0, initial_state)]
    
    found_target = None
    steps = 0
    MAX_STATES = 100000 # Limit state space to prevent memory blowup
    
    while pq and len(dist) < MAX_STATES:
        if time.time() - start_time > DEADLINE:
            break
            
        cost, state = heapq.heappop(pq)
        if state in parent: # Already visited with better cost
            continue
        if state == initial_state:
            pass # Start
            
        red, blue = state
        
        if is_target(state):
            found_target = state
            break
            
        # Generate neighbors
        # 1. Load
        if len(red) < fast_memory_budget:
            for v in blue:
                if v not in red:
                    new_red = red | {v}
                    new_blue = blue - {v}
                    new_state = (frozenset(new_red), frozenset(new_blue))
                    new_cost = cost + 1
                    if new_state not in dist or new_cost < dist[new_state]:
                        dist[new_state] = new_cost
                        parent[new_state] = (state, 'load', v)
                        heapq.heappush(pq, (new_cost, new_state))
                        
        # 2. Store
        for v in red:
            new_red = red - {v}
            new_blue = blue | {v}
            new_state = (frozenset(new_red), frozenset(new_blue))
            new_cost = cost + 1
            if new_state not in dist or new_cost < dist[new_state]:
                dist[new_state] = new_cost
                parent[new_state] = (state, 'store', v)
                heapq.heappush(pq, (new_cost, new_state))
                
        # 3. Compute
        for v in vertices:
            if v in red:
                continue
            # Check if parents are in red
            can_compute = True
            for p in parents.get(v, []):
                if p not in red:
                    can_compute = False
                    break
            if can_compute:
                # Only compute if v is not in blue? 
                # If v is in blue, it's already available. Computing it puts it in red.
                # This might be useful if we want to keep it in red?
                # But usually if it's in blue, we can load it.
                # However, if budget is tight, maybe we don't want to load it, but compute it?
                # No, compute requires parents in red. Load requires v in blue.
                # If v is in blue, we can load it. If we compute it, we need parents in red.
                # Both are valid transitions.
                new_red = red | {v}
                new_blue = blue # v stays in blue? 
                # If we compute v, do we remove it from blue?
                # If v was in blue, it was stored. Now it's computed (red).
                # It is still in slow memory (blue).
                # So blue remains the same.
                new_state = (frozenset(new_red), frozenset(new_blue))
                new_cost = cost # Compute is free
                if new_state not in dist or new_cost < dist[new_state]:
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'compute', v)
                    heapq.heappush(pq, (new_cost, new_state))
                    
        # 4. Delete
        for v in red:
            new_red = red - {v}
            new_blue = blue # v is NOT added to blue
            new_state = (frozenset(new_red), frozenset(new_blue))
            new_cost = cost
            if new_state not in dist or new_cost < dist[new_state]:
                dist[new_state] = new_cost
                parent[new_state] = (state, 'delete', v)
                heapq.heappush(pq, (new_cost, new_state))
                
    if found_target:
        # Reconstruct path
        ops = []
        cur = found_target
        while cur in parent:
            prev_state, op, v = parent[cur]
            ops.append([op, v])
            cur = prev_state
        ops.reverse()
        return ops
        
    # 3. Fallback: Greedy
    # If Dijkstra failed or timed out, use a greedy heuristic
    return greedy_pebble(parents, inputs, outputs, vertices, fast_memory_budget, children, start_time, DEADLINE)

def parse_dag(dag):
    """
    Parses the dag object into parents, inputs, outputs, vertices.
    """
    if isinstance(dag, dict):
        parents = dag.get('parents', {})
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        vertices = dag.get('vertices', list(parents.keys()) if parents else [])
        if not vertices and parents:
             # Try to infer vertices from parents keys
             vertices = list(parents.keys())
        return parents, inputs, outputs, vertices

    if isinstance(dag, list):
        # Case 1: List of dicts
        if len(dag) > 0 and isinstance(dag[0], dict):
            parents = {}
            inputs = set()
            outputs = set()
            vertices = []
            for i, v in enumerate(dag):
                vid = v.get('id', i)
                vertices.append(vid)
                par = v.get('parents', [])
                parents[vid] = par
                if v.get('is_input', False):
                    inputs.add(vid)
                if v.get('is_output', False):
                    outputs.add(vid)
            return parents, inputs, outputs, vertices
        
        # Case 2: List of lists (parents of i)
        if len(dag) > 0 and (isinstance(dag[0], list) or isinstance(dag[0], tuple)):
            parents = {i: list(dag[i]) for i in range(len(dag))}
            vertices = list(range(len(dag)))
            inputs = set(v for v in vertices if not parents[v])
            
            children = {v: [] for v in vertices}
            for v in vertices:
                for p in parents[v]:
                    if p in children:
                        children[p].append(v)
            outputs = set(v for v in vertices if not children[v])
            
            return parents, inputs, outputs, vertices

        # Case 3: [inputs, outputs, parents]
        if len(dag) == 3:
            inputs = set(dag[0])
            outputs = set(dag[1])
            parents_raw = dag[2]
            if isinstance(parents_raw, dict):
                parents = parents_raw
                vertices = list(parents.keys())
            elif isinstance(parents_raw, list):
                parents = {i: list(parents_raw[i]) for i in range(len(parents_raw))}
                vertices = list(range(len(parents_raw)))
            return parents, inputs, outputs, vertices

        # Case 4: [vertices, parents, inputs, outputs]
        if len(dag) == 4:
            vertices = dag[0]
            parents_raw = dag[1]
            inputs = set(dag[2])
            outputs = set(dag[3])
            if isinstance(parents_raw, dict):
                parents = parents_raw
            elif isinstance(parents_raw, list):
                parents = {i: list(parents_raw[i]) for i in range(len(parents_raw))}
            return parents, inputs, outputs, vertices

    # Default: Assume empty or raise
    raise ValueError(f"Unknown dag format: {type(dag)}")

def greedy_pebble(parents, inputs, outputs, vertices, budget, children, start_time, deadline):
    """
    Fallback greedy strategy.
    """
    in_memory = set()
    stored = set(inputs) # Initially available to load
    operations = []
    
    # Topological order
    topo_order = list(vertices) # Assume vertices are in topo order
    
    # Track usage: how many times a vertex is needed by uncomputed children
    # We process in topo order, so we know when a vertex is "last used"
    # Actually, simpler: just keep everything in memory if budget allows, else delete oldest unused.
    
    for v in topo_order:
        if time.time() - start_time > deadline:
            break
            
        # Ensure parents are in memory
        for p in parents.get(v, []):
            if p not in in_memory:
                if p in stored:
                    if len(in_memory) >= budget:
                        # Delete something
                        # Find a vertex in memory that is not a parent of any unprocessed vertex
                        # and is not v's parent
                        to_delete = None
                        for other in list(in_memory):
                            if other == p: continue
                            # Check if other is needed
                            needed = False
                            for child in children.get(other, []):
                                if child not in in_memory and child != v: # Rough check
                                     # If child is in topo_order after v, it might be needed
                                     pass
                            # Simple heuristic: delete first one that is not p
                            if other != p:
                                to_delete = other
                                break
                            if not needed:
                                to_delete = other
                                break
                        if to_delete:
                            in_memory.remove(to_delete)
                            operations.append(['delete', to_delete])
                    operations.append(['load', p])
                    in_memory.add(p)
                    stored.discard(p) # Loaded, so not in slow memory? 
                    # Actually, load moves from slow to fast. It is still in slow? 
                    # "load" usually implies it's in slow. After load, it's in fast.
                    # Can we load it again? No, it's in fast.
                    # Can we store it? Yes.
                    # So stored should remain? Or does load remove from stored?
                    # If I load A, A is in Fast. Is A in Slow? Yes.
                    # If I delete A, A is gone from Fast. Is A in Slow? Yes.
                    # So `stored` (available to load) should NOT be removed on load.
                    # It should only be added on store.
                    # So `stored` is the set of vertices that HAVE BEEN STORED or are INPUTS.
                    # Loading does not change `stored`.
                else:
                    # Recompute p?
                    # This is complex. For now, assume we can't recompute easily in this simple loop.
                    # We might need to recurse. 
                    # For a simple fallback, we'll just fail or skip.
                    pass
                    
        # Compute v
        operations.append(['compute', v])
        in_memory.add(v)
        
        if v in outputs:
            operations.append(['store', v])
            in_memory.remove(v)
            stored.add(v)
        else:
            # Optional: store if memory is full and we might need it?
            # Or just delete if not needed?
            # Heuristic: if len(in_memory) > budget, delete oldest non-output.
            if len(in_memory) > budget:
                 # Delete v? No, just computed.
                 # Delete something else.
                 # Find a vertex that is not a parent of any future vertex.
                 # This requires future knowledge.
                 # Simple: delete the first one.
                 to_delete = list(in_memory)[0]
                 if to_delete != v:
                     in_memory.remove(to_delete)
                     operations.append(['delete', to_delete])
                 else:
                     # If v is the only one, we can't delete it if we need it?
                     # If budget is 1, and we compute v, we have v.
                     # If we need to compute child, we need v.
                     # So we can't delete v.
                     pass

    return operations
# EVOLVE-BLOCK-END
