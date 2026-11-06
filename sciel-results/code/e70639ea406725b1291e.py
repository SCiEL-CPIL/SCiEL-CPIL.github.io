"""Rewritten program for Communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import heapq

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store) operations for DAG pebbling.
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    deadline = time.time() + 55  # Safety margin before 60s limit

    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        parents = dag.get('parents', dag.get('edges', dag.get('pred', None)))
        inputs = dag.get('inputs', dag.get('sources', dag.get('input_nodes', None)))
        outputs = dag.get('outputs', dag.get('sinks', dag.get('output_nodes', None)))
    else:
        vertices = None
        parents = None
        inputs = None
        outputs = None

    if vertices is None or parents is None:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    # Normalize to sets/lists
    if isinstance(vertices, (list, tuple, range)):
        vertex_list = list(vertices)
    elif isinstance(vertices, int):
        vertex_list = list(range(vertices))
    else:
        vertex_list = list(vertices)

    num_vertices = len(vertex_list)

    # Normalize parents to dict
    if isinstance(parents, dict):
        parent_dict = dict(parents)
    elif isinstance(parents, (list, tuple)):
        parent_dict = {i: list(p) for i, p in enumerate(parents)}
    else:
        parent_dict = {}

    for v in vertex_list:
        if v not in parent_dict:
            parent_dict[v] = []

    # Normalize inputs and outputs
    if inputs is None:
        inputs = set(v for v in vertex_list if not parent_dict.get(v, []))
    elif isinstance(inputs, (set, list, tuple)):
        inputs = set(inputs)
    else:
        inputs = set(inputs)

    if outputs is None:
        children = {v: [] for v in vertex_list}
        for v in vertex_list:
            for p in parent_dict.get(v, []):
                children[p].append(v)
        outputs = set(v for v in vertex_list if not children[v])
    elif isinstance(outputs, (set, list, tuple)):
        outputs = set(outputs)
    else:
        outputs = set(outputs)

    # Build children map
    children = {v: [] for v in vertex_list}
    for v in vertex_list:
        for p in parent_dict.get(v, []):
            children[p].append(v)

    # Determine needed vertices (ancestors of outputs)
    needed = set(outputs)
    for v in reversed(vertex_list):
        if v in needed:
            for p in parent_dict.get(v, []):
                needed.add(p)

    needed_vertices = [v for v in vertex_list if v in needed]
    needed_set = set(needed_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    # Compute last use index for each vertex
    last_child_idx = {}
    for v in needed_vertices:
        last_child_idx[v] = -1
        for child in children.get(v, []):
            if child in needed_set:
                last_child_idx[v] = max(last_child_idx[v], vertex_to_idx.get(child, -1))

    # Compute first use index for each vertex
    first_child_idx = {}
    for v in needed_vertices:
        first_child_idx[v] = num_vertices
        for child in children.get(v, []):
            if child in needed_set:
                first_child_idx[v] = min(first_child_idx[v], vertex_to_idx.get(child, num_vertices))

    # For small problems, use A* search with bitset compression
    if fast_memory_budget <= 8 and len(needed_vertices) <= 40:
        result = _astar_pebble(needed_vertices, parent_dict, children, inputs, outputs,
                               fast_memory_budget, deadline, vertex_to_idx, needed_set,
                               last_child_idx, first_child_idx)
        if result is not None:
            return result

    # Fallback to improved greedy
    return _greedy_pebble(needed_vertices, parent_dict, children, inputs, outputs,
                          fast_memory_budget, deadline, vertex_to_idx, needed_set,
                          last_child_idx, first_child_idx)

def _astar_pebble(needed_vertices, parent_dict, children, inputs, outputs,
                  fast_memory_budget, deadline, vertex_to_idx, needed_set,
                  last_child_idx, first_child_idx):
    """
    A* search with bitset compression for optimal pebbling.
    State: (position, fast_mem_bits, computed_bits)
    Uses admissible heuristic based on minimum remaining loads.
    """
    n = len(needed_vertices)
    if n == 0:
        return []

    v_to_local = {v: i for i, v in enumerate(needed_vertices)}
    local_parents = {i: [v_to_local[p] for p in parent_dict.get(needed_vertices[i], []) if p in needed_set] for i in range(n)}
    local_children = {i: [] for i in range(n)}
    for i in range(n):
        for c in children.get(needed_vertices[i], []):
            if c in needed_set:
                local_children[i].append(v_to_local[c])

    local_inputs = set(v_to_local[v] for v in inputs if v in needed_set)
    local_outputs = set(v_to_local[v] for v in outputs if v in needed_set)

    # Precompute heuristic: for each position, minimum loads needed
    # This is an admissible heuristic: for each uncomputed vertex from pos onward,
    # count how many times it must be loaded (at least once if not an input)
    heuristic_cache = {}
    
    def compute_heuristic(pos, computed_bits):
        """
        Admissible heuristic: lower bound on remaining cost.
        For each vertex from pos to n-1 that is not computed:
        - If it's an input: 0 cost (already in slow)
        - Otherwise: at least 1 load needed
        But we can be smarter: if a vertex has multiple children, 
        it might need to be loaded multiple times if it gets evicted.
        Simple admissible: count uncomputed non-input vertices from pos onward.
        """
        key = (pos, computed_bits)
        if key in heuristic_cache:
            return heuristic_cache[key]
        
        h = 0
        for i in range(pos, n):
            if not (computed_bits >> i) & 1:
                if i not in local_inputs:
                    h += 1
        heuristic_cache[key] = h
        return h

    # State: (pos, fast_bits, computed_bits)
    # fast_bits: bitmask of vertices in fast memory
    # computed_bits: bitmask of vertices that have been computed (or are inputs)
    
    # Initial state
    init_fast = 0
    init_computed = 0
    for i in local_inputs:
        init_computed |= (1 << i)
    
    init_state = (0, init_fast, init_computed)
    
    # A* with priority queue: (f, g, pos, fast_bits, computed_bits)
    # f = g + h
    dist = {init_state: 0}
    parent = {}
    
    pq = [(compute_heuristic(0, init_computed), 0, 0, init_fast, init_computed)]
    
    best_final_cost = float('inf')
    best_final_state = None
    
    steps = 0
    max_steps = 1000000
    
    while pq:
        if time.time() > deadline:
            break
        if steps >= max_steps:
            break
        steps += 1
        
        f, g, pos, fast_bits, computed_bits = heapq.heappop(pq)
        state = (pos, fast_bits, computed_bits)
        
        if dist.get(state, float('inf')) < g:
            continue
        
        # Check if we've processed all vertices
        if pos >= n:
            # Check if all outputs are in fast memory
            all_outputs_in_fast = True
            for o in local_outputs:
                if not (fast_bits >> o) & 1:
                    all_outputs_in_fast = False
                    break
            if all_outputs_in_fast:
                if g < best_final_cost:
                    best_final_cost = g
                    best_final_state = state
                break
            continue
        
        v_local = pos
        
        # Generate successor states
        # Action 1: Compute v_local if all parents are in fast_mem
        parents_local = local_parents.get(v_local, [])
        all_parents_in_fast = True
        for p in parents_local:
            if not (fast_bits >> p) & 1:
                all_parents_in_fast = False
                break
        
        if all_parents_in_fast and not (computed_bits >> v_local) & 1:
            new_computed = computed_bits | (1 << v_local)
            new_fast = fast_bits | (1 << v_local)
            if bin(new_fast).count('1') <= fast_memory_budget:
                new_pos = pos + 1
                new_state = (new_pos, new_fast, new_computed)
                new_g = g  # compute is free
                if dist.get(new_state, float('inf')) > new_g:
                    dist[new_state] = new_g
                    parent[new_state] = (state, 'compute', v_local)
                    new_h = compute_heuristic(new_pos, new_computed)
                    heapq.heappush(pq, (new_g + new_h, new_g, new_pos, new_fast, new_computed))
        
        # Action 2: Load a vertex
        # A vertex can be loaded if it's in inputs or in computed (and not in fast_mem)
        loadable = []
        for u in range(n):
            if not (fast_bits >> u) & 1:
                if u in local_inputs or (computed_bits >> u) & 1:
                    loadable.append(u)
        
        for u in loadable:
            if bin(fast_bits).count('1') < fast_memory_budget:
                new_fast = fast_bits | (1 << u)
                new_state = (pos, new_fast, computed_bits)
                new_g = g + 1  # load costs 1
                if dist.get(new_state, float('inf')) > new_g:
                    dist[new_state] = new_g
                    parent[new_state] = (state, 'load', u)
                    new_h = compute_heuristic(pos, computed_bits)
                    heapq.heappush(pq, (new_g + new_h, new_g, pos, new_fast, computed_bits))
            else:
                # Need to evict one vertex first
                for u_evict in range(n):
                    if (fast_bits >> u_evict) & 1:
                        # If u_evict is in slow, delete is free; otherwise store costs 1
                        if u_evict in local_inputs:
                            evict_cost = 0
                        elif (computed_bits >> u_evict) & 1:
                            evict_cost = 1
                        else:
                            evict_cost = 1
                        
                        new_fast = (fast_bits & ~(1 << u_evict)) | (1 << u)
                        new_state = (pos, new_fast, computed_bits)
                        new_g = g + 1 + evict_cost
                        if dist.get(new_state, float('inf')) > new_g:
                            dist[new_state] = new_g
                            parent[new_state] = (state, f'load_evict_{u_evict}', u)
                            new_h = compute_heuristic(pos, computed_bits)
                            heapq.heappush(pq, (new_g + new_h, new_g, pos, new_fast, computed_bits))
        
        # Action 3: Store a vertex in fast_mem that is in computed
        for u in range(n):
            if (fast_bits >> u) & 1:
                if (computed_bits >> u) & 1 and u not in local_inputs:
                    new_fast = fast_bits & ~(1 << u)
                    new_state = (pos, new_fast, computed_bits)
                    new_g = g + 1  # store costs 1
                    if dist.get(new_state, float('inf')) > new_g:
                        dist[new_state] = new_g
                        parent[new_state] = (state, 'store', u)
                        new_h = compute_heuristic(pos, computed_bits)
                        heapq.heappush(pq, (new_g + new_h, new_g, pos, new_fast, computed_bits))
        
        # Action 4: Delete a vertex in fast_mem that is in slow (inputs)
        for u in range(n):
            if (fast_bits >> u) & 1:
                if u in local_inputs:
                    new_fast = fast_bits & ~(1 << u)
                    new_state = (pos, new_fast, computed_bits)
                    new_g = g  # delete is free
                    if dist.get(new_state, float('inf')) > new_g:
                        dist[new_state] = new_g
                        parent[new_state] = (state, 'delete', u)
                        new_h = compute_heuristic(pos, computed_bits)
                        heapq.heappush(pq, (new_g + new_h, new_g, pos, new_fast, computed_bits))
    
    if best_final_state is None:
        return None
    
    # Reconstruct path
    ops = []
    state = best_final_state
    while state in parent:
        prev_state, action, vertex_local = parent[state]
        if 0 <= vertex_local < n:
            original_vertex = needed_vertices[vertex_local]
        else:
            original_vertex = vertex_local
        
        if action == 'compute':
            ops.append(['compute', original_vertex])
        elif action == 'load':
            ops.append(['load', original_vertex])
        elif action.startswith('load_evict_'):
            evict_local = int(action.split('_')[2])
            evict_vertex = needed_vertices[evict_local]
            if evict_local in local_inputs:
                ops.append(['delete', evict_vertex])
            else:
                ops.append(['store', evict_vertex])
            ops.append(['load', original_vertex])
        elif action == 'store':
            ops.append(['store', original_vertex])
        elif action == 'delete':
            ops.append(['delete', original_vertex])
        
        state = prev_state
    
    ops.reverse()
    
    # Verify the solution
    if _verify_solution(ops, needed_vertices, parent_dict, children, inputs, outputs, 
                        fast_memory_budget, needed_set):
        return ops
    else:
        return None

def _verify_solution(ops, needed_vertices, parent_dict, children, inputs, outputs,
                     fast_memory_budget, needed_set):
    """Verify that a solution is valid."""
    in_fast = set()
    in_slow = set(inputs)
    computed = set(inputs)
    
    for op, v in ops:
        if op == 'load':
            if v not in in_slow:
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.discard(v)
            in_slow.add(v)
        elif op == 'compute':
            parents = [p for p in parent_dict.get(v, []) if p in needed_set]
            if not all(p in in_fast for p in parents):
                return False
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
            computed.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.discard(v)
    
    # Check all outputs are in fast memory
    for o in outputs:
        if o in needed_set and o not in in_fast:
            return False
    
    return True

def _greedy_pebble(needed_vertices, parent_dict, children, inputs, outputs,
                   fast_memory_budget, deadline, vertex_to_idx, needed_set,
                   last_child_idx, first_child_idx):
    """Improved greedy pebbling with lookahead."""
    in_fast = set()
    in_slow = set(inputs)
    computed = set(inputs)
    operations = []

    def load(v):
        operations.append(['load', v])
        in_fast.add(v)
        
    def store(v):
        operations.append(['store', v])
        in_fast.discard(v)
        in_slow.add(v)
        
    def compute(v):
        operations.append(['compute', v])
        computed.add(v)
        in_fast.add(v)
        
    def delete(v):
        operations.append(['delete', v])
        in_fast.discard(v)

    # Process vertices in topological order
    for v in needed_vertices:
        if time.time() > deadline:
            break

        parents = [p for p in parent_dict.get(v, []) if p in needed_set]
        v_idx = vertex_to_idx.get(v, -1)

        # Ensure parents are in fast memory
        for p in parents:
            if p in in_fast:
                continue
            
            # Need to load parent p
            if len(in_fast) >= fast_memory_budget:
                # Evict: Find best candidate to remove using improved heuristic
                best_evict = None
                best_score = -float('inf')
                best_is_slow = False
                
                for u in list(in_fast):
                    # Score: how good is it to evict u?
                    # Higher score = better to evict
                    score = 0
                    
                    # 1. If u is no longer needed (last use in past), high score
                    u_last = last_child_idx.get(u, -1)
                    if u_last < v_idx:
                        score += 1000
                    
                    # 2. If u is a parent of v, penalize (we need it soon)
                    if u in parents:
                        score -= 500
                    
                    # 3. If u is in slow memory, we can delete for free (good)
                    if u in in_slow:
                        score += 100
                    else:
                        score -= 50  # Must store, costs a transfer
                    
                    # 4. Lookahead: if u is needed for many future vertices, penalize
                    future_uses = 0
                    for child in children.get(u, []):
                        if child in needed_set and vertex_to_idx.get(child, -1) > v_idx:
                            future_uses += 1
                    score -= future_uses * 10
                    
                    if score > best_score:
                        best_score = score
                        best_evict = u
                        best_is_slow = (u in in_slow)

                if best_evict is not None:
                    if best_is_slow:
                        delete(best_evict)
                    else:
                        store(best_evict)

            load(p)

        # Compute v
        if v not in in_fast:
            if v in inputs:
                # Input vertex, just ensure it's in fast
                if v not in in_fast:
                    if len(in_fast) >= fast_memory_budget:
                        # Evict
                        best_evict = None
                        best_score = -float('inf')
                        best_is_slow = False
                        for u in list(in_fast):
                            u_last = last_child_idx.get(u, -1)
                            score = 0
                            if u_last < v_idx:
                                score += 1000
                            if u in in_slow:
                                score += 100
                            else:
                                score -= 50
                            if score > best_score:
                                best_score = score
                                best_evict = u
                                best_is_slow = (u in in_slow)
                        if best_evict is not None:
                            if best_is_slow:
                                delete(best_evict)
                            else:
                                store(best_evict)
                    load(v)
            else:
                compute(v)
        # If v is in in_fast, it was loaded as an input or computed earlier

        # Proactive cleanup: remove vertices that are no longer needed
        # This helps keep memory clean for future loads
        if len(in_fast) > fast_memory_budget * 0.5:  # Only clean up if memory is somewhat full
            for u in list(in_fast):
                u_last = last_child_idx.get(u, -1)
                if u_last < v_idx and u not in outputs:
                    # u is no longer needed
                    if u in in_slow:
                        delete(u)
                    else:
                        store(u)

    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                # Evict something
                best_evict = None
                best_score = -float('inf')
                best_is_slow = False
                for u in list(in_fast):
                    if u in outputs:
                        continue
                    u_last = last_child_idx.get(u, -1)
                    score = 0
                    if u_last < num_vertices:
                        score += 1000
                    if u in in_slow:
                        score += 100
                    else:
                        score -= 50
                    if score > best_score:
                        best_score = score
                        best_evict = u
                        best_is_slow = (u in in_slow)
                
                if best_evict is not None:
                    if best_is_slow:
                        delete(best_evict)
                    else:
                        store(best_evict)

            load(v)

    return operations
# EVOLVE-BLOCK-END
