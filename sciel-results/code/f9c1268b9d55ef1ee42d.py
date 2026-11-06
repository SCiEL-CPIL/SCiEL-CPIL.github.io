# EVOLVE-BLOCK-START
import time
import sys
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

    # Try optimal Dijkstra-based search for small problems
    if len(needed_vertices) <= 12 and fast_memory_budget <= 4:
        result = _optimal_pebble(needed_vertices, parent_dict, needed_set,
                                 inputs, outputs, fast_memory_budget, deadline)
        if result is not None:
            return result

    # Fallback to improved Greedy FNU approach
    return _greedy_pebble_improved(needed_vertices, parent_dict, needed_set,
                                   inputs, outputs, fast_memory_budget, deadline,
                                   vertex_to_idx, children)


def _optimal_pebble(needed_vertices, parent_dict, needed_set,
                    inputs, outputs, fast_memory_budget, deadline):
    """
    Dijkstra-based search over memory states to find minimum-transfer pebbling.
    """
    init_fast = frozenset()
    init_computed = frozenset()
    init_state = (init_fast, init_computed)

    dist = {init_state: 0}
    parent = {}
    pq = [(0, init_fast, init_computed)]
    
    states_explored = 0
    max_states = 50000
    
    while pq:
        if time.time() > deadline - 2:
            return None
        if states_explored > max_states:
            return None
            
        cost, fast, computed = heapq.heappop(pq)
        
        if cost > dist.get((fast, computed), float('inf')):
            continue
            
        states_explored += 1
        
        if outputs.issubset(fast):
            ops = []
            state = (fast, computed)
            while state in parent:
                prev_state, op, vertex = parent[state]
                ops.append([op, vertex])
                state = prev_state
            ops.reverse()
            return ops
        
        # 1. LOAD
        if len(fast) < fast_memory_budget:
            loadable = (inputs | computed) - fast
            for v in loadable:
                if v not in needed_set:
                    continue
                new_fast = fast | {v}
                new_state = (new_fast, computed)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = ((fast, computed), 'load', v)
                    heapq.heappush(pq, (new_cost, new_fast, computed))
        
        # 2. STORE
        for v in fast:
            if v in inputs or v in computed:
                new_fast = fast - {v}
                new_state = (new_fast, computed)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = ((fast, computed), 'store', v)
                    heapq.heappush(pq, (new_cost, new_fast, computed))
        
        # 3. COMPUTE
        for v in needed_set:
            if v in computed:
                continue
            v_parents = [p for p in parent_dict.get(v, []) if p in needed_set]
            if all(p in fast for p in v_parents):
                if len(fast) < fast_memory_budget:
                    new_fast = fast | {v}
                    new_computed = computed | {v}
                    new_state = (new_fast, new_computed)
                    new_cost = cost
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        parent[new_state] = ((fast, computed), 'compute', v)
                        heapq.heappush(pq, (new_cost, new_fast, new_computed))
        
        # 4. DELETE
        for v in fast:
            if v in computed and v not in inputs:
                new_fast = fast - {v}
                new_state = (new_fast, computed)
                new_cost = cost
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = ((fast, computed), 'delete', v)
                    heapq.heappush(pq, (new_cost, new_fast, computed))
    
    return None


def _greedy_pebble_improved(needed_vertices, parent_dict, needed_set,
                            inputs, outputs, fast_memory_budget, deadline, vertex_to_idx, children):
    """
    Improved Greedy FNU-based pebbling.
    1. Topological Sort.
    2. Precompute next-use positions.
    3. Simulate with proactive deletion and FNU eviction.
    4. Final pass to ensure outputs are in fast memory.
    """
    
    # 1. Topological Sort of needed vertices
    # We only care about needed vertices
    in_degree = {v: 0 for v in needed_vertices}
    for v in needed_vertices:
        for p in parent_dict.get(v, []):
            if p in needed_set:
                in_degree[v] += 1
                
    queue = deque([v for v in needed_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for c in children.get(u, []):
            if c in needed_set:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
                    
    # If topo_order is shorter than needed_vertices, there's a cycle or error.
    # Fallback to original list if that happens.
    if len(topo_order) != len(needed_vertices):
        topo_order = needed_vertices

    # 2. Precompute uses
    # uses[u] = list of indices in topo_order where u is a parent
    uses = {u: [] for u in needed_vertices}
    for i, v in enumerate(topo_order):
        for p in parent_dict.get(v, []):
            if p in needed_set:
                uses[p].append(i)
                
    # next_use_ptr[u] = index in uses[u] to the next use
    next_use_ptr = {u: 0 for u in needed_vertices}
    
    def get_next_use(u):
        ptr = next_use_ptr[u]
        if ptr < len(uses[u]):
            return uses[u][ptr]
        return float('inf')

    # State
    fast_set = set()
    slow_set = set(inputs)
    computed_set = set(inputs)
    operations = []
    
    def load(v):
        fast_set.add(v)
        slow_set.discard(v)
        operations.append(['load', v])
        
    def store(v):
        fast_set.discard(v)
        slow_set.add(v)
        operations.append(['store', v])
        
    def delete(v):
        fast_set.discard(v)
        operations.append(['delete', v])
        
    def compute(v):
        fast_set.add(v)
        computed_set.add(v)
        operations.append(['compute', v])
        
    def evict_one(protect_set):
        """
        Evict one vertex from fast memory.
        1. If any non-protected vertex has next_use == inf, delete it (cost 0).
        2. Else, pick the protected or non-protected vertex with max next_use.
           If it's an output, store it.
           If it's not an output, store it.
           (Actually, if next_use is finite, we store. If inf, we delete if not output, store if output).
        """
        if not fast_set:
            return None
            
        candidates = [u for u in fast_set if u not in protect_set]
        if not candidates:
            candidates = list(fast_set)
            
        # Find best candidate to evict
        best_candidate = None
        best_score = -1
        
        for u in candidates:
            nu = get_next_use(u)
            is_output = u in outputs
            
            # Score:
            # If nu is inf and not output: Score = (0, 0) -> Delete (Best)
            # If nu is inf and is output: Score = (1, inf) -> Store (Bad, but necessary if only outputs)
            # If nu is finite: Score = (1, nu) -> Store (Worse than delete, better than storing inf? No, storing inf is bad because we lose it? No, storing inf is bad because we might need it? No, if inf, we don't need it. But if it's an output, we DO need it at the end. So we must store it.)
            
            # Let's define priority:
            # 1. Delete non-output with inf next use. (Cost 0)
            # 2. Store non-output with finite next use. (Cost 1)
            # 3. Store output with inf next use. (Cost 1)
            
            if nu == float('inf'):
                if not is_output:
                    # Delete is best
                    if best_candidate is None or best_score[0] > 0:
                        best_candidate = u
                        best_score = (0, 0)
                else:
                    # Store output
                    if best_candidate is None or best_score[0] > 1 or (best_score[0] == 1 and best_score[1] < float('inf')):
                        best_candidate = u
                        best_score = (1, float('inf'))
            else:
                # Store non-output
                if best_candidate is None or best_score[0] > 1 or (best_score[0] == 1 and best_score[1] < nu):
                    best_candidate = u
                    best_score = (1, nu)
                    
        if best_candidate is None:
            return None
            
        nu = get_next_use(best_candidate)
        is_output = best_candidate in outputs
        
        if nu == float('inf') and not is_output:
            delete(best_candidate)
        else:
            store(best_candidate)
            
        return best_candidate

    # 3. Simulation
    for i, v in enumerate(topo_order):
        if time.time() > deadline:
            break
            
        # Proactive Delete: Delete any non-output with no more uses
        to_delete = []
        for u in list(fast_set):
            if u in outputs:
                continue
            if get_next_use(u) == float('inf'):
                to_delete.append(u)
        
        for u in to_delete:
            delete(u)
            
        # Ensure parents are in fast
        parents = [p for p in parent_dict.get(v, []) if p in needed_set]
        
        for p in parents:
            if p in fast_set:
                continue
                
            # Need to load p
            if len(fast_set) >= fast_memory_budget:
                # Evict
                # Protect current parent and current vertex?
                # We need p, so protect p? No, p is not in fast yet.
                # Protect other parents of v that are not yet loaded?
                remaining_parents = set(parents) - {p}
                evict_one(remaining_parents | {v})
                
            if p in slow_set:
                load(p)
            else:
                # Should not happen if logic is correct
                pass

        # Compute v
        if v in inputs:
            # Inputs are not computed, but might need to be in fast?
            # No, only if they are parents. If v is an input, it has no parents.
            # It might be a parent of future vertices.
            # We don't load it now. It will be loaded when a child needs it.
            pass
        else:
            if len(fast_set) >= fast_memory_budget:
                evict_one({v})
            compute(v)
            
        # Update next_use pointers for parents of v
        for p in parents:
            # The use at index i is now consumed
            # Advance pointer if it points to i
            if next_use_ptr[p] < len(uses[p]) and uses[p][next_use_ptr[p]] == i:
                next_use_ptr[p] += 1

    # 4. Final Pass: Ensure all outputs are in fast
    for v in outputs:
        if v in needed_set and v not in fast_set:
            if len(fast_set) >= fast_memory_budget:
                # Evict, protecting all outputs
                evict_one(outputs)
            if v in slow_set:
                load(v)
            else:
                # If not in slow, it was deleted. This is an error in logic.
                # But if it was deleted, it's gone. We can't recover.
                # However, with the "protect outputs" logic in evict_one,
                # we should have stored it, not deleted it.
                pass

    return operations
# EVOLVE-BLOCK-END
