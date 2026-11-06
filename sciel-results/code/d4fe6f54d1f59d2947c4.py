"""Optimized DAG pebbling solution."""
import time
import heapq
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG to minimize data transfers.
    
    Args:
        dag: Dictionary with 'parents' (dict vertex -> list of parent vertices),
             'inputs' (list of input vertices), 'outputs' (list of output vertices)
        fast_memory_budget: Maximum number of values in fast memory
    
    Returns:
        List of [operation, vertex] operations
    """
    start_time = time.time()
    DEADLINE = 85.0
    
    # Parse DAG structure
    parents_dict = dag.get('parents', {})
    inputs = list(dag.get('inputs', []))
    outputs = list(dag.get('outputs', []))
    outputs_set = set(outputs)
    
    # Build children map and collect all vertices
    children = defaultdict(list)
    all_vertices = set()
    for v, ps in parents_dict.items():
        all_vertices.add(v)
        for p in ps:
            all_vertices.add(p)
            children[p].append(v)
    for v in inputs:
        all_vertices.add(v)
    for v in outputs:
        all_vertices.add(v)
    
    # Input vertices: those with no parents
    input_set = set(v for v in all_vertices if len(parents_dict.get(v, [])) == 0)
    
    # Precompute parent frozensets
    parent_sets = {v: frozenset(parents_dict.get(v, [])) for v in all_vertices}
    
    # Initial state: inputs in fast memory
    initial_state = frozenset(inputs)
    
    # If budget is 0 or negative, we can't hold anything
    if fast_memory_budget <= 0:
        # Can't hold any values, but outputs must be blue...
        # This might be infeasible, return empty or minimal
        return []
    
    # Check if already done
    if initial_state >= outputs_set:
        return []
    
    # Dijkstra on state space
    # State: frozenset of vertices in fast memory
    # Cost: number of store operations (transfers)
    
    dist = {initial_state: 0}
    parent_map = {}  # state -> (prev_state, op, vertex)
    
    pq = [(0, initial_state)]
    
    max_states = 30000
    found_goal = None
    
    while pq and len(dist) < max_states:
        if time.time() - start_time > DEADLINE:
            break
        
        cost, state = heapq.heappop(pq)
        
        if cost > dist.get(state, float('inf')):
            continue
        
        # Goal check
        if state >= outputs_set:
            found_goal = state
            break
        
        # Generate transitions
        # 1. Compute: add vertex v if all parents in state and v not in state
        if len(state) < fast_memory_budget:
            for v in all_vertices:
                if v in state:
                    continue
                if parent_sets[v] <= state:
                    new_state = state | {v}
                    new_cost = cost  # compute is free
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        parent_map[new_state] = (state, 'compute', v)
                        heapq.heappush(pq, (new_cost, new_state))
        
        # 2. Store: remove vertex v from state, cost +1
        for v in state:
            # Don't store inputs (can't recompute)
            if v in input_set and v not in outputs_set:
                continue
            new_state = state - {v}
            new_cost = cost + 1
            if new_cost < dist.get(new_state, float('inf')):
                dist[new_state] = new_cost
                parent_map[new_state] = (state, 'store', v)
                heapq.heappush(pq, (new_cost, new_state))
    
    # Reconstruct
    if found_goal is not None:
        ops = []
        state = found_goal
        while state != initial_state:
            prev_state, op, v = parent_map[state]
            ops.append([op, v])
            state = prev_state
        ops.reverse()
        
        # Validate
        mem = set(inputs)
        valid = True
        for op, v in ops:
            if op == 'compute':
                if v in mem:
                    valid = False
                    break
                ps = parent_sets.get(v, frozenset())
                if not ps <= mem:
                    valid = False
                    break
                if len(mem) + 1 > fast_memory_budget:
                    valid = False
                    break
                mem.add(v)
            elif op == 'store':
                if v not in mem:
                    valid = False
                    break
                mem.discard(v)
        
        if valid and mem >= outputs_set:
            return ops
    
    # Fallback: greedy
    return _greedy_fallback(parents_dict, inputs, outputs_set, fast_memory_budget, all_vertices, children, input_set, parent_sets)


def _greedy_fallback(parents_dict, inputs, outputs_set, fast_memory_budget, all_vertices, children, input_set, parent_sets):
    """Greedy pebbling fallback."""
    ops = []
    in_fast = set(inputs)
    computed = set(inputs)  # Vertices whose values are available (in fast or computed)
    
    # We need to compute all ancestors of outputs
    # Process in topological order
    
    from collections import deque
    in_degree = {v: len(parents_dict.get(v, [])) for v in all_vertices}
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    topo_order = []
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # Process in topological order
    for v in topo_order:
        if v in in_fast:
            continue
        ps = parent_sets[v]
        # Check if all parents are in fast memory
        if ps <= in_fast:
            if len(in_fast) < fast_memory_budget:
                ops.append(['compute', v])
                in_fast.add(v)
            else:
                # Need to free space
                # Store a vertex that's not an output and not an input
                for u in list(in_fast):
                    if u in outputs_set or u in input_set:
                        continue
                    # Check if u is needed: if any child of u is not yet computed
                    # Simple heuristic: store u if all its children are in in_fast or not needed
                    ops.append(['store', u])
                    in_fast.discard(u)
                    break
                if len(in_fast) < fast_memory_budget:
                    ops.append(['compute', v])
                    in_fast.add(v)
    
    # Ensure outputs are in fast memory
    for v in outputs_set:
        if v not in in_fast:
            # Try to compute it
            ps = parent_sets[v]
            if ps <= in_fast:
                if len(in_fast) < fast_memory_budget:
                    ops.append(['compute', v])
                    in_fast.add(v)
                else:
                    # Free space
                    for u in list(in_fast):
                        if u in outputs_set or u in input_set:
                            continue
                        ops.append(['store', u])
                        in_fast.discard(u)
                        break
                    if len(in_fast) < fast_memory_budget:
                        ops.append(['compute', v])
                        in_fast.add(v)
    
    return ops
# EVOLVE-BLOCK-END
