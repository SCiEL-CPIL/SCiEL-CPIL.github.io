"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import heapq
from collections import defaultdict

def _parse_dag(dag):
    """Parse various possible DAG formats into a unified structure."""
    # Try to extract: vertices, parents dict, outputs, inputs
    if isinstance(dag, dict):
        # Common keys: 'parents', 'outputs', 'inputs', 'vertices', 'n'
        parents = dag.get('parents', None)
        outputs = dag.get('outputs', None)
        inputs = dag.get('inputs', None)
        vertices = dag.get('vertices', None)
        n = dag.get('n', None)
        
        if parents is None:
            # Maybe it's a list of parent lists
            if isinstance(dag, list):
                parents = [list(p) for p in dag]
                n = len(dag)
                if outputs is None:
                    outputs = [n - 1]  # last vertex
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
                # Outputs are vertices with no children
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


def _naive_schedule(parents, outputs, fast_memory_budget):
    """Generate a naive topological schedule for baseline comparison."""
    n = len(parents)
    ops = []
    # Simple: load all inputs, compute in topo order, store outputs
    # Process in topological order (vertices are already in topo order)
    in_fast = set()
    
    for v in range(n):
        # Load inputs if not already in fast memory
        for p in parents[v]:
            if p not in in_fast:
                ops.append(['load', p])
                in_fast.add(p)
        
        if v in outputs:
            # Compute and store
            if not parents[v]:
                # Input that's also output
                if v not in in_fast:
                    ops.append(['load', v])
                    in_fast.add(v)
                ops.append(['store', v])
                in_fast.discard(v)
            else:
                # Compute
                ops.append(['compute', v])
                in_fast.add(v)
                # Store output
                ops.append(['store', v])
                in_fast.discard(v)
        else:
            if parents[v]:
                # Compute intermediate
                ops.append(['compute', v])
                in_fast.add(v)
                # If budget exceeded, delete some
                while len(in_fast) > fast_memory_budget:
                    # Delete the one that won't be needed soonest
                    # Simple: delete first one added that's not an output
                    for p in list(in_fast):
                        if p not in outputs:
                            ops.append(['delete', p])
                            in_fast.discard(p)
                            break
                    else:
                        break
    
    return ops


def _smart_pebbling(parents, outputs, fast_memory_budget):
    """
    Smart pebbling strategy for layered DAGs.
    Process layer by layer, keeping values in fast memory when budget allows.
    """
    n = len(parents)
    if n == 0:
        return []
    
    # Determine layers (topological depth)
    depth = [0] * n
    for v in range(n):
        if parents[v]:
            depth[v] = max(depth[p] for p in parents[v]) + 1
    
    max_depth = max(depth) if depth else 0
    
    # Group vertices by depth
    layers = [[] for _ in range(max_depth + 1)]
    for v in range(n):
        layers[depth[v]].append(v)
    
    ops = []
    in_fast = set()
    
    # Track which vertices are outputs
    is_output = set(outputs)
    
    # For each layer, decide what to keep in fast memory
    # Strategy: keep vertices that will be needed in future layers
    # and fit in the budget
    
    for d in range(max_depth + 1):
        layer_vertices = layers[d]
        
        for v in layer_vertices:
            if not parents[v]:
                # Input vertex
                if v in is_output:
                    # Input that's also output: load and store
                    if v not in in_fast:
                        ops.append(['load', v])
                        in_fast.add(v)
                    ops.append(['store', v])
                    in_fast.discard(v)
                else:
                    # Regular input: load if needed in future
                    # Check if any child in future layers needs it
                    needs_in_future = False
                    for future_d in range(d + 1, max_depth + 1):
                        for fv in layers[future_d]:
                            if v in parents[fv]:
                                needs_in_future = True
                                break
                        if needs_in_future:
                            break
                    
                    if needs_in_future:
                        # Load and keep if budget allows
                        if v not in in_fast:
                            # Make room if needed
                            while len(in_fast) >= fast_memory_budget:
                                # Delete a vertex that's no longer needed
                                deleted = False
                                for u in list(in_fast):
                                    # Check if u is still needed in future
                                    still_needed = False
                                    for future_d2 in range(d + 1, max_depth + 1):
                                        for fv in layers[future_d2]:
                                            if u in parents[fv]:
                                                still_needed = True
                                                break
                                        if still_needed:
                                            break
                                    if not still_needed:
                                        ops.append(['delete', u])
                                        in_fast.discard(u)
                                        deleted = True
                                        break
                                if not deleted:
                                    break
                                ops.append(['load', v])
                                in_fast.add(v)
                    # If not needed in future, don't load
            else:
                # Computed vertex
                # Ensure parents are in fast memory
                for p in parents[v]:
                    if p not in in_fast:
                        ops.append(['load', p])
                        in_fast.add(p)
                
                # Compute v
                ops.append(['compute', v])
                in_fast.add(v)
                
                # Free parents that are no longer needed
                for p in parents[v]:
                    still_needed = False
                    for future_d in range(d + 1, max_depth + 1):
                        for fv in layers[future_d]:
                            if p in parents[fv]:
                                still_needed = True
                                break
                        if still_needed:
                            break
                    if not still_needed and p not in is_output:
                        ops.append(['delete', p])
                        in_fast.discard(p)
                
                if v in is_output:
                    # Store output
                    ops.append(['store', v])
                    in_fast.discard(v)
                else:
                    # Keep in fast memory if budget allows, else delete
                    if len(in_fast) > fast_memory_budget:
                        # Need to delete something
                        # Delete the one least likely to be needed
                        for u in list(in_fast):
                            if u not in is_output:
                                still_needed = False
                                for future_d in range(d + 1, max_depth + 1):
                                    for fv in layers[future_d]:
                                        if u in parents[fv]:
                                            still_needed = True
                                            break
                                    if still_needed:
                                        break
                                if not still_needed:
                                    ops.append(['delete', u])
                                    in_fast.discard(u)
                                    break
    
    return ops


def _dijkstra_pebble(parents, outputs, fast_memory_budget, time_limit=80.0):
    """
    Exact state-space search using Dijkstra's algorithm.
    State: (frozenset of vertices in fast memory, frozenset of computed vertices)
    Goal: all outputs stored (i.e., all outputs have been computed and their values persisted).
    Cost: number of transfers (load + store operations).
    """
    n = len(parents)
    t0 = time.time()
    
    # Precompute children for each vertex
    children = [[] for _ in range(n)]
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Precompute: for each vertex, the set of descendants (vertices that transitively depend on it)
    # This helps determine if a value is still needed
    # But for exact search, we track computed set explicitly
    
    is_output = set(outputs)
    
    # Initial state: no vertices in fast memory, no vertices computed
    # But wait - inputs start blue, meaning they are available in slow memory
    # We need to load them to compute
    
    # State: (frozenset(fast_memory), frozenset(computed))
    # A vertex is "computed" if we've done a compute op on it and it's in fast memory
    # or if we've stored it (but storing removes from fast memory)
    
    # Actually, let's think more carefully:
    # - "computed" means the value has been derived (via compute or it's an input)
    # - "in_fast" means it's currently in fast memory
    # - An input vertex is "computed" from the start (it's blue/available)
    # - A non-input vertex is "computed" only after a compute operation
    
    # State: (frozenset(in_fast), frozenset(computed))
    # where computed includes all inputs (they're always available)
    
    initial_computed = frozenset(v for v in range(n) if not parents[v])
    initial_state = (frozenset(), initial_computed)
    
    # Goal: all outputs must be "stored" - meaning at some point they were in fast memory
    # and we performed a store. After storing, they're no longer in fast memory.
    # We need to track which outputs have been stored.
    
    # Revised state: (frozenset(in_fast), frozenset(computed), frozenset(stored))
    # But stored is a subset of computed, and once stored, the value is persisted.
    
    # Actually, for the goal condition: all outputs must have been stored at least once.
    # Let's track stored_outputs separately.
    
    # State: (frozenset(in_fast), frozenset(computed), frozenset(stored_outputs))
    # stored_outputs only tracks which output vertices have been stored
    
    initial_state = (frozenset(), initial_computed, frozenset())
    goal_stored = frozenset(outputs)
    
    # Dijkstra
    # dist[state] = min cost to reach state
    # Use a heap: (cost, state)
    
    dist = {initial_state: 0}
    heap = [(0, initial_state)]
    parent = {}  # For path reconstruction: state -> (prev_state, op, vertex)
    
    best_solution = None
    best_cost = float('inf')
    
    while heap:
        if time.time() - t0 > time_limit:
            break
        
        cost, state = heapq.heappop(heap)
        
        if cost > dist.get(state, float('inf')):
            continue
        
        in_fast, computed, stored = state
        
        # Check if we've reached goal
        if stored == goal_stored:
            best_cost = cost
            # Reconstruct path
            path = []
            curr = state
            while curr in parent:
                prev, op, v = parent[curr]
                path.append([op, v])
                curr = prev
            path.reverse()
            best_solution = path
            break
        
        # Generate successor states
        # 1. Load a vertex: vertex must be computed (available) and not in fast, and fast not full
        for v in range(n):
            if v not in in_fast and v in computed and len(in_fast) < fast_memory_budget:
                new_in_fast = in_fast | {v}
                new_state = (new_in_fast, computed, stored)
                new_cost = cost + 1  # load is a transfer
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'load', v)
                    heapq.heappush(heap, (new_cost, new_state))
        
        # 2. Compute a vertex: all parents must be in fast, vertex not already computed
        for v in range(n):
            if v not in computed and v not in in_fast:
                # Check all parents are in fast
                all_parents_in_fast = all(p in in_fast for p in parents[v])
                if all_parents_in_fast and len(in_fast) < fast_memory_budget:
                    new_in_fast = in_fast | {v}
                    new_computed = computed | {v}
                    new_state = (new_in_fast, new_computed, stored)
                    new_cost = cost  # compute is NOT a transfer
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        parent[new_state] = (state, 'compute', v)
                        heapq.heappush(heap, (new_cost, new_state))
        
        # 3. Store a vertex: must be in fast and be an output
        for v in in_fast:
            if v in is_output and v not in stored:
                new_in_fast = in_fast - {v}
                new_stored = stored | {v}
                new_state = (new_in_fast, computed, new_stored)
                new_cost = cost + 1  # store is a transfer
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'store', v)
                    heapq.heappush(heap, (new_cost, new_state))
        
        # 4. Delete a vertex: must be in fast, and not an output that hasn't been stored yet
        for v in in_fast:
            if v not in is_output or v in stored:
                new_in_fast = in_fast - {v}
                new_state = (new_in_fast, computed, stored)
                new_cost = cost  # delete is NOT a transfer
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent[new_state] = (state, 'delete', v)
                    heapq.heappush(heap, (new_cost, new_state))
    
    if best_solution is not None:
        return best_solution
    
    # If we didn't find a solution within time limit, return None
    return None


def _validate_ops(ops, parents, outputs, fast_memory_budget):
    """Validate that operations are legal and complete."""
    n = len(parents)
    in_fast = set()
    computed = set()
    stored = set()
    
    for op, v in ops:
        if op == 'load':
            if v in in_fast:
                continue  # Already loaded
            if len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.discard(v)
            stored.add(v)
        elif op == 'compute':
            # All parents must be in fast memory
            for p in parents[v]:
                if p not in in_fast:
                    return False
            if v in in_fast:
                return False  # Already computed
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
    
    # All outputs must be stored
    for o in outputs:
        if o not in stored:
            return False
    
    return True


def _count_transfers(ops):
    """Count the number of transfers (load + store operations)."""
    return sum(1 for op, v in ops if op in ('load', 'store'))


def pebble(dag, fast_memory_budget):
    import time
    t0 = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    try:
        # Try to use the baseline if available
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        result = pebble_baseline(dag, fast_memory_budget)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Parse the DAG
    parents, outputs, inputs, vertices, n = _parse_dag(dag)
    
    if parents is None or n is None:
        return []
    
    if outputs is None:
        outputs = [n - 1]
    
    if fast_memory_budget is None:
        fast_memory_budget = n  # Default to no budget constraint
    
    if fast_memory_budget <= 0:
        fast_memory_budget = 1
    
    # Try Dijkstra exact search first (best for small/medium DAGs)
    # Only attempt if the state space is likely manageable
    # Heuristic: if n * fast_memory_budget is small enough
    elapsed = time.time() - t0
    if n <= 15 and fast_memory_budget <= 8:
        try:
            dijkstra_time_limit = max(1.0, deadline - elapsed - 5.0)
            ops = _dijkstra_pebble(parents, outputs, fast_memory_budget, time_limit=dijkstra_time_limit)
            if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
                return ops
        except Exception:
            pass
    
    # Try smart pebbling
    try:
        ops = _smart_pebbling(parents, outputs, fast_memory_budget)
        if _validate_ops(ops, parents, outputs, fast_memory_budget):
            return ops
    except Exception:
        pass
    
    # Fallback to naive
    try:
        ops = _naive_schedule(parents, outputs, fast_memory_budget)
        if _validate_ops(ops, parents, outputs, fast_memory_budget):
            return ops
    except Exception:
        pass
    
    # Last resort: simple topological with aggressive loading
    try:
        ops = []
        in_fast = set()
        for v in range(n):
            for p in parents[v]:
                if p not in in_fast:
                    if len(in_fast) >= fast_memory_budget:
                        # Delete something
                        for u in list(in_fast):
                            if u not in outputs:
                                ops.append(['delete', u])
                                in_fast.discard(u)
                                break
                    ops.append(['load', p])
                    in_fast.add(p)
            
            if parents[v]:
                ops.append(['compute', v])
                in_fast.add(v)
            elif v in outputs:
                if v not in in_fast:
                    ops.append(['load', v])
                    in_fast.add(v)
            
            if v in outputs:
                if v in in_fast:
                    ops.append(['store', v])
                    in_fast.discard(v)
                # Clean up parents
                for p in parents[v]:
                    if p in in_fast and p not in outputs:
                        # Check if still needed
                        still_needed = False
                        for fv in range(v + 1, n):
                            if p in parents[fv]:
                                still_needed = True
                                break
                        if not still_needed:
                            ops.append(['delete', p])
                            in_fast.discard(p)
        
        if _validate_ops(ops, parents, outputs, fast_memory_budget):
            return ops
    except Exception:
        pass
    
    # If still no valid solution, try Dijkstra with more time for medium-sized DAGs
    elapsed = time.time() - t0
    remaining = deadline - elapsed - 3.0
    if remaining > 2.0 and n <= 20:
        try:
            ops = _dijkstra_pebble(parents, outputs, fast_memory_budget, time_limit=remaining)
            if ops is not None and _validate_ops(ops, parents, outputs, fast_memory_budget):
                return ops
        except Exception:
            pass
    
    return []

# EVOLVE-BLOCK-END
