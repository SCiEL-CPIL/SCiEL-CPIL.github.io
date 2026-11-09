"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill

    def time_left():
        return deadline - (time.time() - t0)

    # Try to import baseline first as a fallback
    baseline_ops = None
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        baseline_ops = pebble_baseline(dag, fast_memory_budget)
    except Exception:
        baseline_ops = None

    # Try to understand the DAG structure
    nodes = None
    parents = None
    outputs = None

    # Try to parse dag structure
    try:
        if isinstance(dag, dict):
            # Convention 1: dag = {'parents': [...], 'outputs': [...]}
            if 'parents' in dag:
                parents = dag['parents']
            if 'outputs' in dag:
                outputs = dag['outputs']
            # Get all nodes
            if isinstance(parents, list):
                nodes = list(range(len(parents)))
            elif isinstance(parents, dict):
                nodes = list(parents.keys())
        elif isinstance(dag, list):
            # Convention 2: dag is a list of node dicts or tuples
            nodes = list(range(len(dag)))
            parents = []
            for i, node in enumerate(dag):
                if isinstance(node, dict):
                    p = node.get('parents', [])
                    parents.append(p)
                elif isinstance(node, (list, tuple)):
                    if len(node) >= 2 and isinstance(node[1], (list, tuple)):
                        parents.append(list(node[1]))
                    else:
                        parents.append([])
                else:
                    parents.append([])
            # Try to find outputs
            outputs = None
            for i, node in enumerate(dag):
                if isinstance(node, dict):
                    if node.get('is_output') or node.get('output'):
                        if outputs is None:
                            outputs = []
                        outputs.append(i)
        else:
            pass
    except Exception:
        pass

    # If we couldn't parse, try baseline
    if baseline_ops is not None and (nodes is None or parents is None):
        return baseline_ops

    if nodes is None or parents is None:
        if baseline_ops is not None:
            return baseline_ops
        return []

    # Normalize parents to list of lists of ints
    if isinstance(parents, dict):
        parent_list = []
        for i in range(len(nodes)):
            if i in parents:
                parent_list.append(list(parents[i]))
            else:
                parent_list.append([])
        parents = parent_list

    # Ensure outputs is a list
    if outputs is None:
        all_parents = set()
        for p_list in parents:
            for p in p_list:
                all_parents.add(p)
        outputs = [i for i in range(len(nodes)) if i not in all_parents]
        if not outputs:
            outputs = [len(nodes) - 1]

    if not isinstance(outputs, list):
        outputs = [outputs]

    n = len(nodes)
    budget = fast_memory_budget

    # Build adjacency
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in parents[i]:
            children[p].append(i)

    input_nodes = [i for i in range(n) if len(parents[i]) == 0]
    output_set = set(outputs)

    # Compute depth and topological order
    depth = [-1] * n
    in_degree = [len(parents[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        if depth[u] == -1:
            depth[u] = 0
        for v in children[u]:
            in_degree[v] -= 1
            depth[v] = max(depth[v], depth[u] + 1)
            if in_degree[v] == 0:
                queue.append(v)

    max_depth = max(depth) if depth else 0
    layers = [[] for _ in range(max_depth + 1)]
    for v in range(n):
        if depth[v] >= 0:
            layers[depth[v]].append(v)

    # Compute max descendant depth
    max_desc_depth = [depth[i] for i in range(n)]
    for v in reversed(topo):
        for child in children[v]:
            max_desc_depth[v] = max(max_desc_depth[v], max_desc_depth[child])

    # For small DAGs, try exact Dijkstra
    if n <= 12 and time_left() > 10.0:
        start_state = frozenset()
        
        # State: frozenset of red vertices
        # Cost: number of transfers (load + store)
        # We want to minimize transfers, with secondary objective of minimizing ops
        
        dist = {start_state: (0, 0)}
        parent = {start_state: (None, None)}
        pq = [(0, 0, start_state)]
        
        max_states = 200000
        states_explored = 0
        
        while pq and states_explored < max_states and time_left() > 3.0:
            cost, num_ops, state = heapq.heappop(pq)
            
            if (cost, num_ops) > dist.get(state, (float('inf'), float('inf'))):
                continue
            
            states_explored += 1
            
            # Check if target: all outputs are blue (not in red set)
            if all(v not in state for v in outputs):
                # Reconstruct path
                path = []
                cur = state
                while parent[cur][0] is not None:
                    prev, op = parent[cur]
                    path.append(op)
                    cur = prev
                path.reverse()
                return path
            
            red_set = state
            red_count = len(red_set)
            
            for v in range(n):
                is_red = v in red_set
                is_input = len(parents[v]) == 0
                parents_all_red = all(p in red_set for p in parents[v])
                
                # load(v): v must be blue, v must be input
                if not is_red and is_input and red_count < budget:
                    new_state = red_set | frozenset([v])
                    new_cost = cost + 1
                    new_ops = num_ops + 1
                    if new_ops <= 20000:
                        if new_state not in dist or (new_cost, new_ops) < dist[new_state]:
                            dist[new_state] = (new_cost, new_ops)
                            parent[new_state] = (state, ['load', v])
                            heapq.heappush(pq, (new_cost, new_ops, new_state))
                
                # compute(v): all parents must be red, v must be blue
                if not is_red and parents_all_red and red_count < budget:
                    new_state = red_set | frozenset([v])
                    new_cost = cost
                    new_ops = num_ops + 1
                    if new_ops <= 20000:
                        if new_state not in dist or (new_cost, new_ops) < dist[new_state]:
                            dist[new_state] = (new_cost, new_ops)
                            parent[new_state] = (state, ['compute', v])
                            heapq.heappush(pq, (new_cost, new_ops, new_state))
                
                # store(v): v must be red, v must be an output
                if is_red and v in output_set:
                    new_state = red_set - frozenset([v])
                    new_cost = cost + 1
                    new_ops = num_ops + 1
                    if new_ops <= 20000:
                        if new_state not in dist or (new_cost, new_ops) < dist[new_state]:
                            dist[new_state] = (new_cost, new_ops)
                            parent[new_state] = (state, ['store', v])
                            heapq.heappush(pq, (new_cost, new_ops, new_state))
                
                # delete(v): v must be red, v must NOT be an output
                if is_red and v not in output_set:
                    new_state = red_set - frozenset([v])
                    new_cost = cost
                    new_ops = num_ops + 1
                    if new_ops <= 20000:
                        if new_state not in dist or (new_cost, new_ops) < dist[new_state]:
                            dist[new_state] = (new_cost, new_ops)
                            parent[new_state] = (state, ['delete', v])
                            heapq.heappush(pq, (new_cost, new_ops, new_state))
        
        # If exact search didn't finish, fall through to heuristic

    # Layered heuristic approach
    # Key insight: for layered DAGs, we can process layer by layer
    # and delete nodes as soon as they're no longer needed
    
    def validate_schedule(ops):
        """Validate that a schedule is correct and within memory budget."""
        if not ops:
            return False
        if len(ops) > 20000:
            return False
        
        red = set()
        max_mem = 0
        
        for op, v in ops:
            if v < 0 or v >= n:
                return False
            
            if op == 'load':
                if v in red:
                    return False
                if not is_input_node(v):
                    return False
                red.add(v)
            elif op == 'store':
                if v not in red:
                    return False
                red.remove(v)
            elif op == 'compute':
                if v in red:
                    return False
                if not all(p in red for p in parents[v]):
                    return False
                red.add(v)
            elif op == 'delete':
                if v not in red:
                    return False
                red.remove(v)
            else:
                return False
            
            max_mem = max(max_mem, len(red))
        
        if max_mem > budget:
            return False
        
        # All outputs must be blue (not in red)
        for v in outputs:
            if v in red:
                return False
        
        return True
    
    def is_input_node(v):
        return len(parents[v]) == 0
    
    # Strategy 1: Greedy layer-by-layer with aggressive deletion
    def greedy_layer_schedule():
        ops = []
        current_memory = set()
        
        # We need to track which nodes are "active" - still needed
        # A node is needed if it has descendants not yet computed
        
        # Process in topological order, but be smart about memory management
        computed = set()
        
        for v in topo:
            if v in current_memory:
                continue
            
            # Check if we can compute v
            if all(p in current_memory for p in parents[v]):
                # Compute v
                if len(current_memory) < budget:
                    ops.append(['compute', v])
                    current_memory.add(v)
                    computed.add(v)
                else:
                    # Need to free space - delete nodes that are no longer needed
                    # Find nodes that can be deleted
                    deletable = []
                    for u in current_memory:
                        if u in output_set:
                            continue
                        # Check if u is still needed
                        # u is needed if any child of u is not yet computed
                        # Actually, u is needed if any descendant is not yet computed
                        # Simplified: u is needed if max_desc_depth[u] > depth[v]
                        # Or more precisely, if any child of u is not in computed
                        still_needed = False
                        for child in children[u]:
                            if child not in computed:
                                still_needed = True
                                break
                        if not still_needed:
                            deletable.append(u)
                    
                    # Delete enough nodes to make room
                    for u in deletable:
                        if len(current_memory) >= budget:
                            ops.append(['delete', u])
                            current_memory.remove(u)
                        else:
                            break
                    
                    # If still can't fit, this is a problem
                    # For now, assume we can always make room in layered DAGs
                    if len(current_memory) < budget:
                        ops.append(['compute', v])
                        current_memory.add(v)
                        computed.add(v)
                    else:
                        # Fallback: can't compute, skip for now (shouldn't happen in valid DAG)
                        pass
            else:
                # Some parents not in memory - need to load them
                # This shouldn't happen in a proper layered DAG if we keep parents
                # But handle it gracefully
                for p in parents[v]:
                    if p not in current_memory:
                        # Need to load p
                        if len(current_memory) >= budget:
                            # Free space
                            deletable = []
                            for u in current_memory:
                                if u in output_set:
                                    continue
                                still_needed = False
                                for child in children[u]:
                                    if child not in computed:
                                        still_needed = True
                                        break
                                if not still_needed:
                                    deletable.append(u)
                            
                            for u in deletable:
                                if len(current_memory) >= budget:
                                    ops.append(['delete', u])
                                    current_memory.remove(u)
                                else:
                                    break
                        
                        if len(current_memory) < budget:
                            ops.append(['load', p])
                            current_memory.add(p)
                        else:
                            # Can't load, this is a failure case
                            return None
                
                # Now try to compute v
                if all(p in current_memory for p in parents[v]):
                    if len(current_memory) < budget:
                        ops.append(['compute', v])
                        current_memory.add(v)
                        computed.add(v)
        
        # Store all outputs
        for v in outputs:
            if v in current_memory:
                ops.append(['store', v])
                current_memory.remove(v)
            else:
                # Output not in memory - problem
                return None
        
        # Delete remaining non-output nodes
        for v in list(current_memory):
            if v not in output_set:
                ops.append(['delete', v])
                current_memory.remove(v)
        
        return ops
    
    # Strategy 2: More careful layer-by-layer with explicit layer management
    def layer_schedule():
        ops = []
        current_memory = set()
        
        # Load all inputs first
        for v in input_nodes:
            ops.append(['load', v])
            current_memory.add(v)
        
        # Process layer by layer
        for layer_idx in range(1, max_depth + 1):
            layer_nodes = layers[layer_idx]
            
            # Compute all nodes in this layer
            for v in layer_nodes:
                # All parents should be in memory (from previous layer)
                if all(p in current_memory for p in parents[v]):
                    if len(current_memory) < budget:
                        ops.append(['compute', v])
                        current_memory.add(v)
                    else:
                        # Need to free space - delete from previous layers
                        # Nodes from layer < layer_idx that are no longer needed
                        for u in list(current_memory):
                            if depth[u] < layer_idx and u not in output_set:
                                # Check if u is still needed
                                # u is needed if max_desc_depth[u] > layer_idx
                                if max_desc_depth[u] <= layer_idx:
                                    ops.append(['delete', u])
                                    current_memory.remove(u)
                                    if len(current_memory) < budget:
                                        break
                        
                        if len(current_memory) < budget:
                            ops.append(['compute', v])
                            current_memory.add(v)
                        else:
                            # Can't fit - this is a problem
                            # For now, just try to compute anyway (will fail validation)
                            ops.append(['compute', v])
                            current_memory.add(v)
                
                # After computing v, check if we can delete parents that are no longer needed
                for p in parents[v]:
                    if p in current_memory and p not in output_set:
                        # Check if p is still needed by other nodes in this or later layers
                        still_needed = False
                        for child in children[p]:
                            if child not in current_memory and depth[child] >= layer_idx:
                                # Child not yet computed or in memory
                                # Actually, we should check if child is in computed set
                                # Simplified: if child's depth >= layer_idx and child != v, p might still be needed
                                if child != v:
                                    still_needed = True
                                    break
                        if not still_needed:
                            ops.append(['delete', p])
                            current_memory.remove(p)
            
            # After processing the layer, clean up
            for u in list(current_memory):
                if depth[u] < layer_idx and u not in output_set:
                    if max_desc_depth[u] <= layer_idx:
                        ops.append(['delete', u])
                        current_memory.remove(u)
        
        # Store all outputs
        for v in outputs:
            if v in current_memory:
                ops.append(['store', v])
                current_memory.remove(v)
        
        # Delete remaining
        for v in list(current_memory):
            if v not in output_set:
                ops.append(['delete', v])
                current_memory.remove(v)
        
        return ops
    
    # Strategy 3: Naive topological order (baseline-like)
    def naive_schedule():
        ops = []
        for v in topo:
            if len(parents[v]) == 0:
                ops.append(['load', v])
            else:
                ops.append(['compute', v])
        
        for v in outputs:
            if v in topo:
                ops.append(['store', v])
        
        for v in topo:
            if v not in output_set:
                ops.append(['delete', v])
        
        return ops
    
    # Try strategies in order of expected quality
    best_ops = None
    
    # Try layer schedule first (should work well for layered DAGs)
    if time_left() > 5.0:
        ops = layer_schedule()
        if ops and validate_schedule(ops):
            best_ops = ops
    
    # Try greedy layer schedule
    if best_ops is None and time_left() > 5.0:
        ops = greedy_layer_schedule()
        if ops and validate_schedule(ops):
            best_ops = ops
    
    # Try naive schedule
    if best_ops is None and time_left() > 2.0:
        ops = naive_schedule()
        if ops and validate_schedule(ops):
            best_ops = ops
    
    # If we have a valid schedule, use it
    if best_ops is not None:
        return best_ops
    
    # Fallback to baseline
    if baseline_ops is not None:
        return baseline_ops
    
    return []

# EVOLVE-BLOCK-END
