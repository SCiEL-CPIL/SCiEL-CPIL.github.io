"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def _extract_dag_info(dag):
    """
    Extract DAG structure. Handles common formats.
    """
    if isinstance(dag, dict):
        nodes = dag.get('nodes', [])
        if not nodes and 'vertices' in dag:
            nodes = dag['vertices']
        edges = dag.get('edges', [])
        inputs = dag.get('inputs', [])
        outputs = dag.get('outputs', [])
        parents = dag.get('parents', {})
        
        if not parents:
            parents = {n: [] for n in nodes}
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    parents[e[1]].append(e[0])
        
        if not inputs:
            inputs = [n for n in nodes if len(parents.get(n, [])) == 0]
        if not outputs:
            children = {n: set() for n in nodes}
            for n in nodes:
                for p in parents.get(n, []):
                    children[p].add(n)
            outputs = [n for n in nodes if len(children[n]) == 0]
        
        return nodes, parents, inputs, outputs
    
    elif isinstance(dag, (list, tuple)):
        nodes = []
        parents = {}
        inputs = []
        outputs = []
        
        for item in dag:
            if isinstance(item, dict):
                node = item.get('node', item.get('id', item.get('name')))
                par = item.get('parents', item.get('inputs', []))
                is_input = item.get('is_input', item.get('input', False))
                is_output = item.get('is_output', item.get('output', False))
                nodes.append(node)
                parents[node] = list(par)
                if is_input:
                    inputs.append(node)
                if is_output:
                    outputs.append(node)
            elif isinstance(item, (list, tuple)):
                node = item[0]
                par = list(item[1]) if len(item) > 1 else []
                nodes.append(node)
                parents[node] = par
        
        if not inputs:
            inputs = [n for n in nodes if len(parents.get(n, [])) == 0]
        if not outputs:
            children = {n: set() for n in nodes}
            for n in nodes:
                for p in parents.get(n, []):
                    children[p].add(n)
            outputs = [n for n in nodes if len(children[n]) == 0]
        
        return nodes, parents, inputs, outputs
    
    else:
        raise ValueError(f"Unsupported dag format: {type(dag)}")

def pebble(dag, fast_memory_budget):
    """
    Hybrid pebbling: 
    1. Attempt Dijkstra's algorithm over the state space of red-pebble configurations for small/medium DAGs.
    2. Fall back to a greedy heuristic with lookahead eviction for larger DAGs or if Dijkstra times out.
    """
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    
    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    input_set = set(inputs)
    output_set = set(outputs)
    all_nodes = set(nodes)
    
    # Topological sort
    in_degree = {n: len(parents.get(n, [])) for n in nodes}
    children = {n: [] for n in nodes}
    for n in nodes:
        for p in parents.get(n, []):
            children[p].append(n)
            
    queue = deque([n for n in nodes if in_degree[n] == 0])
    topo_order = []
    while queue:
        n = queue.popleft()
        topo_order.append(n)
        for c in children[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != len(nodes):
        # Cycle or error, fallback
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

    # --- Strategy 1: Dijkstra for small state spaces ---
    # State: (frozenset(red_nodes), frozenset(completed_outputs))
    # We only use Dijkstra if the number of nodes is small enough that the state space might be tractable.
    # Heuristic: if N <= 15 and budget >= max_in_degree, try Dijkstra.
    
    max_in_deg = max(len(parents.get(n, [])) for n in nodes) if nodes else 0
    n_nodes = len(nodes)
    
    use_dijkstra = (n_nodes <= 15) and (fast_memory_budget >= max_in_deg)
    
    if use_dijkstra:
        try:
            result = _dijkstra_pebble(nodes, parents, inputs, outputs, fast_memory_budget, deadline)
            if result is not None:
                return result
        except Exception:
            pass # Fall through to greedy

    # --- Strategy 2: Greedy with Lookahead ---
    result = _greedy_pebble(nodes, parents, inputs, outputs, fast_memory_budget, topo_order, children, deadline)
    if result is not None:
        return result
    
    # Final fallback
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    return pebble_baseline(dag, fast_memory_budget)

def _dijkstra_pebble(nodes, parents, inputs, outputs, budget, deadline):
    """
    Dijkstra's algorithm over (red_set, completed_set) states.
    Returns list of operations or None if timeout/failure.
    """
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Initial state: no red, no completed
    init_red = frozenset()
    init_comp = frozenset()
    init_state = (init_red, init_comp)
    
    dist = {init_state: 0}
    parent_map = {init_state: None} # (prev_state, op, vertex)
    pq = [(0, init_state)]
    
    while pq:
        if time.time() > deadline:
            return None
            
        d, state = heapq.heappop(pq)
        if d > dist.get(state, float('inf')):
            continue
            
        red, completed = state
        
        # Check goal
        if completed == output_set:
            # Reconstruct path
            schedule = []
            curr = state
            while parent_map[curr] is not None:
                prev, op, v = parent_map[curr]
                schedule.append([op, v])
                curr = prev
            schedule.reverse()
            return schedule
        
        # Generate neighbors
        # 1. Load inputs
        for inp in inputs:
            if inp not in red and inp not in completed:
                if len(red) < budget:
                    new_red = frozenset(red | {inp})
                    new_state = (new_red, completed)
                    new_d = d + 1
                    if new_d < dist.get(new_state, float('inf')):
                        dist[new_state] = new_d
                        parent_map[new_state] = (state, 'load', inp)
                        heapq.heappush(pq, (new_d, new_state))
                        
        # 2. Compute nodes
        for node in nodes:
            if node in completed or node in red:
                continue
            # Check if all parents are red
            pars = parents.get(node, [])
            can_compute = all(p in red for p in pars)
            if can_compute:
                new_red = frozenset(red | {node})
                new_state = (new_red, completed)
                new_d = d # compute costs 0 transfers
                if new_d < dist.get(new_state, float('inf')):
                    dist[new_state] = new_d
                    parent_map[new_state] = (state, 'compute', node)
                    heapq.heappush(pq, (new_d, new_state))
                    
        # 3. Store outputs
        for out in outputs:
            if out in red and out not in completed:
                new_red = frozenset(red - {out})
                new_comp = frozenset(completed | {out})
                new_state = (new_red, new_comp)
                new_d = d + 1
                if new_d < dist.get(new_state, float('inf')):
                    dist[new_state] = new_d
                    parent_map[new_state] = (state, 'store', out)
                    heapq.heappush(pq, (new_d, new_state))
                    
        # 4. Delete non-essential red nodes (optional, but helps free space)
        # Only delete if we are at budget limit and need to make space? 
        # Dijkstra explores all states, so we should allow deletion of any red node 
        # that is not an output waiting to be stored? 
        # Actually, deleting a node that is a parent of an uncomputed node is bad unless we recompute.
        # However, in this state space, "recompute" isn't an explicit state transition, 
        # it's implied by loading parents again. 
        # So we should allow deleting any red node.
        for r in list(red):
            # Don't delete if it's an output we intend to store? No, we can store it first.
            # If we delete it, we must recompute it later if it's an output.
            new_red = frozenset(red - {r})
            new_state = (new_red, completed)
            new_d = d # delete costs 0
            if new_d < dist.get(new_state, float('inf')):
                dist[new_state] = new_d
                parent_map[new_state] = (state, 'delete', r)
                heapq.heappush(pq, (new_d, new_state))
                
    return None

def _greedy_pebble(nodes, parents, inputs, outputs, budget, topo_order, children, deadline):
    """
    Greedy pebbling with lookahead eviction.
    """
    input_set = set(inputs)
    output_set = set(outputs)
    
    red = set()
    completed = set()
    schedule = []
    transfers = 0
    
    # Precompute "last use" or "future fan-out" metrics?
    # For each node, we can precompute the number of children.
    # But "future" children depends on current position in topo_order.
    
    # To make eviction smarter, we can look ahead a few steps.
    # However, simple heuristic: evict the node with the fewest remaining children.
    # If tie, evict the one used least recently (LRU).
    
    # LRU tracking
    timestamp = {}
    current_time = 0
    
    # Map node to index in topo_order for quick access
    topo_index = {n: i for i, n in enumerate(topo_order)}
    
    for node in topo_order:
        if time.time() > deadline:
            return None
            
        # 1. Ensure parents are in red
        for p in parents.get(node, []):
            if p not in red:
                # Load parent
                if len(red) >= budget:
                    # Evict
                    victim = _select_victim(red, completed, children, timestamp)
                    if victim:
                        red.remove(victim)
                        schedule.append(['delete', victim])
                        # timestamp.pop(victim, None)
                    
                red.add(p)
                timestamp[p] = current_time
                current_time += 1
                schedule.append(['load', p])
                transfers += 1
        
        # 2. Compute node
        if node not in red:
            if len(red) >= budget:
                # Evict
                victim = _select_victim(red, completed, children, timestamp)
                if victim:
                    red.remove(victim)
                    schedule.append(['delete', victim])
            
            red.add(node)
            timestamp[node] = current_time
            current_time += 1
            schedule.append(['compute', node])
        
        # 3. If node is output, store it
        if node in output_set:
            # Store node
            red.remove(node)
            completed.add(node)
            schedule.append(['store', node])
            transfers += 1
            # timestamp.pop(node, None)
            
            # Optional: Delete other red nodes that are no longer needed?
            # Greedy approach: keep them until they are evicted.
            # But we can be smarter: if a node in red has no children that are not completed 
            # and not in the current "window", we can delete it to free space.
            # This is a "proactive deletion" strategy.
            
            # Let's try to delete red nodes that have no future children.
            # Future children: children[c] where c is not in completed and topo_index[c] > topo_index[node]
            # Actually, since we process in topo order, any child of a red node that is not completed 
            # and has a higher topo index is "future".
            # If a red node has NO such children, it's safe to delete.
            
            for r in list(red):
                if r in completed:
                    continue
                has_future_child = False
                for c in children[r]:
                    if c not in completed:
                        # Check if c is in the future? 
                        # Since we are processing in topo order, if c is not completed, 
                        # it must be either:
                        # 1. Already processed but not stored (impossible for non-outputs? No, intermediate nodes are not stored, they stay red or deleted)
                        # Wait, intermediate nodes are NOT stored. Only outputs are stored.
                        # So "completed" only contains outputs.
                        # An intermediate node c is "future" if it hasn't been computed yet.
                        # How do we know if c has been computed? 
                        # We don't explicitly track "computed" for intermediates, only "red" and "completed".
                        # If c is not in red and not in completed, it might be blue (deleted) or not yet computed.
                        # If c is blue (deleted), we might need to recompute it.
                        # If c is not yet computed, we will compute it later.
                        # In both cases, if c is a child of r, and c is not red, 
                        # then r is only useful if we need to recompute c.
                        # But if c is not red, and we are about to process c (if it's in the future), 
                        # we will need r to be red then.
                        # So if c is in the future (topo_index[c] > topo_index[node]), 
                        # we should keep r.
                        
                        if topo_index.get(c, 0) > topo_index.get(node, 0):
                            has_future_child = True
                            break
                if not has_future_child:
                    # Safe to delete
                    red.remove(r)
                    schedule.append(['delete', r])
                    # timestamp.pop(r, None)

    # Check if all outputs are completed
    if completed != output_set:
        return None
        
    return schedule

def _select_victim(red, completed, children, timestamp):
    """
    Select a victim for eviction.
    Heuristic: Minimize (future_children, timestamp).
    Future children: children of the node that are not completed.
    Note: This is an approximation. Ideally we'd check if they are in the future,
    but 'not completed' is a good proxy for 'might be needed'.
    """
    min_val = (float('inf'), float('inf'))
    victim = None
    
    for r in red:
        if r in completed:
            continue
        # Count children not completed
        fut = len([c for c in children[r] if c not in completed])
        t = timestamp.get(r, 0)
        
        val = (fut, t)
        if val < min_val:
            min_val = val
            victim = r
            
    return victim
# EVOLVE-BLOCK-END
