"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque
import sys
from functools import lru_cache

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

def _compute_layered_dag_schedule(nodes, parents, inputs, outputs, fast_memory_budget):
    """
    Compute a schedule for a layered DAG using dynamic programming.
    Assumes the DAG is layered (inputs -> layer1 -> layer2 -> ... -> outputs).
    """
    # Identify layers
    # Inputs are layer 0, outputs are the last layer
    # Nodes in between are intermediate layers
    
    # For a layered DAG, we can process layer by layer.
    # Within each layer, we need to compute all nodes.
    # We can keep the previous layer's results in fast memory if budget allows.
    
    # Determine layers
    # A node's layer is 1 + max(layer(parents))
    # Inputs have layer 0
    
    layer_of = {}
    for n in nodes:
        if n in inputs:
            layer_of[n] = 0
        else:
            max_parent_layer = -1
            for p in parents.get(n, []):
                if p in layer_of:
                    max_parent_layer = max(max_parent_layer, layer_of[p])
            layer_of[n] = max_parent_layer + 1 if max_parent_layer >= 0 else 0
    
    # Group nodes by layer
    from collections import defaultdict
    layers = defaultdict(list)
    for n in nodes:
        layers[layer_of[n]].append(n)
    
    max_layer = max(layers.keys()) if layers else 0
    
    # For each layer, we need to compute all nodes.
    # We can keep the previous layer's results in fast memory.
    # If budget is large enough to hold a full layer, we can avoid reloading.
    
    # Let's assume the DAG is a tree or has a simple structure.
    # For now, use a greedy approach that keeps the most recently used nodes.
    
    red = set()
    completed = set()
    schedule = []
    transfers = 0
    
    output_set = set(outputs)
    
    # Process nodes in topological order, but with a better eviction strategy
    # Use a priority queue to decide which nodes to keep in fast memory
    
    # For now, just use the existing greedy approach but with a better heuristic
    # We'll improve the eviction score to consider the "depth" of remaining uses
    
    return None  # Placeholder, will use the main pebble function's logic


def pebble(dag, fast_memory_budget):
    """
    Improved pebbling with better A* heuristic and greedy fallback.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    input_set = set(inputs)
    output_set = set(outputs)
    all_nodes = set(nodes)
    n_nodes = len(nodes)
    
    # Precompute children map
    children = {n: [] for n in nodes}
    for n in nodes:
        for p in parents.get(n, []):
            children[p].append(n)
            
    # Topological sort
    in_degree = {n: len(parents.get(n, [])) for n in nodes}
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
    
    # Precompute for each node:
    # - remaining_uses: number of children not yet completed (dynamic)
    # - is_ancestor_of: for heuristic
    
    # For A*, we need a good heuristic. Let's use:
    # h = (number of remaining outputs) + (number of inputs not in red that are needed)
    # More refined: count the minimum number of loads needed
    
    # For each node, precompute the set of descendants
    desc_cache = {}
    def get_descendants(n):
        if n in desc_cache:
            return desc_cache[n]
        desc = set()
        for c in children.get(n, []):
            desc.add(c)
            desc |= get_descendants(c)
        desc_cache[n] = desc
        return desc
    
    for n in nodes:
        get_descendants(n)
    
    # Precompute for each node, the set of ancestors (for quick check)
    anc_cache = {}
    def get_ancestors(n):
        if n in anc_cache:
            return anc_cache[n]
        anc = set()
        for p in parents.get(n, []):
            anc.add(p)
            anc |= get_ancestors(p)
        anc_cache[n] = anc
        return anc
    
    for n in nodes:
        get_ancestors(n)
    
    # --- Attempt 1: A* Search for Optimal Schedule ---
    # Use A* if the DAG is small enough
    if n_nodes <= 25 and fast_memory_budget <= 8:
        try:
            # State: (frozenset of red nodes, frozenset of completed outputs)
            # Cost: number of transfers (load + store)
            
            def heuristic(red_frozen, completed_frozen):
                """
                Lower bound on remaining transfers.
                """
                remaining_outputs = output_set - completed_frozen
                if not remaining_outputs:
                    return 0
                
                # Each remaining output needs at least 1 store
                cost = len(remaining_outputs)
                
                # For each remaining output, we need its ancestors in fast memory.
                # Count the number of ancestors not already in red.
                # This is a lower bound on loads needed.
                needed_ancestors = set()
                for out in remaining_outputs:
                    needed_ancestors |= get_ancestors(out)
                
                # Inputs that need to be loaded
                inputs_needed = input_set & needed_ancestors
                inputs_not_in_red = inputs_needed - red_frozen
                cost += len(inputs_not_in_red)
                
                return cost
            
            initial_state = (frozenset(), frozenset())
            
            g_score = {initial_state: 0}
            parent_map = {initial_state: None}
            h0 = heuristic(*initial_state)
            pq = [(h0, 0, initial_state)]  # (f, g, state)
            
            found_goal = False
            goal_state = None
            iterations = 0
            
            while pq:
                iterations += 1
                if time.time() > deadline - 5.0:
                    break
                    
                f, g, state = heapq.heappop(pq)
                
                if g > g_score.get(state, float('inf')):
                    continue
                    
                red = set(state[0])
                completed = set(state[1])
                
                if completed == output_set:
                    found_goal = True
                    goal_state = state
                    best_cost = g
                    break
                
                # Generate neighbors
                
                # 1. Load an input (if not already in red and not completed)
                for inp in inputs:
                    if inp not in red and inp not in completed:
                        if len(red) < fast_memory_budget:
                            new_red = red | {inp}
                            new_state = (frozenset(new_red), state[1])
                            new_g = g + 1
                            if new_g < g_score.get(new_state, float('inf')):
                                g_score[new_state] = new_g
                                parent_map[new_state] = (state, 'load', inp)
                                h = heuristic(*new_state)
                                heapq.heappush(pq, (new_g + h, new_g, new_state))
                
                # 2. Compute a node (if all parents are in red)
                for node in nodes:
                    if node in completed or node in red:
                        continue
                    
                    can_compute = all(p in red for p in parents.get(node, []))
                    if can_compute:
                        if len(red) < fast_memory_budget:
                            new_red = red | {node}
                            new_state = (frozenset(new_red), state[1])
                            new_g = g  # compute is free
                            if new_g < g_score.get(new_state, float('inf')):
                                g_score[new_state] = new_g
                                parent_map[new_state] = (state, 'compute', node)
                                h = heuristic(*new_state)
                                heapq.heappush(pq, (new_g + h, new_g, new_state))
                        else:
                            # Need to evict one red node, then compute
                            for victim in red:
                                new_red = (red - {victim}) | {node}
                                new_state = (frozenset(new_red), state[1])
                                new_g = g  # delete is free
                                if new_g < g_score.get(new_state, float('inf')):
                                    g_score[new_state] = new_g
                                    parent_map[new_state] = (state, 'delete', victim)
                                    h = heuristic(*new_state)
                                    heapq.heappush(pq, (new_g + h, new_g, new_state))
                
                # 3. Store an output (if in red and not completed)
                for node in nodes:
                    if node in red and node in output_set and node not in completed:
                        new_red = red - {node}
                        new_completed = completed | {node}
                        new_state = (frozenset(new_red), frozenset(new_completed))
                        new_g = g + 1
                        if new_g < g_score.get(new_state, float('inf')):
                            g_score[new_state] = new_g
                            parent_map[new_state] = (state, 'store', node)
                            h = heuristic(*new_state)
                            heapq.heappush(pq, (new_g + h, new_g, new_state))
                
                # 4. Delete a red node (to free space)
                for node in red:
                    new_red = red - {node}
                    new_state = (frozenset(new_red), state[1])
                    new_g = g  # delete is free
                    if new_g < g_score.get(new_state, float('inf')):
                        g_score[new_state] = new_g
                        parent_map[new_state] = (state, 'delete', node)
                        h = heuristic(*new_state)
                        heapq.heappush(pq, (new_g + h, new_g, new_state))

            if found_goal and goal_state:
                # Reconstruct path
                path = []
                curr = goal_state
                while parent_map[curr] is not None:
                    prev_state, op, node = parent_map[curr]
                    path.append([op, node])
                    curr = prev_state
                path.reverse()
                return path

        except Exception:
            pass # Fall through to greedy

    # --- Attempt 2: Improved Greedy with Better Eviction Heuristic ---
    
    red = set()
    completed = set()
    schedule = []
    transfers = 0
    
    # Track last access time for LRU
    timestamp = {}
    current_time = 0
    
    # Precompute: for each node, count of remaining children (not completed)
    # We'll update this dynamically
    
    def get_eviction_score(node, red_set, completed_set, current_node):
        """
        Score a node for eviction. Lower is better to evict.
        Factors:
        1. Number of remaining children (fewer = better to evict)
        2. Whether it's an ancestor of current_node (avoid if so)
        3. LRU timestamp (older = better to evict)
        """
        # Count children not yet completed
        remaining_children = 0
        for c in children.get(node, []):
            if c not in completed_set:
                remaining_children += 1
        
        # Is this node an ancestor of current_node? (i.e., will it be needed soon?)
        # If current_node is in descendants of node, then node is an ancestor
        if current_node in get_descendants(node):
            ancestor_penalty = 1000  # Strong penalty for evicting ancestors of current node
        else:
            ancestor_penalty = 0
        
        # LRU: older is better to evict (lower timestamp)
        lru_score = timestamp.get(node, 0)
        
        # Combine: prioritize fewer remaining children, avoid ancestors, prefer older
        score = (remaining_children, ancestor_penalty, lru_score)
        return score
    
    for node in topo_order:
        if time.time() > deadline:
            break
            
        # 1. Ensure parents are in red
        for p in parents.get(node, []):
            if p not in red:
                # Need to load p.
                # If budget is full, evict.
                if len(red) >= fast_memory_budget:
                    # Find victim using improved heuristic
                    best_victim = None
                    best_score = None
                    for r in red:
                        if r in completed:
                            continue
                        score = get_eviction_score(r, red, completed, node)
                        if best_score is None or score < best_score:
                            best_victim = r
                            best_score = score
                    
                    if best_victim:
                        red.remove(best_victim)
                        schedule.append(['delete', best_victim])
                        if best_victim in timestamp:
                            del timestamp[best_victim]
                
                red.add(p)
                timestamp[p] = current_time
                current_time += 1
                schedule.append(['load', p])
                transfers += 1
        
        # 2. Compute node
        if node not in red:
            if len(red) >= fast_memory_budget:
                # Evict
                best_victim = None
                best_score = None
                for r in red:
                    if r in completed:
                        continue
                    score = get_eviction_score(r, red, completed, node)
                    if best_score is None or score < best_score:
                        best_victim = r
                        best_score = score
                
                if best_victim:
                    red.remove(best_victim)
                    schedule.append(['delete', best_victim])
                    if best_victim in timestamp:
                        del timestamp[best_victim]
            
            red.add(node)
            timestamp[node] = current_time
            current_time += 1
            schedule.append(['compute', node])
        
        # 3. If node is output, store it
        if node in output_set:
            if node in red:
                red.remove(node)
            completed.add(node)
            schedule.append(['store', node])
            transfers += 1
            if node in timestamp:
                del timestamp[node]
    
    # Check if all outputs are completed
    if completed != output_set:
        # Fallback if we didn't finish (e.g. time limit)
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Verify schedule length
    if len(schedule) > 20000:
        # Truncate or handle gracefully
        pass
        
    return schedule

# EVOLVE-BLOCK-END
