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
    Improved greedy pebbling with cost-aware eviction.
    """
    deadline = time.time() + 50.0
    
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
    
    # Precompute topo index for each node
    topo_index = {n: i for i, n in enumerate(topo_order)}
    
    # For each node, precompute the list of children in topological order
    children_topo = {}
    for n in nodes:
        ch = children[n]
        ch.sort(key=lambda x: topo_index.get(x, float('inf')))
        children_topo[n] = ch
    
    # Precompute for each node: the earliest child that requires it (in topo order)
    # This helps with urgency scoring
    # For a node v, its "next need" is the minimum topo_index among its children not yet processed
    
    red = set()
    completed = set()
    schedule = []
    transfers = 0
    
    # Track when nodes were loaded/computed for LRU tie-breaking
    timestamp = {}
    current_time = 0
    
    # For eviction: we want to evict the node with the lowest "value"
    # Value = urgency * cost_to_recompute
    # - Inputs: cost_to_recompute = 1 (just load)
    # - Computed nodes: cost_to_recompute = 1 + sum of costs to recompute parents
    #   Approximation: use the number of ancestors that need to be recomputed
    # Urgency: how soon is this node needed again?
    #   If never needed again, urgency = 0
    #   If needed soon, urgency is high
    
    def get_eviction_score(node, current_topo_pos):
        """
        Compute eviction score for a node.
        Lower score = better candidate for eviction.
        """
        if node in completed:
            return 0  # Already completed, definitely evict if needed
        
        # Is this an input?
        is_input = node in input_set
        
        # Cost to recompute/reload
        if is_input:
            recompute_cost = 1
        else:
            # Approximate cost: 1 for compute + cost of parents
            # Use a simple heuristic: number of parents + 1
            recompute_cost = len(parents.get(node, [])) + 1
        
        # Urgency: when is this node next needed?
        # Find the next child in topo_order that is after current position
        next_need = float('inf')
        for c in children_topo.get(node, []):
            c_idx = topo_index.get(c, float('inf'))
            if c_idx > current_topo_pos and c not in completed:
                if c_idx < next_need:
                    next_need = c_idx
        
        if next_need == float('inf'):
            # Never needed again
            urgency = 0
        else:
            # Urgency decreases with distance
            urgency = 1.0 / (1 + (next_need - current_topo_pos))
        
        # Combine: we want to evict nodes with low urgency and low recompute cost
        # Score = urgency * recompute_cost (lower is better to evict)
        score = urgency * recompute_cost
        
        # Tie-break by LRU (older = lower timestamp = better to evict)
        t = timestamp.get(node, 0)
        # Add small penalty for recent usage
        score += (1 - t / (current_time + 1)) * 0.1
        
        return score
    
    for idx, node in enumerate(topo_order):
        if time.time() > deadline:
            break
            
        # 1. Ensure parents are in red
        for p in parents.get(node, []):
            if p not in red:
                # Load parent
                if len(red) >= fast_memory_budget:
                    # Evict: find node with lowest eviction score
                    min_score = float('inf')
                    victim = None
                    for r in red:
                        s = get_eviction_score(r, idx)
                        if s < min_score:
                            min_score = s
                            victim = r
                    
                    if victim:
                        # Delete victim
                        red.remove(victim)
                        schedule.append(['delete', victim])
                        # No transfer cost for delete
                    
                red.add(p)
                timestamp[p] = current_time
                current_time += 1
                schedule.append(['load', p])
                transfers += 1
        
        # 2. Compute node
        if node not in red:
            if len(red) >= fast_memory_budget:
                # Evict: find node with lowest eviction score
                min_score = float('inf')
                victim = None
                for r in red:
                    s = get_eviction_score(r, idx)
                    if s < min_score:
                        min_score = s
                        victim = r
                
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
            # timestamp[node] is no longer relevant as it's not in red
    
    # Check if all outputs are completed
    if completed != output_set:
        # Fallback if we didn't finish (e.g. time limit)
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Verify schedule length
    if len(schedule) > 20000:
        # This shouldn't happen for reasonable DAGs with greedy
        pass
        
    return schedule
# EVOLVE-BLOCK-END
