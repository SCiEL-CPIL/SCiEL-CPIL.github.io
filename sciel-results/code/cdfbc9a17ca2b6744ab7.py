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
    Improved greedy pebbling with depth-aware cost, precomputed urgency,
    and a local search refinement pass (2-opt style swaps).
    """
    deadline = time.time() + 85.0  # Safety margin
    
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
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # Precompute topo index
    topo_index = {n: i for i, n in enumerate(topo_order)}
    
    # Precompute Depth (longest path from input)
    depth = {n: 0 for n in nodes}
    for n in topo_order:
        if parents.get(n):
            max_parent_depth = max(depth[p] for p in parents[n])
            depth[n] = max_parent_depth + 1
    
    # Precompute next use indices for each node
    children_indices = {n: [] for n in nodes}
    for n in nodes:
        for c in children[n]:
            children_indices[n].append(topo_index[c])
        children_indices[n].sort()
    
    # Pointer for each node to the next child to be processed
    next_child_ptr = {n: 0 for n in nodes}
    
    red = set()
    completed = set()
    schedule = []
    transfers = 0
    
    timestamp = {}
    current_time = 0
    
    def get_eviction_score(node, current_idx):
        """
        Lower score = better candidate for eviction.
        Score = RecomputeCost * Urgency
        """
        # Recompute Cost
        if node in input_set:
            recompute_cost = 1.0
        else:
            recompute_cost = float(depth[node] + 1)
            
        # Urgency
        ptr = next_child_ptr[node]
        ch_idx = children_indices[node]
        
        while ptr < len(ch_idx) and ch_idx[ptr] <= current_idx:
            ptr += 1
        next_child_ptr[node] = ptr
        
        if ptr < len(ch_idx):
            next_use_idx = ch_idx[ptr]
            distance = next_use_idx - current_idx
            if distance <= 0:
                urgency = 1.0
            else:
                urgency = 1.0 / distance
        else:
            urgency = 0.0
            
        score = recompute_cost * urgency
        score += timestamp.get(node, 0) * 1e-9
        
        return score

    # --- Phase 1: Greedy construction ---
    for idx, node in enumerate(topo_order):
        if time.time() > deadline:
            break
            
        # 1. Ensure parents are in red
        for p in parents.get(node, []):
            if p not in red:
                if len(red) >= fast_memory_budget:
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
                
                red.add(p)
                timestamp[p] = current_time
                current_time += 1
                schedule.append(['load', p])
                transfers += 1
        
        # 2. Compute node
        if node not in red:
            if len(red) >= fast_memory_budget:
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
            if node in red:
                red.remove(node)
            completed.add(node)
            schedule.append(['store', node])
            transfers += 1
    
    if completed != output_set:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    # --- Phase 2: Local search refinement ---
    # Try to reduce transfers by swapping evicted nodes.
    # We simulate the schedule and look for opportunities to keep a node
    # that is used soon, by evicting a node that is used later.
    
    def simulate_schedule(sched):
        """Simulate schedule, return (transfers, valid)."""
        red_sim = set()
        completed_sim = set()
        transfers_sim = 0
        for op, v in sched:
            if op == 'load':
                if v in red_sim:
                    continue
                if len(red_sim) >= fast_memory_budget:
                    return None, False  # invalid
                red_sim.add(v)
                transfers_sim += 1
            elif op == 'store':
                if v not in red_sim:
                    return None, False
                red_sim.remove(v)
                completed_sim.add(v)
                transfers_sim += 1
            elif op == 'compute':
                # Check parents
                for p in parents.get(v, []):
                    if p not in red_sim:
                        return None, False
                if v not in red_sim:
                    if len(red_sim) >= fast_memory_budget:
                        return None, False
                    red_sim.add(v)
            elif op == 'delete':
                if v in red_sim:
                    red_sim.remove(v)
        if completed_sim != output_set:
            return None, False
        return transfers_sim, True
    
    # Initial transfers
    init_transfers, valid = simulate_schedule(schedule)
    if not valid:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    best_transfers = init_transfers
    best_schedule = schedule[:]
    
    # Local search: try to swap two 'delete' operations
    # This is a heuristic improvement; we only try a few swaps to stay within time.
    delete_indices = [i for i, (op, v) in enumerate(schedule) if op == 'delete']
    
    max_swaps = min(50, len(delete_indices) * (len(delete_indices) - 1) // 2)
    swaps_done = 0
    
    for i in range(len(delete_indices)):
        if time.time() > deadline - 5.0:
            break
        if swaps_done >= max_swaps:
            break
        for j in range(i + 1, len(delete_indices)):
            if time.time() > deadline - 5.0:
                break
            if swaps_done >= max_swaps:
                break
            
            idx1 = delete_indices[i]
            idx2 = delete_indices[j]
            
            # Try swapping the vertices of these two delete operations
            # This effectively changes which node is evicted at each step.
            # We need to check if the resulting schedule is valid.
            
            new_sched = best_schedule[:]
            op1, v1 = new_sched[idx1]
            op2, v2 = new_sched[idx2]
            
            # Swap the vertices
            new_sched[idx1] = [op1, v2]
            new_sched[idx2] = [op2, v1]
            
            # But wait, swapping delete vertices might break the logic if
            # the node being deleted at idx1 is not in red, or the node being
            # deleted at idx2 is not in red.
            # The simulation will catch this.
            
            t, valid_new = simulate_schedule(new_sched)
            if valid_new and t < best_transfers:
                best_transfers = t
                best_schedule = new_sched
                swaps_done += 1
    
    return best_schedule
# EVOLVE-BLOCK-END
