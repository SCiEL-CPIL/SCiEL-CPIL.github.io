import time
import heapq
from collections import defaultdict

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
    deadline = time.time() + 85.0
    
    try:
        nodes, parents, inputs, outputs = _extract_dag_info(dag)
    except Exception:
        # Fallback
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

    input_set = set(inputs)
    output_set = set(outputs)
    all_nodes = set(nodes)
    
    # Build children map
    children = {n: [] for n in all_nodes}
    for n in all_nodes:
        for p in parents.get(n, []):
            children[p].append(n)
    
    # Topological Sort
    in_degree = {n: len(parents.get(n, [])) for n in all_nodes}
    queue = [n for n in all_nodes if in_degree[n] == 0]
    topo_order = []
    while queue:
        # Use a list, pop from front for stability
        n = queue.pop(0)
        topo_order.append(n)
        for c in children[n]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo_order) != len(all_nodes):
        # Cycle detected or error, fallback
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)

    ops = []
    red = set()
    computed = set()
    
    def evict_if_needed():
        nonlocal red, ops
        while len(red) > fast_memory_budget:
            if time.time() > deadline:
                return
            # Find a node to evict
            # Priority 1: Node that is not a parent of any uncomputed node
            # Priority 2: Node with fewest uncomputed children
            
            candidates = []
            for u in red:
                # Check if u is needed for any uncomputed node
                needed = False
                uncomputed_children = 0
                for c in children[u]:
                    if c not in computed:
                        needed = True
                        uncomputed_children += 1
                
                if not needed:
                    # This node is done. Safe to evict.
                    candidates.append((0, uncomputed_children, u))
                else:
                    # This node is still needed. Evicting it will cause future loads.
                    # We prefer to evict these only if no "done" nodes exist.
                    candidates.append((1, uncomputed_children, u))
            
            if not candidates:
                # Should not happen if budget >= 1
                return

            # Sort by priority (0 is better), then by uncomputed children (fewer is better? 
            # Actually, if we evict a node with 0 uncomputed children, it's free.
            # If we must evict a needed node, we should evict the one with the LEAST future impact?
            # Or the one with the MOST? 
            # If we evict a node with 1 child, we load it once.
            # If we evict a node with 10 children, we load it 10 times (or recompute).
            # So we should evict the one with the FEWEST uncomputed children among the "needed" ones.
            # But "done" nodes (priority 0) are always better.
            
            candidates.sort(key=lambda x: (x[0], x[1]))
            
            _, _, u = candidates[0]
            
            if u in output_set:
                ops.append(['store', u])
            else:
                ops.append(['delete', u])
            red.remove(u)
            
    # Process nodes in topological order
    for v in topo_order:
        if time.time() > deadline:
            break
            
        if v in input_set:
            if v not in red:
                ops.append(['load', v])
                red.add(v)
                computed.add(v)
        else:
            # Ensure parents are in red
            for p in parents.get(v, []):
                if p not in red:
                    # Load it
                    ops.append(['load', p])
                    red.add(p)
                    computed.add(p)
                    # Check if we are over budget after loading
                    evict_if_needed()
            
            # Compute v
            ops.append(['compute', v])
            red.add(v)
            computed.add(v)
            
            # Check if we are over budget after computing
            evict_if_needed()

    # At the end, ensure all outputs are stored (blue)
    # The evict_if_needed loop stores outputs when they are evicted.
    # But if an output is still in `red` at the end, it is not "blue" yet.
    # We must store all remaining outputs.
    for v in output_set:
        if v in red:
            ops.append(['store', v])
            red.remove(v)
        # If v is not in red, it was already stored or deleted.
        # If it was deleted, it's not blue. 
        # Wait, "all outputs must finish blue".
        # If we deleted an output, we lost it.
        # My eviction logic: "if u in output_set: store".
        # So outputs are NEVER deleted, only stored.
        # So if an output is not in red, it must have been stored.
        # So this loop is just to catch any outputs still in red.

    if len(ops) <= 20000:
        return ops
        
    # Fallback if too many ops (should not happen for reasonable DAGs)
    from autoresearch_suite.science_tcs_expansion import pebble_baseline
    return pebble_baseline(dag, fast_memory_budget)
