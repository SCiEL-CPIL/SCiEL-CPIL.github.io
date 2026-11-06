import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    # Robustly parse DAG
    # Assume dag is a dict with 'parents', 'outputs', 'inputs', 'n'
    # Or a list of parents.
    
    # Determine n
    if isinstance(dag, dict):
        n = dag.get('n', dag.get('num_vertices', None))
        parents_list = dag.get('parents', dag.get('edges', []))
        outputs = dag.get('outputs', dag.get('sinks', []))
        inputs = dag.get('inputs', dag.get('sources', []))
    else:
        # If dag is just a list of parents
        parents_list = dag
        n = len(parents_list)
        outputs = []
        inputs = []
        
    if n is None:
        n = len(parents_list)
        
    # Normalize parents_list to list of lists
    if not isinstance(parents_list, list):
        parents_list = [list(parents_list)] if parents_list else []
        
    # Ensure parents_list has length n
    while len(parents_list) < n:
        parents_list.append([])
    parents_list = parents_list[:n]
    
    # Build children and in-degrees
    children = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in parents_list[i]:
            if 0 <= p < n:
                children[p].append(i)
                in_degree[i] += 1
                
    # Identify inputs and outputs if not provided
    if not inputs:
        inputs = [i for i in range(n) if in_degree[i] == 0]
    if not outputs:
        outputs = [i for i in range(n) if len(children[i]) == 0]
        
    input_set = set(inputs)
    output_set = set(outputs)
    
    # Topological Sort
    in_deg_copy = in_degree[:]
    queue = [i for i in range(n) if in_deg_copy[i] == 0]
    topo_order = []
    while queue:
        # Use heap for deterministic order if needed, but list is fine
        v = queue.pop(0)
        topo_order.append(v)
        for c in children[v]:
            in_deg_copy[c] -= 1
            if in_deg_copy[c] == 0:
                queue.append(c)
                
    # If graph has cycles or is disconnected, add remaining
    if len(topo_order) < n:
        visited = set(topo_order)
        for i in range(n):
            if i not in visited:
                topo_order.append(i)
                
    # Precompute future usage counts for eviction strategy
    # future_usage[v] = number of times v is needed as a parent in the remaining topo order
    # We can approximate this by just counting children that haven't been processed yet.
    # A better static heuristic: Count total children.
    # Let's use a dynamic "remaining children" count.
    remaining_children = [len(children[i]) for i in range(n)]
    
    fast_mem = set()
    on_disk = set(inputs) # Inputs start blue (on disk)
    computed = set()
    ops = []
    
    def get_eviction_candidate():
        # Find vertex in fast_mem with lowest remaining_children
        # If tie, pick any.
        best_v = None
        best_count = float('inf')
        for v in fast_mem:
            if remaining_children[v] < best_count:
                best_count = remaining_children[v]
                best_v = v
        return best_v

    def evict():
        nonlocal fast_mem, on_disk, ops, remaining_children
        if not fast_mem:
            return
        v = get_eviction_candidate()
        if v is not None:
            fast_mem.discard(v)
            on_disk.add(v)
            ops.append(["store", v])
            
    # Step 1: Process vertices in topo order
    for v in topo_order:
        if v in input_set:
            # Input starts blue. We don't need to "compute" it, but we might need to load it if it's a parent of something.
            # However, the loop below handles loading parents.
            # If v is an input and has no children, it doesn't need to be loaded unless it's an output?
            # If it's an output, it must end blue. It starts blue. So if we never touch it, it stays blue.
            # So we can skip inputs that are not needed for computation?
            # But if v is an input AND an output, and we skip it, it remains blue. That is valid.
            # If v is an input but NOT an output, and has children, we will load it when we process the first child.
            continue
            
        # Ensure parents are in fast_mem
        for p in parents_list[v]:
            if p not in fast_mem:
                # p is either on_disk or uncomputed (should not be uncomputed if topo order is correct)
                # If p is uncomputed, it's a bug in topo order or graph structure.
                # Assume p is on_disk (blue).
                if len(fast_mem) >= fast_memory_budget:
                    evict()
                fast_mem.add(p)
                on_disk.discard(p)
                ops.append(["load", p])
                
        # Compute v
        if len(fast_mem) >= fast_memory_budget:
            evict()
            
        fast_mem.add(v)
        computed.add(v)
        ops.append(["compute", v])
        
        # Update remaining_children for parents?
        # Actually, when we compute v, we have "used" the parents.
        # We can decrement remaining_children of parents?
        # No, remaining_children[v] is the number of children of v.
        # When we process v, we "satisfy" v's role as a parent? No, v is the child.
        # The parents' "remaining children" count should decrease by 1 for each child processed.
        for p in parents_list[v]:
            if remaining_children[p] > 0:
                remaining_children[p] -= 1

    # Step 2: Ensure all outputs are blue (on disk)
    for v in outputs:
        if v in fast_mem:
            fast_mem.discard(v)
            on_disk.add(v)
            ops.append(["store", v])
            
    # Note: If an output was never computed (e.g. it's an input with no children), 
    # it is in on_disk (initial state) and not in fast_mem. So it is already blue.
    # This is correct.
    
    return ops
