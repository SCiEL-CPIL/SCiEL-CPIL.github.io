import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    
    # --- Parse DAG Structure ---
    # Assume dag is a dict {node: [parents]}
    # If it's a list of (node, parents), convert to dict
    if isinstance(dag, list):
        dag_dict = {item[0]: item[1] for item in dag}
        # Try to find outputs: nodes not in any parent list
        all_nodes = set(dag_dict.keys())
        parent_set = set()
        for parents in dag_dict.values():
            parent_set.update(parents)
        outputs = list(all_nodes - parent_set)
    elif isinstance(dag, dict):
        dag_dict = dag
        # Check if 'outputs' is a key
        if 'outputs' in dag_dict:
            outputs = dag_dict['outputs']
            # Remove metadata keys if present?
            # Assume keys are nodes if they are not 'outputs'
            # This is risky. Let's assume standard node->parents mapping.
            # If 'outputs' is a key, it might be the only metadata.
            # Let's filter keys that are not nodes.
            # Heuristic: Nodes are usually ints or strings. 'outputs' is a string.
            # But if nodes are strings, 'outputs' could be a node.
            # Let's assume if 'outputs' is present, it's a list of node IDs.
            # And the other keys are nodes.
            node_keys = [k for k in dag_dict.keys() if k != 'outputs']
            dag_dict = {k: dag_dict[k] for k in node_keys}
        else:
            # Infer outputs as nodes with no children
            all_nodes = set(dag_dict.keys())
            parent_set = set()
            for parents in dag_dict.values():
                parent_set.update(parents)
            outputs = list(all_nodes - parent_set)
    else:
        # Fallback: assume empty
        return []

    # Build children map
    children = defaultdict(list)
    for node, parents in dag_dict.items():
        for p in parents:
            children[p].append(node)
    
    # Identify Inputs (nodes with no parents)
    inputs = [n for n, p in dag_dict.items() if not p]
    
    # Pre-compute descendant count for heuristic
    # Topological order
    in_degree = {n: len(p) for n, p in dag_dict.items()}
    queue = [n for n in dag_dict if in_degree[n] == 0]
    topo_order = []
    temp_in_degree = in_degree.copy()
    while queue:
        n = queue.pop(0)
        topo_order.append(n)
        for c in children[n]:
            temp_in_degree[c] -= 1
            if temp_in_degree[c] == 0:
                queue.append(c)
                
    # Descendant count (number of nodes reachable from n)
    desc_count = {n: 0 for n in dag_dict}
    # Process in reverse topological order
    for n in reversed(topo_order):
        if children[n]:
            desc_count[n] = 1 + sum(desc_count[c] for c in children[n])
        else:
            desc_count[n] = 0
            
    # State
    blue_set = set(inputs)
    red_set = set(dag_dict.keys()) - blue_set
    empty_set = set()
    
    # Ready nodes: parents all in blue_set
    # We need to track which nodes are ready
    # A node is ready if all its parents are in blue_set
    
    # Priority for Ready nodes: Higher descendant count (more critical)
    # Priority for Eviction: Lower descendant count (less critical)
    
    operations = []
    
    # Initial state check
    # If inputs > budget, we have a problem. 
    # "Inputs start blue". If len(inputs) > fast_memory_budget, 
    # we must evict some inputs immediately? 
    # But they are needed for computation.
    # If we evict an input, we must load it later.
    # Let's assume budget >= len(inputs) or we handle overflow.
    
    if len(blue_set) > fast_memory_budget:
        # Evict non-critical inputs
        # Sort by descendant count
        sorted_inputs = sorted(inputs, key=lambda x: desc_count[x])
        while len(blue_set) > fast_memory_budget:
            victim = sorted_inputs.pop(0)
            if victim in blue_set:
                blue_set.remove(victim)
                red_set.add(victim)
                operations.append(['store', victim])
                
    # Main Loop
    max_ops = 20000
    op_count = 0
    
    # We need to track which nodes are "Ready"
    # Instead of recalculating every time, we can maintain a set
    # But for simplicity and correctness, we recalculate or update incrementally.
    # Given small size, recalculation is safe.
    
    while op_count < max_ops:
        if time.time() - start_time > 50:
            break
            
        # Check if done: all outputs are in blue_set
        if all(o in blue_set for o in outputs):
            break
            
        # Find Ready nodes
        ready_nodes = []
        for node in dag_dict:
            if node in red_set or node in empty_set:
                if all(p in blue_set for p in dag_dict[node]):
                    ready_nodes.append(node)
                    
        if ready_nodes:
            # Choose best ready node
            # Priority: Highest descendant count (unlocks more)
            ready_nodes.sort(key=lambda x: desc_count[x], reverse=True)
            v = ready_nodes[0]
            
            # Need to compute v
            # If blue_set is full, evict someone
            if len(blue_set) >= fast_memory_budget:
                # Find victim
                # Cannot evict:
                # 1. Outputs
                # 2. Parents of any ready node? (They are already blue, but if we evict, we break future readiness? 
                #    Actually, if we evict a parent of a *future* ready node, that node won't be ready.
                #    But we are about to compute v. v's parents are blue.
                #    Evicting a node u that is a parent of some other node w (not v) might be bad if w is close to ready.
                #    Heuristic: Evict node with min desc_count, excluding outputs.
                
                candidates = [n for n in blue_set if n not in outputs]
                if not candidates:
                    # Must evict an output? Bad. But maybe necessary if budget is tiny.
                    # Or maybe we should have loaded differently.
                    # For now, pick any.
                    candidates = list(blue_set)
                    
                candidates.sort(key=lambda x: desc_count[x])
                victim = candidates[0]
                
                blue_set.remove(victim)
                red_set.add(victim)
                operations.append(['store', victim])
                op_count += 1
                if op_count >= max_ops:
                    break
                
            # Compute v
            blue_set.add(v)
            red_set.discard(v)
            operations.append(['compute', v])
            op_count += 1
            
        else:
            # No ready nodes. Need to load a red node.
            # Find a red node u such that loading it makes some node w ready.
            # i.e., w has all parents blue except u.
            
            candidates = []
            for node in red_set:
                # Check if loading node makes any child ready
                # A child c is ready if all parents of c are in blue_set U {node}
                # i.e., all parents of c except node are in blue_set
                for c in children[node]:
                    if c in red_set or c in empty_set:
                        # Check if c is "almost ready"
                        # All parents of c except node must be blue
                        almost_ready = True
                        for p in dag_dict[c]:
                            if p != node and p not in blue_set:
                                almost_ready = False
                                break
                        if almost_ready:
                            # Loading node makes c ready
                            # Priority: desc_count of c?
                            candidates.append((node, desc_count[c]))
                            break # One child is enough to make it a candidate?
                            # Actually, we want to load the node that unblocks the most critical path.
                            # Let's collect all potential unlocks.
            
            if not candidates:
                # No node can be loaded to make progress?
                # This implies a deadlock or we missed something.
                # If we are stuck, maybe we need to load a node that is not a parent of an almost-ready node?
                # No, that won't help.
                # If we are stuck, it might be that we evicted something we needed.
                # Fallback: Load any red node with max desc_count.
                candidates = [(n, desc_count[n]) for n in red_set]
                if not candidates:
                    break # No red nodes? Then we are done or stuck.
            
            # Sort by unlock criticality
            candidates.sort(key=lambda x: x[1], reverse=True)
            u = candidates[0][0]
            
            # If blue_set is full, evict
            if len(blue_set) >= fast_memory_budget:
                candidates_evict = [n for n in blue_set if n not in outputs]
                if not candidates_evict:
                    candidates_evict = list(blue_set)
                candidates_evict.sort(key=lambda x: desc_count[x])
                victim = candidates_evict[0]
                blue_set.remove(victim)
                red_set.add(victim)
                operations.append(['store', victim])
                op_count += 1
                if op_count >= max_ops:
                    break
                
            # Load u
            blue_set.add(u)
            red_set.discard(u)
            operations.append(['load', u])
            op_count += 1

    return operations
