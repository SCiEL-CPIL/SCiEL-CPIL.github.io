"""
Rewritten pebbling solver using Recursive Black-Red Pebbling for layered DAGs
and Greedy Dynamic Eviction for general DAGs.
"""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin

    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', []))
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', dag.get('sinks', []))
        inputs = dag.get('inputs', dag.get('sources', []))
    elif isinstance(dag, (list, tuple)):
        vertices = list(range(len(dag)))
        parents = {}
        outputs = []
        inputs = []
        for i, v in enumerate(dag):
            if isinstance(v, dict):
                p = v.get('parents', v.get('deps', []))
                parents[i] = list(p) if p else []
            else:
                parents[i] = []
    else:
        return []

    if not vertices:
        return []

    # Build adjacency
    n = len(vertices)
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)

    # Topological sort
    in_degree = defaultdict(int)
    for v in vertices:
        for p in parents.get(v, []):
            in_degree[v] += 1

    topo_order = []
    queue = deque(v for v in vertices if in_degree[v] == 0)
    while queue:
        v = queue.popleft()
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    if len(topo_order) != len(vertices):
        return [] # Cycle or invalid

    # Identify inputs and outputs
    input_set = set(inputs) if inputs else set(v for v in vertices if in_degree[v] == 0)
    output_set = set(outputs) if outputs else set(v for v in vertices if len(children[v]) == 0)
    
    # If no outputs defined, all sinks are outputs
    if not output_set:
        output_set = set(v for v in vertices if len(children[v]) == 0)

    # Strategy 1: Recursive Black-Red for Layered DAGs
    # Check if DAG is layered
    is_layered, layers = check_layered(dag, parents, children, topo_order)
    
    ops = []
    if is_layered:
        ops = pebble_layered(layers, parents, children, input_set, output_set, fast_memory_budget, deadline)
    else:
        ops = pebble_greedy_dynamic(topo_order, parents, children, input_set, output_set, fast_memory_budget, deadline)

    # Post-processing: Clean up redundant ops
    if ops:
        cleaned = []
        for op in ops:
            if op[0] in ('load', 'store') and cleaned:
                last_op = cleaned[-1]
                if op[1] == last_op[1]:
                    if op[0] == 'load' and last_op[0] == 'store':
                        cleaned.pop()
                        continue
                    elif op[0] == 'store' and last_op[0] == 'load':
                        cleaned.pop()
                        continue
                    elif op[0] == 'load' and last_op[0] == 'load':
                        continue
                    elif op[0] == 'store' and last_op[0] == 'store':
                        continue
            cleaned.append(op)
        ops = cleaned

    if len(ops) > 20000:
        ops = ops[:20000]

    return ops

def check_layered(dag, parents, children, topo_order):
    """Check if DAG can be divided into layers where all edges go from layer i to i+1."""
    if not topo_order:
        return False, []
    
    # Assign layer based on longest path from source
    layer = {}
    for v in topo_order:
        if not parents.get(v, []):
            layer[v] = 0
        else:
            max_p_layer = -1
            for p in parents[v]:
                if p in layer:
                    if layer[p] > max_p_layer:
                        max_p_layer = layer[p]
            if max_p_layer == -1:
                return False, []
            layer[v] = max_p_layer + 1
            
    # Verify all edges go from i to i+1
    for v in topo_order:
        for c in children[v]:
            if layer[c] != layer[v] + 1:
                return False, []
                
    # Group by layer
    max_layer = max(layer.values()) if layer else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in topo_order:
        layers[layer[v]].append(v)
        
    return True, layers

def pebble_layered(layers, parents, children, input_set, output_set, budget, deadline):
    """
    Recursive pebbling for layered DAGs.
    Uses Black-Red pebbling rules to minimize memory and transfers.
    """
    # If total vertices fit in memory, do a simple forward pass
    total_verts = sum(len(l) for l in layers)
    if total_verts <= budget:
        return simple_forward_pass(layers, parents, children, input_set, output_set, budget, deadline)
    
    # Find a cut layer to split the problem
    # Ideally, the cut layer size is <= budget
    # We want to split into two roughly equal parts
    mid = len(layers) // 2
    
    # Find a layer index 'cut_idx' such that len(layers[cut_idx]) <= budget
    # If no such layer, we must recompute heavily
    cut_idx = -1
    for i in range(len(layers)):
        if len(layers[i]) <= budget:
            cut_idx = i
            break
            
    if cut_idx == -1:
        # No single layer fits. Use the middle layer as a "virtual cut" 
        # and recompute everything in the bottom half multiple times?
        # Fallback to greedy for this specific case if structure is bad
        # But let's try to use the middle layer as the cut, even if it's large,
        # by processing it in chunks? 
        # For now, fallback to greedy if no layer fits.
        all_verts = [v for l in layers for v in l]
        return pebble_greedy_dynamic(all_verts, parents, children, input_set, output_set, budget, deadline)

    # Recursive Step:
    # G_bottom: layers[0] to layers[cut_idx]
    # G_top: layers[cut_idx] to layers[-1]
    # Cut vertices: layers[cut_idx]
    
    # 1. Pebble Bottom to get Cut (Red)
    ops_bottom_red = pebble_layered(layers[:cut_idx+1], parents, children, input_set, set(layers[cut_idx]), budget, deadline)
    
    # 2. Pebble Top to get Outputs (Red)
    # Inputs to Top are Cut. Outputs are final outputs.
    # Note: The 'input_set' for the top part is layers[cut_idx]
    ops_top_red = pebble_layered(layers[cut_idx:], parents, children, set(layers[cut_idx]), output_set, budget, deadline)
    
    # 3. Pebble Bottom to get Cut (Blue)
    ops_bottom_blue = pebble_layered(layers[:cut_idx+1], parents, children, input_set, set(layers[cut_idx]), budget, deadline)
    
    # 4. Pebble Top to get Outputs (Blue)
    ops_top_blue = pebble_layered(layers[cut_idx:], parents, children, set(layers[cut_idx]), output_set, budget, deadline)
    
    # Combine
    # We need to ensure the state is correct.
    # After step 1, Cut is Red (in slow mem).
    # Step 2 loads Cut, computes Top, stores Outputs (Red). Cut remains Red (or Blue if kept? 
    # In Black-Red, if we compute Top, we need Cut to be Blue. 
    # So Step 2 must Load Cut.
    # After Step 2, Cut might be Blue (if kept in mem) or Red (if evicted).
    # If kept in mem, Step 3 is redundant? 
    # Standard Black-Red:
    # 1. B(R) G_bottom -> Cut (R)
    # 2. B(R) G_top -> Outputs (R)  [Loads Cut]
    # 3. B(R) G_bottom -> Cut (B)   [Recomputes to get Cut Blue]
    # 4. B(R) G_top -> Outputs (B)  [Loads Cut, Computes Top, Stores Outputs Blue]
    
    # This sequence works.
    # However, if the Cut fits in memory, Step 3 might be avoidable?
    # If Cut fits in memory, we can keep it Blue after Step 1.
    # Then Step 2 uses it. Then Step 4 uses it.
    # So if len(Cut) <= budget, we can optimize:
    if len(layers[cut_idx]) <= budget:
        # 1. Pebble Bottom to get Cut (Blue)
        ops_bottom_blue = pebble_layered(layers[:cut_idx+1], parents, children, input_set, set(layers[cut_idx]), budget, deadline)
        # 2. Pebble Top to get Outputs (Blue)
        ops_top_blue = pebble_layered(layers[cut_idx:], parents, children, set(layers[cut_idx]), output_set, budget, deadline)
        return ops_bottom_blue + ops_top_blue
    else:
        return ops_bottom_red + ops_top_red + ops_bottom_blue + ops_top_blue

def simple_forward_pass(layers, parents, children, input_set, output_set, budget, deadline):
    """Simple forward pass assuming everything fits in memory or we don't care about eviction."""
    ops = []
    # Load all inputs
    for v in input_set:
        ops.append(['load', v])
        
    # Compute in layer order
    for layer in layers:
        for v in layer:
            # Parents are already computed and in memory (assumption of simple pass)
            ops.append(['compute', v])
            # Store if not output and not in next layer? 
            # In a simple pass with infinite memory, we don't store.
            # But we have a budget. If we assume it fits, we don't store.
            # If it doesn't fit, this function shouldn't be called.
            
    return ops

def pebble_greedy_dynamic(topo_order, parents, children, input_set, output_set, budget, deadline):
    """
    Greedy pebbling with dynamic remaining uses.
    """
    state = {}
    for v in topo_order:
        state[v] = 'blue' if v in input_set else 'red'
        
    fast_mem = set(v for v in topo_order if state[v] == 'blue')
    ops = []
    
    # Pre-compute total uses
    total_uses = defaultdict(int)
    for v in topo_order:
        for c in children[v]:
            total_uses[v] += 1
            
    # Remaining uses
    remaining_uses = defaultdict(int)
    for v in topo_order:
        remaining_uses[v] = total_uses[v]
        
    # Priority queue for eviction: (remaining_uses, vertex)
    # We want to evict the one with lowest remaining uses.
    # If tie, evict the one with highest depth? Or just arbitrary.
    
    def evict_if_needed(exclude=None):
        nonlocal fast_mem, state, ops, remaining_uses
        while len(fast_mem) > budget:
            if time.time() > deadline:
                return
            # Find victim
            best_victim = None
            min_uses = float('inf')
            for u in fast_mem:
                if u == exclude:
                    continue
                if u in output_set:
                    continue
                # If remaining uses is 0, it's a perfect candidate
                if remaining_uses[u] < min_uses:
                    min_uses = remaining_uses[u]
                    best_victim = u
                    
            if best_victim is None:
                # All are outputs or excluded
                for u in fast_mem:
                    if u != exclude:
                        best_victim = u
                        break
                if best_victim is None:
                    break
                    
            ops.append(['store', best_victim])
            state[best_victim] = 'red'
            fast_mem.discard(best_victim)
            
    for v in topo_order:
        if time.time() > deadline:
            break
            
        parents_list = parents.get(v, [])
        
        # Load parents
        for p in parents_list:
            if state[p] == 'red':
                ops.append(['load', p])
                state[p] = 'blue'
                fast_mem.add(p)
                
        # Evict if needed (exclude v)
        evict_if_needed(exclude=v)
        
        # Compute v
        if state[v] == 'red':
            ops.append(['compute', v])
            state[v] = 'blue'
            fast_mem.add(v)
            
        # Update remaining uses of parents
        for p in parents_list:
            remaining_uses[p] -= 1
            
        # Evict if needed
        evict_if_needed(exclude=None)
        
    # Ensure outputs are blue
    for v in output_set:
        if state[v] == 'red':
            ops.append(['load', v])
            state[v] = 'blue'
            fast_mem.add(v)
        evict_if_needed(exclude=None)
        
    return ops
# EVOLVE-BLOCK-END
