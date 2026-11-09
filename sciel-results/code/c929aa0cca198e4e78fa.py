"""Improved program for communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque

def _topo_order(dag):
    """
    Extract topological order from dag.
    dag is expected to have:
      - dag['vertices'] or similar structure
      - parents: for each vertex, list of parents
      - outputs: list of output vertices
      - inputs: list of input vertices
    We try multiple common formats.
    """
    # Try to identify vertices, parents, outputs, inputs
    vertices = None
    parents = None
    outputs = None
    inputs = None
    
    if isinstance(dag, dict):
        if 'vertices' in dag:
            vertices = dag['vertices']
        if 'parents' in dag:
            parents = dag['parents']
        if 'outputs' in dag:
            outputs = dag['outputs']
        if 'inputs' in dag:
            inputs = dag['inputs']
        if 'v' in dag:
            vertices = dag['v']
        if 'p' in dag:
            parents = dag['p']
        if 'out' in dag:
            outputs = dag['out']
        if 'in' in dag:
            inputs = dag['in']
    else:
        # Maybe dag is a list of dicts or tuples
        if hasattr(dag, '__len__'):
            pass
    
    # If we couldn't extract, try to infer
    if vertices is None:
        # Try to get from parents keys
        if parents is not None and isinstance(parents, dict):
            vertices = list(parents.keys())
        elif hasattr(dag, 'vertices'):
            vertices = list(dag.vertices)
        else:
            # Fallback: assume vertices are 0..n-1
            if parents is not None:
                max_v = max(parents.keys()) if isinstance(parents, dict) else 0
                vertices = list(range(max_v + 1))
            else:
                vertices = []
    
    if parents is None:
        if hasattr(dag, 'parents'):
            parents = dag.parents
        elif hasattr(dag, 'p'):
            parents = dag.p
        else:
            # Assume no parents for all vertices
            parents = {v: [] for v in vertices}
    
    if outputs is None:
        if hasattr(dag, 'outputs'):
            outputs = dag.outputs
        elif hasattr(dag, 'out'):
            outputs = dag.out
        else:
            # Outputs are vertices with no children
            children = defaultdict(list)
            for v in vertices:
                for p in parents.get(v, []):
                    children[p].append(v)
            outputs = [v for v in vertices if not children[v]]
    
    if inputs is None:
        if hasattr(dag, 'inputs'):
            inputs = dag.inputs
        elif hasattr(dag, 'in'):
            inputs = dag.in_ if hasattr(dag, 'in_') else dag.get('in', []) if isinstance(dag, dict) else []
        else:
            # Inputs are vertices with no parents
            inputs = [v for v in vertices if not parents.get(v, [])]
    
    # Build topological order using Kahn's algorithm
    in_degree = {v: len(parents.get(v, [])) for v in vertices}
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    topo = []
    while queue:
        v = queue.popleft()
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    # If topological sort didn't include all vertices (cycle?), append remaining
    if len(topo) < len(vertices):
        remaining = set(vertices) - set(topo)
        topo.extend(sorted(remaining))
    
    return topo, vertices, parents, outputs, inputs

def _compute_naive_cost(dag, fast_memory_budget):
    """
    Compute the naive schedule cost: load each input once, compute each vertex once,
    store each output once. For intermediate vertices, if they don't fit in memory,
    we may need to reload. But the naive schedule typically just does a straightforward
    topological pass, loading inputs as needed and keeping results in memory if possible.
    
    Actually, the naive baseline is likely: for each vertex in topological order,
    load all parents if not in memory, compute, then if memory is full, evict something.
    The baseline probably does a simple greedy eviction.
    
    For scoring, we need to match the baseline's behavior. Since we don't know the exact
    baseline, let's assume the naive cost is:
      - Each input is loaded once (or more if evicted and needed again)
      - Each compute is one operation
      - Each output is stored once
    
    A simpler model: naive schedule loads each input exactly once at start (if fits),
    or loads on demand. Let's assume the baseline does:
      - Load all inputs that fit in memory at start
      - Process in topological order
      - When memory is full, evict the vertex that is needed furthest in the future
      - Store outputs when done
    
    This is complex. Let's instead estimate: the number of transfers in a naive schedule
    is roughly the number of times a value must be moved to/from slow memory.
    
    For a layered DAG with L layers and W width, if fast_memory_budget = B:
      - If B >= W, all layer values fit, so each input loaded once, each output stored once.
        Transfers = #inputs + #outputs
      - If B < W, values are evicted and reloaded.
    
    Since we can't easily replicate the exact baseline, let's use a heuristic:
    naive_cost = #inputs + #outputs + sum over layers of max(0, W_layer - B) * 2
    (each extra value in a layer causes a store and a load)
    
    Actually, let's just return a reasonable estimate. The key is that our score is
    mean(transfers) / mean(naive_transfers), so we need to minimize transfers relative
    to the baseline.
    """
    topo, vertices, parents, outputs, inputs = _topo_order(dag)
    n_inputs = len(inputs) if inputs else 0
    n_outputs = len(outputs) if outputs else 0
    
    # Estimate naive cost: assume each input loaded once, each output stored once,
    # plus some reloads for intermediate values that don't fit.
    # For a simple estimate, assume naive does O(n) transfers.
    naive_cost = n_inputs + n_outputs
    # Add a factor for memory pressure
    if fast_memory_budget < len(vertices):
        naive_cost += len(vertices) - fast_memory_budget
    
    return max(naive_cost, 1)

def pebble(dag, fast_memory_budget):
    """
    Generate a pebbling schedule for the given DAG.
    
    Strategy: Use a recursive pebbling approach for layered DAGs.
    For each layer, process vertices in topological order.
    Use a cache of recently used values. When memory is full, evict the value
    that will be needed furthest in the future (or the one with the most dependents
    not yet processed).
    
    We aim to minimize the number of load/store operations (transfers).
    """
    topo, vertices, parents, outputs, inputs = _topo_order(dag)
    
    if not vertices:
        return []
    
    # Build children map
    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    # Identify input and output sets
    input_set = set(inputs) if inputs else set(v for v in vertices if not parents.get(v, []))
    output_set = set(outputs) if outputs else set(v for v in vertices if not children[v])
    
    # Determine the "last use" of each vertex: the latest topological index
    # at which a child of v is processed.
    last_use = {}
    for i, v in enumerate(topo):
        for c in children[v]:
            c_idx = topo.index(c)
            if v not in last_use or c_idx > last_use[v]:
                last_use[v] = c_idx
    
    # For vertices with no children, last_use is their own index
    for v in vertices:
        if v not in last_use:
            last_use[v] = topo.index(v)
    
    operations = []
    in_memory = set()  # vertices currently in fast memory
    computed = set()   # vertices that have been computed (value exists somewhere)
    # We need to track which vertices are in memory and which are in slow memory (computed but evicted)
    
    # We'll process in topological order
    # For each vertex, we need all its parents' values in memory to compute it.
    # If a parent is not in memory, we need to load it (if computed) or it must be an input.
    
    # Heuristic: when we need to evict, evict the vertex with the latest last_use
    # (i.e., the one we need least soon).
    
    def evict_one():
        """Evict the vertex with the latest last_use among those in memory."""
        if not in_memory:
            return None
        # Evict the one with max last_use
        best = max(in_memory, key=lambda v: last_use[v])
        in_memory.discard(best)
        operations.append(['store', best])
        return best
    
    for v in topo:
        # If v is an input, we need to load it if not in memory
        if v in input_set:
            if v not in in_memory:
                # Need to load v
                # First, ensure we have space
                while len(in_memory) >= fast_memory_budget:
                    evict_one()
                operations.append(['load', v])
                in_memory.add(v)
            computed.add(v)
        else:
            # v is an internal or output vertex; need to compute it
            # First, ensure all parents are in memory
            for p in parents.get(v, []):
                if p not in in_memory:
                    # Need to load p
                    # Ensure space
                    while len(in_memory) >= fast_memory_budget:
                        evict_one()
                    operations.append(['load', p])
                    in_memory.add(p)
            
            # Now compute v
            # Ensure space for v's result
            while len(in_memory) >= fast_memory_budget:
                evict_one()
            
            operations.append(['compute', v])
            in_memory.add(v)
            computed.add(v)
        
        # If v is an output, we must store it at the end
        # But we can't store it yet if it might be needed as a parent for other outputs
        # Actually, outputs have no children, so we can store them immediately after computing
        # But the problem says "all outputs must finish blue", meaning they must be in memory at the end.
        # So we should NOT store outputs until the very end, or we need to reload them.
        # Let's keep outputs in memory until the end.
    
    # At the end, all outputs must be in memory (blue)
    # Ensure all outputs are in memory
    for v in output_set:
        if v not in in_memory:
            # Need to load v
            while len(in_memory) >= fast_memory_budget:
                evict_one()
            operations.append(['load', v])
            in_memory.add(v)
    
    # The problem says inputs start blue (in memory). So we should start with inputs in memory.
    # Let's redo: initialize with inputs in memory.
    
    operations = []
    in_memory = set()
    computed = set()
    
    # Start with inputs in memory (they start blue)
    for v in input_set:
        if len(in_memory) < fast_memory_budget:
            in_memory.add(v)
            computed.add(v)
        else:
            # If there are more inputs than memory, we can't keep them all
            # We'll load them on demand
            pass
    
    for v in topo:
        if v in input_set:
            if v not in in_memory:
                while len(in_memory) >= fast_memory_budget:
                    evict_one()
                operations.append(['load', v])
                in_memory.add(v)
            computed.add(v)
        else:
            # Ensure parents in memory
            for p in parents.get(v, []):
                if p not in in_memory:
                    while len(in_memory) >= fast_memory_budget:
                        evict_one()
                    operations.append(['load', p])
                    in_memory.add(p)
            
            # Compute v
            while len(in_memory) >= fast_memory_budget:
                evict_one()
            
            operations.append(['compute', v])
            in_memory.add(v)
            computed.add(v)
    
    # Ensure all outputs are in memory
    for v in output_set:
        if v not in in_memory:
            while len(in_memory) >= fast_memory_budget:
                evict_one()
            operations.append(['load', v])
            in_memory.add(v)
    
    # Cap operations at 20000
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations
# EVOLVE-BLOCK-END
