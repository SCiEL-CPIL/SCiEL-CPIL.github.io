"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict, deque

def pebble(dag, fast_memory_budget):
    """
    Communication-minimizing DAG pebbling.
    dag structure expected:
      - dag['parents']: dict vertex -> list of parent vertices (topologically ordered)
      - dag['outputs']: list of output vertices
      - dag['inputs']: list of input vertices (start blue)
    Returns list of [operation, vertex] where operation in {load, store, compute, delete}.
    """
    start_time = time.time()
    DEADLINE = 90.0  # wall-clock limit from evaluator
    SAFETY_MARGIN = 5.0  # seconds to spare

    # Try to import baseline as fallback
    baseline_ops = None
    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
    except Exception:
        pebble_baseline = None

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        output_set = set(outputs)
        
        # Build reverse adjacency (children)
        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)
        
        # Topological order (assuming parents are already in topo order or we compute it)
        # Compute in-degree for topological sort
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        # Inputs have no parents, so in_degree = 0
        for v in inputs:
            in_degree[v] = 0
        
        queue = [v for v in all_vertices if in_degree.get(v, 0) == 0]
        topo_order = []
        while queue:
            queue.sort()
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        # Map vertex to index in topo_order for O(1) lookup
        topo_index = {v: i for i, v in enumerate(topo_order)}
        
        # Precompute last_use[v]: the highest topo index of any vertex that needs v as a parent
        # If v is an output, it must be kept until the end
        last_use = {}
        for v in all_vertices:
            max_idx = -1
            for c in children.get(v, []):
                idx = topo_index.get(c, -1)
                if idx > max_idx:
                    max_idx = idx
            if v in output_set:
                # Outputs must be available at the end, so treat their last_use as the end
                max_idx = max(max_idx, len(topo_order) - 1)
            last_use[v] = max_idx
        
        # Strategy: Process vertices in topological order with memory-aware deletion
        # Key improvements:
        # 1. Only load if not already in fast memory
        # 2. Delete vertices as soon as their last use is passed to free memory
        # 3. Track which vertices are still needed to avoid unnecessary loads
        
        ops = []
        fast_mem = set()
        # Track remaining uses: for each vertex, how many more children need it
        remaining_uses = {}
        for v in all_vertices:
            if v in output_set:
                # Outputs must be kept until the very end
                remaining_uses[v] = len(topo_order)
            else:
                remaining_uses[v] = len(children.get(v, []))
        
        for v in topo_order:
            if time.time() > DEADLINE - SAFETY_MARGIN:
                break
            
            ps = parents.get(v, [])
            
            if not ps:
                # Input vertex - start blue, load into fast memory
                if v not in fast_mem:
                    ops.append(['load', v])
                    fast_mem.add(v)
            else:
                # Compute vertex - need all parents in fast memory
                for p in ps:
                    if p not in fast_mem:
                        # Load parent from slow memory
                        ops.append(['load', p])
                        fast_mem.add(p)
                
                # Compute v
                ops.append(['compute', v])
                fast_mem.add(v)
                
                # Decrement remaining uses for parents
                for p in ps:
                    remaining_uses[p] -= 1
                    # If this was the last use and p is not an output, delete it
                    if remaining_uses[p] == 0 and p not in output_set:
                        ops.append(['delete', p])
                        fast_mem.discard(p)
        
        # Ensure all outputs are in fast memory at the end
        for o in outputs:
            if o not in fast_mem:
                ops.append(['load', o])
                fast_mem.add(o)
        
        # Check operation count
        if len(ops) <= 20000:
            return ops
        else:
            # Too many ops, try baseline or truncate
            if pebble_baseline:
                return pebble_baseline(dag, fast_memory_budget)
            return ops[:20000]
            
    except Exception as e:
        # Fallback to baseline if available
        if pebble_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        return []
# EVOLVE-BLOCK-END
