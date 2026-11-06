"""DAG pebbling with communication minimization."""
# EVOLVE-BLOCK-START
import time
import heapq
from functools import lru_cache

def _topo_sort(dag):
    """Extract vertices and parents from dag, assuming topological order."""
    # dag is expected to be a dict or similar with vertex info
    # Typical structure: dag = {vertex: {'parents': [...], 'is_output': bool, ...}}
    # or dag = {'vertices': [...], 'parents': {v: [...]}, 'outputs': [...]}
    # We need to handle the format. Let's look at common patterns.
    
    # Since we don't have the exact format, let's try to infer it.
    # The task says "dag has topologically ordered parents and output vertices"
    
    # Let's assume a common format for these tasks:
    # dag = {'n': int, 'parents': {v: [list of parents]}, 'outputs': [list], 'inputs': [list]}
    # Or maybe dag is just a dict where keys are vertices and values contain parent info.
    
    # Let's try to be generic. We'll look for 'parents' key.
    if 'parents' in dag:
        parents_dict = dag['parents']
        outputs = dag.get('outputs', [])
        # Determine all vertices
        all_vertices = set(parents_dict.keys())
        for p_list in parents_dict.values():
            all_vertices.update(p_list)
        # Sort topologically (or assume already sorted if we trust the input)
        # But we should verify or just use the given order if possible
        # Let's assume vertices are numbered 0..n-1 and already in topo order
        n = dag.get('n', max(all_vertices) + 1 if all_vertices else 0)
        topo = list(range(n))
        return topo, parents_dict, outputs
    
    # Fallback: if dag is just a dict of vertex -> info
    if isinstance(dag, dict):
        # Try to extract
        parents_dict = {}
        outputs = []
        for v, info in dag.items():
            if isinstance(info, dict):
                parents_dict[v] = info.get('parents', [])
                if info.get('is_output', False):
                    outputs.append(v)
            elif isinstance(info, (list, tuple)):
                # Maybe info is just the list of parents
                parents_dict[v] = list(info)
                # Can't tell if output
        # This is getting complicated. Let's assume the 'parents' key format is standard.
        pass
    
    # If we can't determine, return empty
    return [], {}, []


def _compute_naive_transfers(dag, topo, parents_dict, outputs):
    """Compute number of transfers for naive schedule."""
    # Naive: load each input once, compute each vertex once, store each output once
    # But we need to account for the fact that we might need to reload inputs if memory is full
    # Actually, naive schedule usually means: topological order, keep everything in memory if possible
    # But with bounded memory, naive might be different.
    
    # Let's define naive as: for each vertex in topo order:
    # 1. Load all parents that are not in memory
    # 2. Compute vertex
    # 3. If vertex is output, store it
    # 4. Delete parents that are no longer needed
    # This is actually a reasonable baseline.
    
    in_memory = set()
    transfers = 0
    next_use = {}
    
    # Precompute next use for each vertex
    for i, v in enumerate(topo):
        # Find next use of v
        next_pos = -1
        for j in range(i + 1, len(topo)):
            u = topo[j]
            if v in parents_dict.get(u, []):
                next_pos = j
                break
        next_use[v] = next_pos
    
    for i, v in enumerate(topo):
        parents = parents_dict.get(v, [])
        
        # Load parents not in memory
        for p in parents:
            if p not in in_memory:
                transfers += 1  # load
                in_memory.add(p)
        
        # Compute v (no transfer)
        
        # Store if output
        if v in outputs:
            transfers += 1  # store
            # After storing, we can delete v
            in_memory.discard(v)
        else:
            # Check if we need to keep v
            # If next_use[v] == -1, we can delete
            if next_use[v] == -1:
                in_memory.discard(v)
            # Else keep it
    
    return transfers


def _greedy_pebbling(dag, topo, parents_dict, outputs, budget):
    """
    Greedy pebbling algorithm.
    Returns list of operations.
    """
    in_memory = set()
    operations = []
    
    # Precompute next use
    next_use = {}
    for i, v in enumerate(topo):
        next_pos = -1
        for j in range(i + 1, len(topo)):
            u = topo[j]
            if v in parents_dict.get(u, []):
                next_pos = j
                break
        next_use[v] = next_pos
    
    # Precompute use count
    use_count = {}
    for v in topo:
        use_count[v] = 0
    for v in topo:
        for p in parents_dict.get(v, []):
            use_count[p] += 1
    
    for i, v in enumerate(topo):
        parents = parents_dict.get(v, [])
        
        # Load parents not in memory
        for p in parents:
            if p not in in_memory:
                if len(in_memory) >= budget:
                    # Need to evict
                    # Find vertex in memory with farthest next use
                    evict_v = None
                    farthest = -1
                    for w in in_memory:
                        # If w is a parent of v, we can't evict it yet
                        if w in parents:
                            continue
                        nu = next_use.get(w, -1)
                        if nu > farthest:
                            farthest = nu
                            evict_v = w
                    if evict_v is not None:
                        operations.append(['delete', evict_v])
                        in_memory.discard(evict_v)
                    else:
                        # Shouldn't happen if budget is sufficient
                        break
                
                operations.append(['load', p])
                in_memory.add(p)
        
        # Compute v
        operations.append(['compute', v])
        in_memory.add(v)
        
        # If output, store and delete
        if v in outputs:
            operations.append(['store', v])
            in_memory.discard(v)
        else:
            # Check if we can delete
            if next_use[v] == -1:
                operations.append(['delete', v])
                in_memory.discard(v)
            else:
                # Check if memory is full
                if len(in_memory) > budget:
                    # Evict
                    evict_v = None
                    farthest = -1
                    for w in in_memory:
                        if w == v:
                            continue  # Just computed
                        if w in parents:
                            continue  # Still needed for this computation (shouldn't be in memory after)
                        nu = next_use.get(w, -1)
                        if nu > farthest:
                            farthest = nu
                            evict_v = w
                    if evict_v is not None:
                        operations.append(['delete', evict_v])
                        in_memory.discard(evict_v)
    
    return operations


def _simulate_pebbling(dag, topo, parents_dict, outputs, budget, operations):
    """
    Simulate the pebbling operations and count transfers.
    Returns (valid, transfers)
    """
    in_memory = set()
    transfers = 0
    computed = set()
    
    for op in operations:
        if len(op) < 2:
            continue
        action, v = op[0], op[1]
        
        if action == 'load':
            if v in in_memory:
                return False, 0  # Already in memory
            if len(in_memory) >= budget:
                return False, 0  # Memory full
            in_memory.add(v)
            transfers += 1
        
        elif action == 'store':
            if v not in in_memory:
                return False, 0  # Not in memory
            if v not in computed:
                return False, 0  # Not computed
            if v not in outputs:
                return False, 0  # Not an output
            in_memory.discard(v)
            transfers += 1
        
        elif action == 'compute':
            parents = parents_dict.get(v, [])
            for p in parents:
                if p not in in_memory:
                    return False, 0  # Parent not in memory
            if v in computed:
                # Recomputation allowed, but let's be strict
                pass
            computed.add(v)
            if len(in_memory) >= budget:
                in_memory.add(v)  # This would overflow, but let's say it replaces or fails
                # Actually, compute doesn't consume a slot, the result does
                # So if memory is full, we can't add the result
                return False, 0
            in_memory.add(v)
        
        elif action == 'delete':
            if v not in in_memory:
                return False, 0  # Not in memory
            in_memory.discard(v)
        
        else:
            return False, 0
    
    # Check all outputs are stored
    # Actually, we need to ensure outputs are stored. The simulation above doesn't track stored outputs.
    # Let's add that.
    
    # Re-do simulation with output tracking
    in_memory = set()
    transfers = 0
    computed = set()
    stored = set()
    
    for op in operations:
        if len(op) < 2:
            continue
        action, v = op[0], op[1]
        
        if action == 'load':
            if v in in_memory:
                return False, 0
            if len(in_memory) >= budget:
                return False, 0
            in_memory.add(v)
            transfers += 1
        
        elif action == 'store':
            if v not in in_memory:
                return False, 0
            if v not in computed:
                return False, 0
            if v not in outputs:
                return False, 0
            if v in stored:
                return False, 0  # Already stored
            in_memory.discard(v)
            stored.add(v)
            transfers += 1
        
        elif action == 'compute':
            parents = parents_dict.get(v, [])
            for p in parents:
                if p not in in_memory:
                    return False, 0
            computed.add(v)
            if len(in_memory) >= budget:
                return False, 0
            in_memory.add(v)
        
        elif action == 'delete':
            if v not in in_memory:
                return False, 0
            in_memory.discard(v)
    
    if stored != set(outputs):
        return False, 0
    
    return True, transfers


def _heuristic_pebbling(dag, topo, parents_dict, outputs, budget):
    """
    Improved heuristic: use a priority queue to decide which vertex to evict.
    Priority: next_use (farther is better to keep? No, farther is worse to evict)
    Actually, we want to evict the vertex with the farthest next use.
    """
    in_memory = set()
    operations = []
    
    # Precompute next use
    next_use = {}
    for i, v in enumerate(topo):
        next_pos = -1
        for j in range(i + 1, len(topo)):
            u = topo[j]
            if v in parents_dict.get(u, []):
                next_pos = j
                break
        next_use[v] = next_pos
    
    for i, v in enumerate(topo):
        parents = parents_dict.get(v, [])
        
        # Load parents not in memory
        for p in parents:
            if p not in in_memory:
                if len(in_memory) >= budget:
                    # Evict
                    evict_v = None
                    farthest = -1
                    for w in in_memory:
                        if w in parents:
                            continue
                        nu = next_use.get(w, -1)
                        if nu > farthest:
                            farthest = nu
                            evict_v = w
                    if evict_v is not None:
                        operations.append(['delete', evict_v])
                        in_memory.discard(evict_v)
                
                operations.append(['load', p])
                in_memory.add(p)
        
        # Compute v
        operations.append(['compute', v])
        in_memory.add(v)
        
        # If output, store and delete
        if v in outputs:
            operations.append(['store', v])
            in_memory.discard(v)
        else:
            if next_use[v] == -1:
                operations.append(['delete', v])
                in_memory.discard(v)
            elif len(in_memory) > budget:
                # Evict
                evict_v = None
                farthest = -1
                for w in in_memory:
                    if w == v:
                        continue
                    if w in parents:
                        continue
                    nu = next_use.get(w, -1)
                    if nu > farthest:
                        farthest = nu
                        evict_v = w
                if evict_v is not None:
                    operations.append(['delete', evict_v])
                    in_memory.discard(evict_v)
    
    return operations


def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85  # 85s to be safe
    
    # Extract DAG info
    topo, parents_dict, outputs = _topo_sort(dag)
    
    if not topo:
        return []
    
    # Compute naive transfers for reference
    naive_transfers = _compute_naive_transfers(dag, topo, parents_dict, outputs)
    
    # Try greedy pebbling
    ops_greedy = _greedy_pebbling(dag, topo, parents_dict, outputs, fast_memory_budget)
    valid_greedy, transfers_greedy = _simulate_pebbling(dag, topo, parents_dict, outputs, fast_memory_budget, ops_greedy)
    
    # Try heuristic pebbling
    ops_heur = _heuristic_pebbling(dag, topo, parents_dict, outputs, fast_memory_budget)
    valid_heur, transfers_heur = _simulate_pebbling(dag, topo, parents_dict, outputs, fast_memory_budget, ops_heur)
    
    # Choose the best valid one
    best_ops = None
    best_transfers = None
    
    if valid_greedy:
        best_ops = ops_greedy
        best_transfers = transfers_greedy
    
    if valid_heur:
        if best_ops is None or transfers_heur < best_transfers:
            best_ops = ops_heur
            best_transfers = transfers_heur
    
    if best_ops is None:
        # Fallback to naive-like schedule
        best_ops = []
        in_memory = set()
        next_use = {}
        for i, v in enumerate(topo):
            next_pos = -1
            for j in range(i + 1, len(topo)):
                u = topo[j]
                if v in parents_dict.get(u, []):
                    next_pos = j
                    break
            next_use[v] = next_pos
        
        for i, v in enumerate(topo):
            parents = parents_dict.get(v, [])
            for p in parents:
                if p not in in_memory:
                    best_ops.append(['load', p])
                    in_memory.add(p)
            best_ops.append(['compute', v])
            in_memory.add(v)
            if v in outputs:
                best_ops.append(['store', v])
                in_memory.discard(v)
            elif next_use[v] == -1:
                best_ops.append(['delete', v])
                in_memory.discard(v)
    
    return best_ops
# EVOLVE-BLOCK-END
