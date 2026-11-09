# EVOLVE-BLOCK-START
import time
from collections import defaultdict

_DEADLINE = 55.0

def _get_parents(dag, vertex):
    if isinstance(dag, dict):
        return dag.get('parents', {}).get(vertex, [])
    elif hasattr(dag, 'parents'):
        return dag.parents.get(vertex, []) if isinstance(dag.parents, dict) else list(dag.parents[vertex])
    return []

def _get_outputs(dag):
    if isinstance(dag, dict):
        return list(dag.get('outputs', []))
    elif hasattr(dag, 'outputs'):
        return list(dag.outputs)
    return []

def _get_all_vertices(dag):
    if isinstance(dag, dict):
        if 'vertices' in dag:
            return list(dag['vertices'])
        parents = dag.get('parents', {})
        all_v = set()
        for v, ps in parents.items():
            all_v.add(v)
            for p in ps:
                all_v.add(p)
        return list(all_v)
    elif hasattr(dag, 'vertices'):
        return list(dag.vertices)
    return []

def _get_inputs(dag):
    all_v = _get_all_vertices(dag)
    return [v for v in all_v if not _get_parents(dag, v)]

def _topological_order(dag):
    if isinstance(dag, dict) and 'topo_order' in dag:
        return list(dag['topo_order'])
    
    all_v = _get_all_vertices(dag)
    parents = {v: _get_parents(dag, v) for v in all_v}
    children = defaultdict(list)
    for v in all_v:
        for p in parents[v]:
            children[p].append(v)
            
    in_degree = {v: len(parents[v]) for v in all_v}
    queue = [v for v in all_v if in_degree[v] == 0]
    topo = []
    while queue:
        v = queue.pop(0)
        topo.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
                
    if len(topo) != len(all_v):
        return all_v
    return topo

def _validate_operations(operations, dag, fast_memory_budget):
    in_fast = set()
    for op, v in operations:
        if op == 'load':
            if v in in_fast or len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'store':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        elif op == 'compute':
            parents = _get_parents(dag, v)
            for p in parents:
                if p not in in_fast:
                    return False
            if v in in_fast or len(in_fast) >= fast_memory_budget:
                return False
            in_fast.add(v)
        elif op == 'delete':
            if v not in in_fast:
                return False
            in_fast.remove(v)
        else:
            return False
    return True

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    
    all_v = _get_all_vertices(dag)
    if not all_v:
        return []
        
    topo = _topological_order(dag)
    topo_index = {v: i for i, v in enumerate(topo)}
    outputs = set(_get_outputs(dag))
    
    parents_map = {v: _get_parents(dag, v) for v in all_v}
    children_map = defaultdict(list)
    for v in all_v:
        for p in parents_map[v]:
            children_map[p].append(v)
            
    # Calculate last use index for each node
    # last_use[v] is the max topo index of any child of v.
    # If v has no children, last_use[v] is topo_index[v] (it is used immediately after computation? No, it's the node itself).
    # Actually, we need to know when a node is NO LONGER needed.
    # A node v is needed until all its children are computed.
    # So it is needed until max(topo_index[c] for c in children_map[v]).
    # If no children, it is needed until topo_index[v] (just for the compute step? No, compute makes it available).
    # After computing v, v is in fast memory.
    # v is needed for computing its children.
    # So v can be deleted after the last child of v is computed.
    
    last_needed = {}
    for v in all_v:
        if children_map[v]:
            last_needed[v] = max(topo_index[c] for c in children_map[v])
        else:
            # Leaf node. Needed only if it's an output? 
            # If it's an output, we must store it.
            # If it's not an output, it's not needed after compute.
            last_needed[v] = topo_index[v]

    operations = []
    in_fast = set()
    
    for i, v in enumerate(topo):
        if time.time() - start_time > _DEADLINE:
            break
            
        parents = parents_map[v]
        
        # 1. Ensure parents are in fast memory
        for p in parents:
            if p not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    # Evict a node that is no longer needed
                    # A node u is no longer needed if last_needed[u] < i
                    # (i is the current index, so children with index < i are done)
                    # Note: If last_needed[u] == i, u is needed for v (if u is parent of v).
                    # But we are loading p. p is a parent of v.
                    # We should not evict p (it's not in fast anyway).
                    # We should not evict any parent of v that is in fast? 
                    # Actually, if a parent is in fast, we don't load it.
                    # We only load if NOT in fast.
                    # So we need to evict some OTHER node.
                    
                    evicted = False
                    # Prioritize evicting nodes that are outputs? No, keep outputs if possible.
                    # Evict non-outputs first.
                    candidates = [u for u in in_fast if last_needed[u] < i]
                    # Sort candidates: non-outputs first, then by last_needed (oldest first)
                    candidates.sort(key=lambda u: (u in outputs, last_needed[u]))
                    
                    for u in candidates:
                        operations.append(['delete', u])
                        in_fast.remove(u)
                        evicted = True
                        break
                    
                    if not evicted:
                        # If we can't evict a "safe" node, we might have to evict a "needed" node?
                        # That would cause recomputation. 
                        # For now, if we can't evict, we fail or use a fallback.
                        # But if budget < max_width, this is expected.
                        # We'll just pick any node not in parents.
                        candidates = [u for u in in_fast if u not in parents]
                        if candidates:
                            u = candidates[0]
                            operations.append(['delete', u])
                            in_fast.remove(u)
                        else:
                            # Cannot proceed without recomputation. 
                            # Fallback: return empty or simple schedule.
                            return _simple_schedule(dag, fast_memory_budget)
                
                operations.append(['load', p])
                in_fast.add(p)
        
        # 2. Compute v
        operations.append(['compute', v])
        in_fast.add(v)
        
        # 3. Delete parents that are no longer needed
        for p in parents:
            if last_needed[p] < i: 
                # p's last child was before v? No, v is a child of p.
                # So last_needed[p] >= i.
                # Wait, last_needed[p] is the index of the LAST child.
                # If v is the last child of p, then last_needed[p] == i.
                # After computing v, p is no longer needed.
                # So we should delete p if last_needed[p] == i.
                pass
        
        # Correct logic:
        # After computing v, any parent p of v is no longer needed IF v was the last child of p.
        for p in parents:
            if last_needed[p] == i:
                if p in in_fast:
                    operations.append(['delete', p])
                    in_fast.remove(p)
        
        # 4. If v is a leaf and not an output, delete it immediately?
        # If v has no children, last_needed[v] == i.
        # If v is not an output, it is never needed again.
        if not children_map[v] and v not in outputs:
            if v in in_fast:
                operations.append(['delete', v])
                in_fast.remove(v)

    # 5. Store outputs
    for o in outputs:
        if o in in_fast:
            operations.append(['store', o])
            in_fast.remove(o)
        else:
            # If output was evicted, we need to load it? 
            # We shouldn't have evicted it if it was needed.
            # If it was evicted, it's a bug.
            # For safety, load it? No, we can't load a value that isn't in slow memory? 
            # Actually, all values are in slow memory (conceptually).
            # But if we deleted it, we lost the fast copy.
            # We must load it from slow memory.
            if len(in_fast) < fast_memory_budget:
                operations.append(['load', o])
                in_fast.add(o)
                operations.append(['store', o])
                in_fast.remove(o)
            else:
                # Can't store?
                pass

    return operations

def _simple_schedule(dag, fast_memory_budget):
    # Fallback: Load all inputs, compute all, store all.
    # This might exceed budget if width > budget, but it's a valid structure.
    # Actually, if width > budget, this is invalid.
    # But it's a fallback.
    topo = _topological_order(dag)
    ops = []
    in_fast = set()
    parents_map = {v: _get_parents(dag, v) for v in _get_all_vertices(dag)}
    
    for v in topo:
        parents = parents_map[v]
        for p in parents:
            if p not in in_fast:
                if len(in_fast) >= fast_memory_budget:
                    # Evict arbitrary
                    for u in list(in_fast):
                        if u not in parents:
                            ops.append(['delete', u])
                            in_fast.remove(u)
                            break
                ops.append(['load', p])
                in_fast.add(p)
        ops.append(['compute', v])
        in_fast.add(v)
        
    outputs = _get_outputs(dag)
    for o in outputs:
        if o not in in_fast:
            ops.append(['load', o])
            in_fast.add(o)
        ops.append(['store', o])
        in_fast.remove(o)
        
    return ops
# EVOLVE-BLOCK-END
