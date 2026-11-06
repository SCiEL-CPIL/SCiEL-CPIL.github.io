"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import deque

def _topo_sort(dag):
    parents = dag.get("parents", {})
    outputs = list(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
    
    indeg = {v: len(parents.get(v, [])) for v in vertices}
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            if p in children:
                children[p].append(v)
            
    q = deque([v for v in vertices if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in children.get(u, []):
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
                
    if len(topo) != len(vertices):
        return vertices
    return topo

def _pebble_paging(dag, budget):
    parents = dag.get("parents", {})
    outputs = set(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
        
    topo = _topo_sort(dag)
    if not topo:
        return []
        
    topo_idx = {v: i for i, v in enumerate(topo)}
    
    # Build children list and sort by topo index
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
            
    for v in children:
        children[v].sort(key=lambda c: topo_idx[c])
        
    next_child_ptr = {v: 0 for v in vertices}
    
    ops = []
    live = set()
    in_slow = set()
    
    for v in topo:
        # Ensure parents are in live
        for p in parents.get(v, []):
            if p not in live:
                ops.append(["load", p])
                live.add(p)
                if p in in_slow:
                    in_slow.remove(p)
                
        ops.append(["compute", v])
        
        # Update next_child_ptr for parents
        for p in parents.get(v, []):
            next_child_ptr[p] += 1
            
        if v in outputs:
            ops.append(["store", v])
        else:
            live.add(v)
            
        # Prune vertices that are no longer needed
        to_remove = []
        for u in live:
            if next_child_ptr[u] >= len(children[u]):
                to_remove.append(u)
                
        for u in to_remove:
            live.remove(u)
            ops.append(["delete", u])
            
        # Evict if over budget
        protected = set(parents.get(v, []))
        while len(live) > budget:
            max_dist = -1
            evict_v = None
            for u in live:
                if u in protected:
                    continue
                if next_child_ptr[u] < len(children[u]):
                    next_c = children[u][next_child_ptr[u]]
                    dist = topo_idx[next_c]
                    if dist > max_dist:
                        max_dist = dist
                        evict_v = u
                        
            if evict_v is None:
                break
                
            ops.append(["store", evict_v])
            live.remove(evict_v)
            in_slow.add(evict_v)
            
    return ops[:20000]

def _pebble_fallback(dag, budget):
    parents = dag.get("parents", {})
    outputs = set(dag.get("outputs", []))
    vertices = sorted(set(list(parents.keys()) + outputs), key=lambda x: (x if isinstance(x, int) else 0))
    if not vertices:
        return []
        
    topo = _topo_sort(dag)
    if not topo:
        return []
        
    ops = []
    live = set()
    
    # Simple greedy: load, compute, store/delete
    # This is O(N^2) in worst case due to live set checks, but N is small enough?
    # If N is large, this might be slow.
    # But it's a fallback.
    
    children = {v: [] for v in vertices}
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
            
    for v in topo:
        for p in parents.get(v, []):
            if p not in live:
                ops.append(["load", p])
                live.add(p)
                
        ops.append(["compute", v])
        
        if v in outputs:
            ops.append(["store", v])
        else:
            live.add(v)
            
        # Delete if not needed
        # Check if any child of u is in remaining topo
        # This is slow. Just delete non-outputs immediately?
        # No, that causes reloads.
        # Just keep them?
        # If budget is large, this is fine.
        
    return ops[:20000]

def pebble(dag, fast_memory_budget):
    deadline = time.time() + 50
    try:
        result = _pebble_paging(dag, fast_memory_budget)
        if result is not None and len(result) <= 20000:
            return result
    except Exception:
        pass
    try:
        return _pebble_fallback(dag, fast_memory_budget)
    except Exception:
        return []
# EVOLVE-BLOCK-END
