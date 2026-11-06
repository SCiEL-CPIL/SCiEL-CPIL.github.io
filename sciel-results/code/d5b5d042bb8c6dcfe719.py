"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 55.0
    
    parents = dag.get('parents', [])
    outputs = set(dag.get('outputs', []))
    n = len(parents)
    
    if n == 0:
        return []
    
    children = defaultdict(list)
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    is_input = [len(parents[v]) == 0 for v in range(n)]
    
    state = ['none'] * n
    for v in range(n):
        if is_input[v]:
            state[v] = 'blue'
    
    fast_mem = set(v for v in range(n) if state[v] == 'blue')
    ops = []
    
    next_use = [float('inf')] * n
    for v in range(n):
        if children[v]:
            next_use[v] = min(children[v])
    
    computed = [False] * n
    for v in range(n):
        if is_input[v]:
            computed[v] = True
    
    def ensure_available(v):
        nonlocal state, fast_mem, ops
        if state[v] == 'blue':
            return
        if state[v] == 'red':
            ops.append(['load', v])
            state[v] = 'blue'
            fast_mem.add(v)
            return
        # Recompute
        for p in parents[v]:
            ensure_available(p)
        ops.append(['compute', v])
        state[v] = 'blue'
        fast_mem.add(v)
        computed[v] = True
    
    def evict_if_needed(current_v):
        nonlocal state, fast_mem, ops
        while len(fast_mem) > fast_memory_budget:
            best_to_evict = None
            best_score = -1
            for u in fast_mem:
                if u == current_v or u in outputs:
                    continue
                nu = next_use[u] if next_use[u] < float('inf') else 2 * n
                if nu > best_score:
                    best_score = nu
                    best_to_evict = u
            if best_to_evict is None:
                break
            if best_score < current_v + 5:
                ops.append(['store', best_to_evict])
                state[best_to_evict] = 'red'
            else:
                ops.append(['delete', best_to_evict])
                state[best_to_evict] = 'none'
            fast_mem.remove(best_to_evict)
    
    for v in range(n):
        if time.time() - start_time > DEADLINE:
            break
        if is_input[v]:
            continue
        ensure_available(v)
        evict_if_needed(v)
    
    for o in outputs:
        if state[o] != 'blue':
            ensure_available(o)
    
    return ops

# EVOLVE-BLOCK-END
