"""Communication-minimizing DAG pebbling with bounded fast memory."""
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0
    
    try:
        vertices = dag.get('vertices', [])
        parents = dag.get('parents', {})
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
        
        n = len(vertices)
        if n == 0:
            return []
        
        parent_map = {}
        for v in vertices:
            if isinstance(parents, dict):
                parent_map[v] = list(parents.get(v, []))
            else:
                parent_map[v] = []
        
        children = defaultdict(list)
        for v in vertices:
            for p in parent_map.get(v, []):
                children[p].append(v)
        
        in_degree = {v: len(parent_map.get(v, [])) for v in vertices}
        topo_order = []
        queue = [v for v in vertices if in_degree[v] == 0]
        while queue:
            v = queue.pop(0)
            topo_order.append(v)
            for c in children[v]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        if len(topo_order) != n:
            topo_order = list(vertices)
        
        state = {v: 'absent' for v in vertices}
        for v in inputs:
            state[v] = 'blue'
        
        fast_mem_count = 0
        ops = []
        
        last_use = {v: i for i, v in enumerate(topo_order)}
        for v in vertices:
            for c in children[v]:
                if c in last_use:
                    last_use[v] = max(last_use[v], last_use[c])
        
        for i, v in enumerate(topo_order):
            if time.time() - start_time > DEADLINE:
                break
            
            if v in inputs:
                continue
            
            ps = parent_map.get(v, [])
            
            for p in ps:
                if state.get(p) == 'red':
                    ops.append(['store', p])
                    state[p] = 'blue'
                    fast_mem_count -= 1
            
            ops.append(['compute', v])
            state[v] = 'red'
            fast_mem_count += 1
            
            while fast_mem_count > fast_memory_budget:
                best_u = None
                best_last = float('inf')
                for u in vertices:
                    if state.get(u) == 'red' and u not in outputs:
                        if last_use[u] < best_last:
                            best_last = last_use[u]
                            best_u = u
                if best_u is None:
                    for u in vertices:
                        if state.get(u) == 'red':
                            best_u = u
                            break
                if best_u is None:
                    break
                ops.append(['store', best_u])
                state[best_u] = 'blue'
                fast_mem_count -= 1
        
        for o in outputs:
            if state.get(o) == 'red':
                ops.append(['store', o])
                state[o] = 'blue'
                fast_mem_count -= 1
        
        if len(ops) > 20000:
            ops = ops[:20000]
        
        return ops
    
    except Exception:
        return []
