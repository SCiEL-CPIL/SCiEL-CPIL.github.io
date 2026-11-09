# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

def _extract_dag_info(dag):
    parents = dag.get('parents', {})
    outputs = dag.get('outputs', [])
    
    all_vertices = set()
    for v, ps in parents.items():
        all_vertices.add(v)
        all_vertices.update(ps)
    for v in outputs:
        all_vertices.add(v)
    
    inputs = set()
    for v in all_vertices:
        if v not in parents or not parents[v]:
            inputs.add(v)
    
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    
    in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
    topo_order = []
    queue = [v for v in all_vertices if in_degree[v] == 0]
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    
    return {
        'parents': parents,
        'outputs': outputs,
        'inputs': inputs,
        'children': children,
        'all_vertices': all_vertices,
        'topo_order': topo_order
    }

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 85.0
    
    info = _extract_dag_info(dag)
    budget = fast_memory_budget
    outputs = set(info['outputs'])
    inputs = info['inputs']
    topo_order = info['topo_order']
    max_ops = 20000
    
    if budget >= len(info['all_vertices']):
        ops = []
        red = set()
        for v in topo_order:
            if v in inputs:
                ops.append(['load', v])
                red.add(v)
            else:
                ops.append(['compute', v])
                red.add(v)
        for v in outputs:
            if v in red:
                ops.append(['store', v])
                red.discard(v)
        if len(ops) <= max_ops:
            return ops
        return []
    
    initial_state = frozenset()
    
    def heuristic(state):
        return len(outputs - state)
    
    g_score = {initial_state: 0}
    f_score = {initial_state: heuristic(initial_state)}
    parent = {}
    
    open_set = [(f_score[initial_state], initial_state)]
    closed_set = set()
    
    while open_set:
        if time.time() > deadline:
            break
        
        _, current = heapq.heappop(open_set)
        
        if current in closed_set:
            continue
        closed_set.add(current)
        
        if outputs.issubset(current):
            path = []
            cur = current
            while cur != initial_state:
                op, v, prev = parent[cur]
                path.append([op, v])
                cur = prev
            path.reverse()
            if len(path) <= max_ops:
                return path
            return []
        
        red_set = set(current)
        
        for v in inputs:
            if v not in red_set and len(red_set) < budget:
                new_state = frozenset(red_set | {v})
                if new_state not in closed_set:
                    new_g = g_score[current] + 1
                    if new_g < g_score.get(new_state, float('inf')):
                        g_score[new_state] = new_g
                        f_score[new_state] = new_g + heuristic(new_state)
                        parent[new_state] = ('load', v, current)
                        heapq.heappush(open_set, (f_score[new_state], new_state))
        
        for v in red_set:
            if v not in inputs:
                new_state = frozenset(red_set - {v})
                if new_state not in closed_set:
                    new_g = g_score[current] + 1
                    if new_g < g_score.get(new_state, float('inf')):
                        g_score[new_state] = new_g
                        f_score[new_state] = new_g + heuristic(new_state)
                        parent[new_state] = ('store', v, current)
                        heapq.heappush(open_set, (f_score[new_state], new_state))
        
        for v in topo_order:
            if v not in red_set and len(red_set) < budget:
                parents_v = info['parents'].get(v, [])
                if all(p in red_set for p in parents_v):
                    new_state = frozenset(red_set | {v})
                    if new_state not in closed_set:
                        new_g = g_score[current] + 0
                        if new_g < g_score.get(new_state, float('inf')):
                            g_score[new_state] = new_g
                            f_score[new_state] = new_g + heuristic(new_state)
                            parent[new_state] = ('compute', v, current)
                            heapq.heappush(open_set, (f_score[new_state], new_state))
    
    return []
# EVOLVE-BLOCK-END
