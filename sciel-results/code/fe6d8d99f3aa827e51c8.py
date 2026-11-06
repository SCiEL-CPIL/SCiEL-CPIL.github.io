"""Pebbling solver: minimize transfers via cache-aware topological scheduling with Belady-like eviction."""
import time
from collections import OrderedDict, deque

_DEADLINE_START = time.time()
_TIME_LIMIT = 80.0

def _elapsed():
    return time.time() - _DEADLINE_START

def _parse_dag(dag):
    parents_map = {}
    all_vertices = set()
    
    if isinstance(dag, dict):
        for v, info in dag.items():
            all_vertices.add(v)
            if isinstance(info, dict):
                par = info.get("parents", [])
                parents_map[v] = list(par)
                for p in par:
                    all_vertices.add(p)
            else:
                parents_map[v] = list(info)
                for p in info:
                    all_vertices.add(p)
        inputs = [v for v in all_vertices if not parents_map.get(v)]
        outputs = []
        for v in all_vertices:
            if isinstance(dag.get(v), dict) and dag[v].get("output"):
                outputs.append(v)
        if not outputs:
            children = set()
            for v, par in parents_map.items():
                for p in par:
                    children.add(p)
            outputs = [v for v in all_vertices if v not in children]
    elif isinstance(dag, (list, tuple)):
        for item in dag:
            if isinstance(item, dict):
                v = item.get("vertex", item.get("id"))
                par = item.get("parents", [])
                parents_map[v] = list(par)
                all_vertices.add(v)
                for p in par:
                    all_vertices.add(p)
            elif isinstance(item, (list, tuple)):
                v = item[0]
                par = item[1] if len(item) > 1 else []
                parents_map[v] = list(par)
                all_vertices.add(v)
                for p in par:
                    all_vertices.add(p)
        inputs = [v for v in all_vertices if not parents_map.get(v)]
        children = set()
        for v, par in parents_map.items():
            for p in par:
                children.add(p)
        outputs = [v for v in all_vertices if v not in children]
    else:
        raise ValueError("Unknown dag format")
    
    # Topological sort
    indeg = {v: len(parents_map.get(v, [])) for v in all_vertices}
    children_map = {v: [] for v in all_vertices}
    for v, par in parents_map.items():
        for p in par:
            children_map[p].append(v)
    
    q = deque([v for v in all_vertices if indeg[v] == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children_map[v]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    
    if len(topo) != len(all_vertices):
        remaining = [v for v in all_vertices if v not in set(topo)]
        topo.extend(remaining)
    
    return topo, parents_map, inputs, outputs

def pebble(dag, fast_memory_budget):
    try:
        topo, parents_map, inputs, outputs = _parse_dag(dag)
    except Exception:
        return []
    
    budget = max(1, int(fast_memory_budget))
    ops = []
    
    # Precompute last_use
    children_map = {v: [] for v in topo}
    for v in topo:
        for p in parents_map.get(v, []):
            if p in children_map:
                children_map[p].append(v)
    
    last_use = {v: i for i, v in enumerate(topo)}
    for i in range(len(topo) - 1, -1, -1):
        v = topo[i]
        for c in children_map[v]:
            if last_use.get(c, -1) > last_use[v]:
                last_use[v] = last_use[c]
    
    fast = OrderedDict()
    
    def load(v):
        nonlocal ops
        if v not in fast:
            ops.append(["load", v])
            fast[v] = last_use.get(v, 0)
        else:
            fast.move_to_end(v)
    
    def store(v):
        nonlocal ops
        if v in fast:
            del fast[v]
            ops.append(["store", v])
    
    def evict_one():
        nonlocal ops, fast
        if not fast:
            return
        min_v = min(fast.keys(), key=lambda x: fast[x])
        store(min_v)
    
    computed = set(inputs)
    
    for v in topo:
        if _elapsed() > _TIME_LIMIT or len(ops) > 20000:
            break
        if v in fast:
            fast.move_to_end(v)
            continue
        par = parents_map.get(v, [])
        for p in par:
            if p not in fast:
                while len(fast) >= budget:
                    evict_one()
                load(p)
        ops.append(["compute", v])
        fast[v] = last_use.get(v, 0)
        computed.add(v)
        fast.move_to_end(v)
    
    for v in outputs:
        if v not in computed and _elapsed() <= _TIME_LIMIT and len(ops) < 20000:
            par = parents_map.get(v, [])
            for p in par:
                if p not in fast:
                    while len(fast) >= budget:
                        evict_one()
                    load(p)
            ops.append(["compute", v])
            fast[v] = last_use.get(v, 0)
            computed.add(v)
    
    return ops
