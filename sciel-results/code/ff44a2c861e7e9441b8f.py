import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80
    
    # Parse DAG
    if isinstance(dag, dict):
        vertices = list(dag.get('vertices', range(len(dag.get('edges', [])))))
        edges = list(dag.get('edges', []))
        inputs = set(dag.get('inputs', []))
        outputs = set(dag.get('outputs', []))
    elif hasattr(dag, 'vertices'):
        vertices = list(dag.vertices)
        edges = list(dag.edges)
        inputs = set(dag.inputs) if hasattr(dag, 'inputs') else set()
        outputs = set(dag.outputs) if hasattr(dag, 'outputs') else set()
    elif isinstance(dag, (list, tuple)):
        vertices = list(range(len(dag)))
        edges = []
        inputs = set()
        outputs = set()
        for i, item in enumerate(dag):
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                ps = list(item[0])
                for p in ps:
                    edges.append((p, i))
                if not ps:
                    inputs.add(i)
                if len(item) >= 2 and item[1]:
                    outputs.add(i)
            else:
                inputs.add(i)
    else:
        vertices = list(range(10))
        edges = []
        inputs = set()
        outputs = set()
    
    parents = {v: [] for v in vertices}
    children = {v: [] for v in vertices}
    for p, c in edges:
        parents[c].append(p)
        children[p].append(c)
    
    layer_of = {}
    for v in vertices:
        if not parents[v]:
            layer_of[v] = 0
        else:
            layer_of[v] = 1 + max(layer_of[p] for p in parents[v])
    
    max_layer = max(layer_of.values()) if layer_of else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v in vertices:
        layers[layer_of[v]].append(v)
    
    ops = []
    
    for i in range(1, max_layer + 1):
        if time.time() > deadline:
            break
        prev = layers[i-1]
        curr = layers[i]
        if not curr:
            continue
        if len(prev) <= fast_memory_budget:
            for v in prev:
                ops.append(('load', v))
            for v in curr:
                ops.append(('compute', v))
                ops.append(('store', v))
            for v in prev:
                ops.append(('delete', v))
        else:
            ps_map = defaultdict(list)
            for v in curr:
                ps = tuple(sorted(parents[v]))
                ps_map[ps].append(v)
            for ps, vs in ps_map.items():
                for p in ps:
                    ops.append(('load', p))
                for v in vs:
                    ops.append(('compute', v))
                    ops.append(('store', v))
                for p in ps:
                    ops.append(('delete', p))
    
    return ops
