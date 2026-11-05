"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    vertices = list(dag['vertices'])
    parents = dag['parents']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
    
    pos = {v: i for i, v in enumerate(vertices)}
    
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    red = set()
    computed = set()
    ops = []
    
    for i, v in enumerate(vertices):
        if v not in inputs:
            for p in parents.get(v, []):
                if p not in red:
                    if p not in computed:
                        raise ValueError(f"Parent {p} of {v} not computed")
                    ops.append(['load', p])
                    red.add(p)
            
            ops.append(['compute', v])
            red.add(v)
            computed.add(v)
            
            while len(red) > fast_memory_budget:
                best_delete = None
                best_score = float('inf')
                for r in red:
                    if r in outputs:
                        continue
                    count = sum(1 for c in children[r] if pos[c] > i)
                    if count == 0:
                        best_delete = r
                        break
                    if count < best_score:
                        best_score = count
                        best_delete = r
                
                if best_delete is None:
                    for r in red:
                        if r not in outputs:
                            best_delete = r
                            break
                    if best_delete is None:
                        break
                
                ops.append(['delete', best_delete])
                red.remove(best_delete)
    
    for o in outputs:
        if o in red:
            ops.append(['store', o])
            red.remove(o)
    
    return ops
# EVOLVE-BLOCK-END
