"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 75.0
    
    # Parse DAG
    try:
        if isinstance(dag, dict):
            vertices = list(dag.get('vertices', []))
            parents = {v: list(ps) for v, ps in dag.get('parents', {}).items()}
            inputs = list(dag.get('inputs', []))
            outputs = list(dag.get('outputs', []))
        else:
            vertices = list(getattr(dag, 'vertices', []))
            parents = {v: list(ps) for v, ps in getattr(dag, 'parents', {}).items()}
            inputs = list(getattr(dag, 'inputs', []))
            outputs = list(getattr(dag, 'outputs', []))
    except Exception:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
        return pebble_baseline(dag, fast_memory_budget)
    
    if not vertices:
        return []
    
    # Build children map
    children = {v: [] for v in vertices}
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
    
    input_set = set(inputs) if inputs else set(v for v in vertices if not parents.get(v))
    output_set = set(outputs) if outputs else set(v for v in vertices if not children.get(v))
    
    n = len(vertices)
    
    # If budget is large enough, simple strategy
    if fast_memory_budget >= n:
        ops = []
        for v in vertices:
            if v in input_set:
                ops.append(['load', v])
            elif parents.get(v):
                ops.append(['compute', v])
        for v in vertices:
            if v in output_set:
                ops.append(['store', v])
        return ops
    
    # General pebbling
    fast = set()
    slow = set(input_set)
    ops = []
    op_count = 0
    
    def check():
        return (time.time() - start_time) < DEADLINE and op_count < 20000
    
    def load(v):
        nonlocal op_count
        if not check():
            return False
        ops.append(['load', v]); fast.add(v); slow.discard(v); op_count += 1
        return True
    
    def store(v):
        nonlocal op_count
        if not check():
            return False
        ops.append(['store', v]); slow.add(v); fast.discard(v); op_count += 1
        return True
    
    def compute(v):
        nonlocal op_count
        if not check():
            return False
        ops.append(['compute', v]); fast.add(v); op_count += 1
        return True
    
    def delete(v):
        nonlocal op_count
        if not check():
            return False
        ops.append(['delete', v]); fast.discard(v); op_count += 1
        return True
    
    remaining_children = {v: len(children.get(v, [])) for v in vertices}
    
    for v in vertices:
        if not check():
            break
        
        if v in input_set:
            if not load(v):
                break
        else:
            ok = True
            for p in parents.get(v, []):
                if p not in fast:
                    if p in slow:
                        if not load(p):
                            ok = False; break
                    else:
                        if not load(p):
                            ok = False; break
            if not ok:
                break
            if not compute(v):
                break
            for p in parents.get(v, []):
                remaining_children[p] -= 1
        
        # Cleanup: delete vertices no longer needed
        for x in list(fast):
            if x != v and x not in output_set and remaining_children.get(x, 0) == 0:
                if not delete(x):
                    break
        
        # If v is not output and has no remaining children, delete it
        if v not in output_set and remaining_children.get(v, 0) == 0 and v in fast:
            if not delete(v):
                break
        
        # If over budget, store some values
        if len(fast) > fast_memory_budget:
            # Store vertices that will be needed again
            for x in sorted(fast, key=lambda y: -remaining_children.get(y, 0)):
                if len(fast) <= fast_memory_budget:
                    break
                if x not in output_set and remaining_children.get(x, 0) > 0:
                    if not store(x):
                        break
    
    # Final: store all outputs
    for v in output_set:
        if v in fast and not check():
            break
        if v in fast:
            if not store(v):
                break
    
    return ops
# EVOLVE-BLOCK-END
