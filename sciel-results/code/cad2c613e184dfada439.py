"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _parse_dag(dag):
    """Parse DAG structure."""
    if isinstance(dag, dict):
        parents = dag.get('parents', [])
        outputs = dag.get('outputs', [])
        n = dag.get('n', len(parents))
    else:
        parents = list(dag)
        n = len(parents)
        children = [set() for _ in range(n)]
        for i, par in enumerate(parents):
            for p in par:
                children[p].add(i)
        outputs = [i for i in range(n) if not children[i]]
    return parents, outputs, n

def _simulate(ops, parents, outputs, n, budget):
    """Simulate and count transfers. Returns (transfers, valid)."""
    fast = set()  # red: in fast memory
    slow = set()  # blue: in slow memory
    is_input = [len(p) == 0 for p in parents]
    
    for v in range(n):
        if is_input[v]:
            slow.add(v)
    
    transfers = 0
    for op, v in ops:
        if op == 'load':
            if v not in slow or v in fast:
                return (10**9, False)
            if len(fast) >= budget:
                return (10**9, False)
            slow.remove(v)
            fast.add(v)
            transfers += 1
        elif op == 'store':
            if v not in fast or v in slow:
                return (10**9, False)
            fast.remove(v)
            slow.add(v)
            transfers += 1
        elif op == 'compute':
            if v in fast or v in slow:
                return (10**9, False)
            for p in parents[v]:
                if p not in fast:
                    return (10**9, False)
            if len(fast) >= budget:
                return (10**9, False)
            fast.add(v)
        elif op == 'delete':
            if v not in fast:
                return (10**9, False)
            fast.remove(v)
        else:
            return (10**9, False)
    
    for v in outputs:
        if v not in slow:
            return (10**9, False)
    
    return (transfers, True)

def _naive_schedule(parents, outputs, n):
    """Naive: load inputs, compute in order, store outputs."""
    ops = []
    is_input = [len(p) == 0 for p in parents]
    for v in range(n):
        if is_input[v]:
            ops.append(['load', v])
        else:
            ops.append(['compute', v])
    for v in outputs:
        ops.append(['store', v])
    return ops

def _optimal_pebble(parents, outputs, n, budget):
    """Optimized pebbling using reference counting and smart deletion."""
    ops = []
    fast = set()
    slow = set()
    is_input = [len(p) == 0 for p in parents]
    
    # Compute how many times each vertex is used as a parent
    usage_count = [0] * n
    for v in range(n):
        for p in parents[v]:
            usage_count[p] += 1
    
    # Track remaining uses
    remaining = list(usage_count)
    
    for v in range(n):
        if is_input[v]:
            if len(fast) < budget:
                ops.append(['load', v])
                fast.add(v)
            else:
                # Find a vertex to delete (one that's no longer needed)
                deleted = False
                for u in sorted(fast, key=lambda x: (remaining[x], x)):
                    if remaining[u] == 0 and u not in outputs:
                        ops.append(['delete', u])
                        fast.remove(u)
                        deleted = True
                        break
                if not deleted:
                    # Delete any non-output
                    for u in sorted(fast, key=lambda x: (remaining[x], x)):
                        if u not in outputs:
                            ops.append(['delete', u])
                            fast.remove(u)
                            deleted = True
                            break
                if not deleted:
                    return None
                ops.append(['load', v])
                fast.add(v)
        else:
            # Ensure parents are in fast
            for p in parents[v]:
                if p not in fast:
                    if p in slow:
                        if len(fast) < budget:
                            ops.append(['load', p])
                            fast.add(p)
                        else:
                            deleted = False
                            for u in sorted(fast, key=lambda x: (remaining[x], x)):
                                if remaining[u] == 0 and u not in outputs:
                                    ops.append(['delete', u])
                                    fast.remove(u)
                                    deleted = True
                                    break
                            if not deleted:
                                for u in sorted(fast, key=lambda x: (remaining[x], x)):
                                    if u not in outputs:
                                        ops.append(['delete', u])
                                        fast.remove(u)
                                        deleted = True
                                        break
                            if not deleted:
                                return None
                            ops.append(['load', p])
                            fast.add(p)
            if len(fast) < budget:
                ops.append(['compute', v])
                fast.add(v)
            else:
                deleted = False
                for u in sorted(fast, key=lambda x: (remaining[x], x)):
                    if remaining[u] == 0 and u not in outputs:
                        ops.append(['delete', u])
                        fast.remove(u)
                        deleted = True
                        break
                if not deleted:
                    for u in sorted(fast, key=lambda x: (remaining[x], x)):
                        if u not in outputs:
                            ops.append(['delete', u])
                            fast.remove(u)
                            deleted = True
                            break
                if not deleted:
                    return None
                ops.append(['compute', v])
                fast.add(v)
        
        # Decrement remaining for parents
        for p in parents[v]:
            remaining[p] -= 1
    
    # Store outputs
    for v in outputs:
        if v in fast:
            ops.append(['store', v])
            fast.remove(v)
            slow.add(v)
    
    t, valid = _simulate(ops, parents, outputs, n, budget)
    if not valid:
        return None
    return ops

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 55
    
    parents, outputs, n = _parse_dag(dag)
    if n == 0:
        return []
    
    # Naive baseline
    naive_ops = _naive_schedule(parents, outputs, n)
    naive_t, naive_valid = _simulate(naive_ops, parents, outputs, n, fast_memory_budget)
    
    if not naive_valid:
        result = _optimal_pebble(parents, outputs, n, fast_memory_budget)
        if result is not None:
            return result
        return naive_ops
    
    best_ops = naive_ops
    best_t = naive_t
    
    if time.time() < deadline:
        opt = _optimal_pebble(parents, outputs, n, fast_memory_budget)
        if opt is not None:
            t, valid = _simulate(opt, parents, outputs, n, fast_memory_budget)
            if valid and t < best_t:
                best_ops = opt
                best_t = t
    
    final_t, final_valid = _simulate(best_ops, parents, outputs, n, fast_memory_budget)
    if not final_valid or len(best_ops) > 20000:
        return _naive_schedule(parents, outputs, n)
    
    return best_ops
# EVOLVE-BLOCK-END
