<<<<<<< SEARCH
def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = t0 + 60.0 * _DEADLINE_FRAC
    try:
        n, layers, depth, parents, outputs, inputs_set, outputs_set = _topo_layers(dag)
        ops, ok = _layered_schedule(
            n, layers, depth, parents, outputs, inputs_set, outputs_set,
            fast_memory_budget
        )
        # Truncate to max ops if needed (should not happen for layered DAGs)
        if len(ops) > _MAX_OPS:
            ops = ops[:_MAX_OPS]
        return ops
    except Exception:
        # Fallback: naive sequential schedule
        return _naive_fallback(dag, fast_memory_budget)
=======
def pebble(dag, fast_memory_budget):
    t0 = time.time()
    deadline = t0 + 60.0 * _DEADLINE_FRAC
    try:
        n, layers, depth, parents, outputs, inputs_set, outputs_set, order = _topo_layers(dag)
        ops, ok = _layered_schedule(
            n, layers, depth, parents, outputs, inputs_set, outputs_set,
            fast_memory_budget, order
        )
        # Truncate to max ops if needed (should not happen for layered DAGs)
        if len(ops) > _MAX_OPS:
            ops = ops[:_MAX_OPS]
        return ops
    except Exception:
        # Fallback: naive sequential schedule
        return _naive_fallback(dag, fast_memory_budget)
>>>>>>> REPLACE

