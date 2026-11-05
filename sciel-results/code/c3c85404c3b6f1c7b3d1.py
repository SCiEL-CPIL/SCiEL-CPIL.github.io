if len(relevant_inputs) <= fast_memory_budget:
    for x in relevant_inputs:
        fast.add(x)
        blue.discard(x)
        ops.append(['load', x])
    for o in outputs:
        if now() > deadline: break
        memo = set()
        compute_vertex_no_evict(o, memo)  # inputs all red, no eviction needed
        # store output
        fast.discard(o); blue.add(o); ops.append(['store', o])
        # delete intermediates
        for v in list(memo):
            if v not in input_set and v not in output_set and v in fast:
                fast.discard(v); ops.append(['delete', v])
        if o in remaining: remaining.remove(o)
        if len(ops) > 19000: break
