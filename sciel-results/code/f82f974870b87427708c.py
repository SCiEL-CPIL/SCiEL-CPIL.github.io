def _fallback(vertices, parents_idx, inputs_mask, outputs_mask, budget, children, topo):
    n = len(vertices)
    fast = set()
    slow = set()
    ops = []
    processed = set()
    
    def is_input(i): return (inputs_mask >> i) & 1
    def is_output(i): return (outputs_mask >> i) & 1
    
    def make_room():
        # delete a vertex in fast that has no unprocessed children
        cands = [x for x in fast if not any(c not in processed for c in children[x])]
        non_out = [x for x in cands if not is_output(x)]
        pool = non_out if non_out else cands
        if not pool:
            pool = list(fast)
        if pool:
            d = pool[0]
            fast.discard(d)
            ops.append(['delete', vertices[d]])
        else:
            # can't make room; this shouldn't happen if budget >= 1
            pass
    
    def ensure_in_fast(i):
        # ensure i is in fast memory
        if i in fast:
            return
        if is_input(i) or i in slow:
            while len(fast) >= budget:
                make_room()
            fast.add(i)
            ops.append(['load', vertices[i]])
        else:
            # recompute i: ensure its parents in fast, then compute i
            for p in parents_idx[i]:
                ensure_in_fast(p)
            if i not in fast:
                while len(fast) >= budget:
                    make_room()
                fast.add(i)
                ops.append(['compute', vertices[i]])
    
    for v in topo:
        for p in parents_idx[v]:
            ensure_in_fast(p)
        if v not in fast:
            while len(fast) >= budget:
                make_room()
            fast.add(v)
            ops.append(['compute', vertices[v]])
        if is_output(v) and v not in slow:
            slow.add(v)
            ops.append(['store', vertices[v]])
        processed.add(v)
        # delete parents that are no longer needed
        for p in parents_idx[v]:
            if not is_output(p) and p in fast and not any(c not in processed for c in children[p]):
                fast.discard(p)
                ops.append(['delete', vertices[p]])
    
    return ops
