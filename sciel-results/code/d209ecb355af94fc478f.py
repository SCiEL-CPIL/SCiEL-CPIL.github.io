    def ensure_available(v):
        if v in fast_mem:
            return
        if v in inputs:
            ops.append(['load', v])
            fast_mem.add(v)
        else:
            # Re-compute v
            # First ensure its parents are available
            for p in parents.get(v, []):
                ensure_available(p)
            ops.append(['compute', v])
            fast_mem.add(v)
            # Manage budget after compute
            manage_budget()
    