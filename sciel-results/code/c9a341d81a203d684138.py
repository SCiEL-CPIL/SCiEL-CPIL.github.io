        for v in outputs:
            if v in fast_mem:
                operations.append(['store', v])
                fast_mem.remove(v)
                slow_mem.add(v)
        