N = 16

# EVOLVE-BLOCK-START
def construct():
    comparators = []
    
    def _merge(offset, size):
        if size <= 0:
            return
            
        def v2p(v):
            if v % 2 == 0:
                return offset + v // 2
            else:
                return offset + size + v // 2
        
        # Number of steps is log2(size) + 1
        log_size = size.bit_length() - 1
        num_steps = log_size + 1
        
        for step in range(1, num_steps + 1):
            if step % 2 == 1:
                # Compare (2i, 2i+1)
                for i in range(size):
                    u = 2 * i
                    v = 2 * i + 1
                    pu = v2p(u)
                    pv = v2p(v)
                    if pu < pv:
                        comparators.append((pu, pv))
                    else:
                        comparators.append((pv, pu))
            else:
                # Compare (2i+1, 2i+2)
                for i in range(size - 1):
                    u = 2 * i + 1
                    v = 2 * i + 2
                    pu = v2p(u)
                    pv = v2p(v)
                    if pu < pv:
                        comparators.append((pu, pv))
                    else:
                        comparators.append((pv, pu))

    def _sort(offset, size):
        if size <= 1:
            return
        half = size // 2
        _sort(offset, half)
        _sort(offset + half, half)
        _merge(offset, half)

    _sort(0, N)
    return comparators
# EVOLVE-BLOCK-END
