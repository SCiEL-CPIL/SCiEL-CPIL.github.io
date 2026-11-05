def synthesize(matrix, depth_limit):
    import numpy as np
    import time
    from itertools import combinations
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    A = np.array(matrix, dtype=np.int8) % 2
    n, m = A.shape
    
    if not A.any():
        return {'gates': [], 'outputs': [n] * m}
    
    output_sets = []
    for j in range(m):
        col = A[:, j]
        nz = frozenset(i for i in range(n) if col[i] == 1)
        output_sets.append(nz)
    
    wire_for_set = {}
    for i in range(n):
        wire_for_set[frozenset([i])] = i
    wire_for_set[frozenset()] = n
    
    gate_list = []
    next_wire = [n + 1]
    
    def get_wire_for_set(s):
        if s in wire_for_set:
            return wire_for_set[s]
        
        if len(s) <= 6:
            elems = list(s)
            for size in range(1, len(s)):
                for subset in combinations(elems, size):
                    sub1 = frozenset(subset)
                    sub2 = frozenset(s - sub1)
                    if sub1 in wire_for_set and sub2 in wire_for_set:
                        w1 = wire_for_set[sub1]
                        w2 = wire_for_set[sub2]
                        gate_list.append((w1, w2))
                        wire = next_wire[0]
                        next_wire[0] += 1
                        wire_for_set[s] = wire
                        return wire
        
        elems = list(s)
        current = elems[0]
        for k in range(1, len(elems)):
            gate_list.append((current, elems[k]))
            current = next_wire[0]
            next_wire[0] += 1
        wire_for_set[s] = current
        return current
    
    output_wires = []
    for j in range(m):
        if time.time() > deadline:
            s = output_sets[j]
            if not s:
                output_wires.append(n)
            elif len(s) == 1:
                output_wires.append(list(s)[0])
            else:
                elems = list(s)
                current = elems[0]
                for k in range(1, len(elems)):
                    gate_list.append((current, elems[k]))
                    current = next_wire[0]
                    next_wire[0] += 1
                output_wires.append(current)
        else:
            s = output_sets[j]
            if not s:
                output_wires.append(n)
            elif len(s) == 1:
                output_wires.append(list(s)[0])
            else:
                output_wires.append(get_wire_for_set(s))
    
    return {'gates': gate_list, 'outputs': output_wires}
