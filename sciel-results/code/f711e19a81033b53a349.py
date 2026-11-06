import time
import random

def synthesize(matrix, depth_limit):
    start = time.time()
    DEADLINE = 80.0  # safety margin under 90s
    
    m = len(matrix)
    n = len(matrix[0]) if m else 0
    rows = []
    for i in range(m):
        v = 0
        for j in range(n):
            if matrix[i][j]:
                v |= (1 << j)
        rows.append(v)
    
    # popcounts
    pop = [bin(v).count('1') for v in rows]
    
    def build_circuit(order, rng):
        # returns (gate_count, gates, outputs, max_depth)
        wire_of = {}
        depth = {}
        for j in range(n):
            wire_of[1 << j] = j
            depth[j] = 0
        wire_of[0] = n
        depth[n] = 0
        gates = []
        nextw = n + 1
        maxd = 0
        
        def add_gate(a, b):
            nonlocal nextw, maxd
            w = nextw
            nextw += 1
            gates.append([a, b])
            d = 1 + max(depth[a], depth[b])
            depth[w] = d
            if d > maxd:
                maxd = d
            return w
        
        def build_from_bits(v):
            # balanced tree over set bits
            bits = [j for j in range(n) if (v >> j) & 1]
            if not bits:
                wire_of[v] = n
                return n
            if len(bits) == 1:
                wire_of[v] = bits[0]
                return bits[0]
            # balanced: build leaves then combine pairwise
            wires = [wire_of[1 << j] for j in bits]
            while len(wires) > 1:
                neww = []
                for k in range(0, len(wires), 2):
                    if k + 1 < len(wires):
                        neww.append(add_gate(wires[k], wires[k+1]))
                    else:
                        neww.append(wires[k])
                wires = neww
            wire_of[v] = wires[0]
            return wires[0]
        
        # remaining rows for reuse heuristic
        remaining = set(range(m))
        
        def ensure(v):
            if v in wire_of:
                return wire_of[v]
            # find decompositions
            cands = []
            for av, aw in wire_of.items():
                if av == 0 or av == v:
                    continue
                bv = v ^ av
                if bv == 0 or bv == v:
                    continue
                if bv in wire_of:
                    bw = wire_of[bv]
                    cands.append((aw, bw, av, bv))
            if cands:
                # choose best: prefer low depth, then high reuse
                def score(c):
                    aw, bw, av, bv = c
                    d = max(depth[aw], depth[bw])
                    # reuse: count remaining rows r where (rows[r]^av) in wire_of or (rows[r]^bv) in wire_of
                    reuse = 0
                    for ri in remaining:
                        r = rows[ri]
                        if (r ^ av) in wire_of or (r ^ bv) in wire_of:
                            reuse += 1
                    return (d, -reuse)
                cands.sort(key=score)
                aw, bw, av, bv = cands[0]
                w = add_gate(aw, bw)
                wire_of[v] = w
                return w
            else:
                return build_from_bits(v)
        
        for i in order:
            ensure(rows[i])
            remaining.discard(i)
        
        outputs = [wire_of[rows[i]] for i in range(m)]
        return len(gates), gates, outputs, maxd
    
    # candidate orders
    orders = []
    # by popcount asc
    orders.append(sorted(range(m), key=lambda i: (pop[i], i)))
    # by popcount desc
    orders.append(sorted(range(m), key=lambda i: (-pop[i], i)))
    # natural
    orders.append(list(range(m)))
    # random
    rng = random.Random(12345)
    for _ in range(20):
        o = list(range(m))
        rng.shuffle(o)
        orders.append(o)
    
    best = None
    for o in orders:
        if time.time() > DEADLINE:
            break
        gc, gates, outputs, md = build_circuit(o, rng)
        if md > depth_limit:
            # depth violation; try to fix by not using deep chains — rebuild is complex; skip or penalize
            # For safety, we can still accept if we rebuild violating wires; but let's just skip this order
            continue
        if best is None or gc < best[0]:
            best = (gc, gates, outputs, md)
    
    if best is None:
        # fallback: natural order, accept any depth (shouldn't happen)
        gc, gates, outputs, md = build_circuit(list(range(m)), rng)
        best = (gc, gates, outputs, md)
    
    return {'gates': best[1], 'outputs': best[2]}
