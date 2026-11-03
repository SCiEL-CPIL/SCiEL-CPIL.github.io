import time
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    q = k

    def f_add(a, b): return (a + b) % q
    def f_sub(a, b): return (a - b) % q
    def f_mul(a, b): return (a * b) % q

    def build_luw(d):
        M = q ** d
        total_states = 2 * M
        rules = {}
        rules[2] = (1, 1)
        if d >= 3: rules[3] = (2, 1)
        if d >= 4: rules[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            base = c - r
            candidates = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
            rules[c] = candidates[r]
        coords_table = np.zeros((M, d), dtype=np.int32)
        for val in range(M):
            v = val
            for i in range(d - 1, -1, -1):
                coords_table[val, i] = v % q
                v //= q
        trans0 = np.zeros((M, q), dtype=np.int32)
        trans1 = np.zeros((M, q), dtype=np.int32)
        for val in range(M):
            C = coords_table[val]
            for fv in range(q):
                nc = np.zeros(d, dtype=np.int32)
                nc[0] = fv
                for ci in range(1, d):
                    c = ci + 1
                    a, b = rules[c]
                    na = fv if a == 1 else nc[a-1]
                    nc[ci] = f_add(C[ci], f_mul(na, C[b-1]))
                nv = 0
                for i in range(d): nv = nv * q + nc[i]
                trans0[val, fv] = nv
                nc1 = np.zeros(d, dtype=np.int32)
                nc1[0] = fv
                for ci in range(1, d):
                    c = ci + 1
                    a, b = rules[c]
                    cb = fv if b == 1 else nc1[b-1]
                    nc1[ci] = f_sub(C[ci], f_mul(C[a-1], cb))
                nv1 = 0
                for i in range(d): nv1 = nv1 * q + nc1[i]
                trans1[val, fv] = nv1
        visited = [-1] * total_states
        visited[0] = 0
        queue = [0]
        head = 0
        edges = []
        while head < len(queue):
            if time.time() > deadline: return None
            cs = queue[head]; head += 1
            cl = visited[cs]
            side = cs // M
            val = cs % M
            tt = trans0 if side == 0 else trans1
            ns = 1 - side
            for fv in range(q):
                nv = tt[val, fv]
                st = ns * M + nv
                if visited[st] == -1:
                    visited[st] = len(queue)
                    queue.append(st)
                if side == 0:
                    edges.append((cl, visited[st]))
        return len(queue), edges

    for d in range(3, 25):
        eg = 2*d + 2 if d % 2 == 0 else 2*d + 3
        if eg < g: continue
        if time.time() > deadline - 5.0: break
        res = build_luw(d)
        if res is not None: return res
    return build_luw(4)
