"""Shared XOR-circuit synthesis: global 2-bit edge sharing (Steiner-style)."""
# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    import numpy as np

    A = np.asarray(matrix)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    m, n = A.shape
    depth_limit = 16 if depth_limit is None else int(depth_limit)
    depth_limit = min(depth_limit, 16)

    # rows = outputs, cols = inputs (standard linear-map convention y = A x)
    row_masks = []
    for r in range(m):
        mk = 0
        row = A[r]
        for c in range(n):
            if row[c]:
                mk |= (1 << c)
        row_masks.append(mk)

    def build_independent():
        gates = []
        next_wire = n + 1
        outputs = []
        for mk in row_masks:
            bits = [c for c in range(n) if (mk >> c) & 1]
            if not bits:
                outputs.append(n)
                continue
            if len(bits) == 1:
                outputs.append(bits[0])
                continue
            cur = bits[0]
            for b in bits[1:]:
                gates.append([cur, b])
                cur = next_wire
                next_wire += 1
            outputs.append(cur)
        return gates, outputs

    def verify(gates, outputs):
        val = list(range(n)) + [0]
        for a, b in gates:
            val.append(val[a] ^ val[b])
        for j, ow in enumerate(outputs):
            if ow >= len(val) or val[ow] != row_masks[j]:
                return False
        depth = [0] * len(val)
        for k, (a, b) in enumerate(gates):
            depth[n + 1 + k] = max(depth[a], depth[b]) + 1
        if max(depth) > depth_limit:
            return False
        return True

    # ---- shared 2-bit edge construction ----
    used_edges = set()
    row_terms = []
    for mk in row_masks:
        bits = [c for c in range(n) if (mk >> c) & 1]
        terms = []
        rem = list(bits)
        # reuse-biased greedy matching: prefer pairs already seen globally
        while len(rem) >= 2:
            i = rem[0]
            chosen = None
            for idx in range(1, len(rem)):
                j = rem[idx]
                key = (i, j) if i < j else (j, i)
                if key in used_edges:
                    chosen = (idx, j)
                    break
            if chosen is None:
                chosen = (1, rem[1])
            idx, j = chosen
            key = (i, j) if i < j else (j, i)
            terms.append(('edge', key))
            used_edges.add(key)
            rem.pop(idx)
            rem.pop(0)
        if rem:
            terms.append(('single', rem[0]))
        row_terms.append(terms)

    edges = sorted(used_edges)
    edge_wire = {}
    gates = []
    next_wire = n + 1
    for e in edges:
        i, j = e
        edge_wire[e] = next_wire
        gates.append([i, j])
        next_wire += 1

    outputs = []
    for terms in row_terms:
        wires = []
        for t in terms:
            if t[0] == 'edge':
                wires.append(edge_wire[t[1]])
            else:
                wires.append(t[1])
        if not wires:
            outputs.append(n)
        elif len(wires) == 1:
            outputs.append(wires[0])
        else:
            cur = wires[0]
            for w in wires[1:]:
                gates.append([cur, w])
                cur = next_wire
                next_wire += 1
            outputs.append(cur)

    if verify(gates, outputs):
        return {'gates': gates, 'outputs': outputs}

    # fallback
    gates, outputs = build_independent()
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
