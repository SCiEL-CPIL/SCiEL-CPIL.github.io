# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def synthesize(matrix, depth_limit):
    t0 = time.time()
    deadline = t0 + 55.0

    if hasattr(matrix, 'shape'):
        M = np.array(matrix, dtype=np.uint8)
    else:
        M = np.array(matrix, dtype=np.uint8)

    m, n = M.shape
    rows = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if M[i, j]:
                bits |= (1 << j)
        rows.append(bits)

    wire_to_vec = {}
    vec_to_wires = {}

    for j in range(n):
        v = (1 << j)
        wire_to_vec[j] = v
        vec_to_wires.setdefault(v, []).append(j)
    wire_to_vec[n] = 0
    vec_to_wires.setdefault(0, []).append(n)

    gates = []
    next_wire = n + 1
    target_set = set(rows)

    def get_vec(w):
        return wire_to_vec[w]

    def add_gate(wa, wb):
        nonlocal next_wire
        va = wire_to_vec[wa]
        vb = wire_to_vec[wb]
        vnew = va ^ vb
        if vnew in vec_to_wires and vec_to_wires[vnew]:
            return vec_to_wires[vnew][0]
        gates.append([wa, wb])
        wire_to_vec[next_wire] = vnew
        vec_to_wires.setdefault(vnew, []).append(next_wire)
        w = next_wire
        next_wire += 1
        return w

    unsatisfied = [i for i in range(m) if rows[i] not in vec_to_wires]

    max_gates = 512
    while unsatisfied and len(gates) < max_gates:
        if time.time() > deadline - 5:
            break

        current_vecs = list(wire_to_vec.values())
        wire_ids = list(wire_to_vec.keys())
        nw = len(current_vecs)

        best_score = -1
        best_wa = None
        best_wb = None

        if nw <= 48:
            pairs = [(a, b) for a in range(nw) for b in range(a + 1, nw)]
        else:
            pairs = []
            for _ in range(1000):
                a = random.randrange(nw)
                b = random.randrange(nw)
                if a != b:
                    pairs.append((a, b))

        for a, b in pairs:
            va = current_vecs[a]
            vb = current_vecs[b]
            vnew = va ^ vb
            if vnew in vec_to_wires and vec_to_wires[vnew]:
                continue
            score = 0
            if vnew in target_set:
                score = 100
            else:
                for i in range(nw):
                    if (vnew ^ current_vecs[i]) in target_set:
                        score = 10
                        break
            if score > best_score:
                best_score = score
                best_wa = wire_ids[a]
                best_wb = wire_ids[b]

        if best_wa is None:
            if nw >= 2:
                a, b = random.sample(range(nw), 2)
                vnew = current_vecs[a] ^ current_vecs[b]
                if vnew in vec_to_wires and vec_to_wires[vnew]:
                    break
                best_wa = wire_ids[a]
                best_wb = wire_ids[b]
            else:
                break

        add_gate(best_wa, best_wb)
        unsatisfied = [i for i in unsatisfied if rows[i] not in vec_to_wires]

    for i in range(m):
        if time.time() > deadline - 2:
            break
        v = rows[i]
        if v not in vec_to_wires or not vec_to_wires[v]:
            bits = [j for j in range(n) if (v >> j) & 1]
            if not bits:
                w = n
            elif len(bits) == 1:
                w = bits[0]
            else:
                temp = list(bits)
                while len(temp) > 1:
                    wa = temp.pop(0)
                    wb = temp.pop(0)
                    w = add_gate(wa, wb)
                    temp.append(w)
                w = temp[0]
            vec_to_wires.setdefault(v, []).append(w)

    outputs = []
    for v in rows:
        if v in vec_to_wires and vec_to_wires[v]:
            outputs.append(vec_to_wires[v][0])
        else:
            outputs.append(n)

    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
