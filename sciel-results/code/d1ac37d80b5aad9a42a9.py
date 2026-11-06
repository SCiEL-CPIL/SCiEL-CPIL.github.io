def synthesize(matrix, depth_limit):
    import time
    start_time = time.time()
    DEADLINE = 85.0

    m = len(matrix)
    n = len(matrix[0])
    zero_wire = n

    rows = []
    for i in range(m):
        v = 0
        for j in range(n):
            if matrix[i][j]:
                v |= (1 << j)
        rows.append(v)

    wires = {}
    val_to_wire = {}
    for j in range(n):
        w = (1 << j)
        wires[j] = w
        val_to_wire[w] = j
    wires[zero_wire] = 0
    val_to_wire[0] = zero_wire

    next_wire = n + 1
    gates = []

    row_order = sorted(range(m), key=lambda i: bin(rows[i]).count('1'), reverse=True)

    for idx in row_order:
        if time.time() - start_time > DEADLINE:
            break
        target = rows[idx]
        if target in val_to_wire:
            continue

        # Try depth 1: target = wires[a] ^ wires[b]
        pair = None
        for wid, vm in wires.items():
            need = target ^ vm
            if need in val_to_wire:
                pair = (wid, val_to_wire[need])
                break

        if pair is not None:
            a, b = pair
            val = wires[a] ^ wires[b]
            if val not in val_to_wire:
                w = next_wire
                next_wire += 1
                gates.append([a, b])
                wires[w] = val
                val_to_wire[val] = w
            continue

        # Try depth 2: target = (wires[a] ^ wires[b]) ^ wires[c]
        triple = None
        wire_items = list(wires.items())
        for wid_c, vm_c in wire_items:
            mid = target ^ vm_c
            if mid in val_to_wire:
                continue
            for wid_a, vm_a in wire_items:
                need_b = mid ^ vm_a
                if need_b in val_to_wire and need_b != vm_a:
                    triple = (wid_a, val_to_wire[need_b], wid_c)
                    break
            if triple is not None:
                break

        if triple is not None:
            a, b, c = triple
            val1 = wires[a] ^ wires[b]
            if val1 in val_to_wire:
                w1 = val_to_wire[val1]
            else:
                w1 = next_wire
                next_wire += 1
                gates.append([a, b])
                wires[w1] = val1
                val_to_wire[val1] = w1
            val2 = val1 ^ wires[c]
            if val2 not in val_to_wire:
                w2 = next_wire
                next_wire += 1
                gates.append([w1, c])
                wires[w2] = val2
                val_to_wire[val2] = w2
            continue

        # Fallback: sequential XOR of input bits
        bits = [j for j in range(n) if (target >> j) & 1]
        if len(bits) <= 1:
            continue
        cur = bits[0]
        for j in bits[1:]:
            if time.time() - start_time > DEADLINE:
                break
            val = wires[cur] ^ (1 << j)
            if val in val_to_wire:
                cur = val_to_wire[val]
                continue
            w = next_wire
            next_wire += 1
            gates.append([cur, j])
            wires[w] = val
            val_to_wire[val] = w
            cur = w

    # Polish pass: rebuild gates, skipping redundant ones
    wires2 = {}
    val_to_wire2 = {}
    for j in range(n):
        w = (1 << j)
        wires2[j] = w
        val_to_wire2[w] = j
    wires2[zero_wire] = 0
    val_to_wire2[0] = zero_wire

    gates2 = []
    next_wire2 = n + 1
    for g in gates:
        if time.time() - start_time > DEADLINE:
            break
        a, b = g
        va = wires2.get(a)
        vb = wires2.get(b)
        if va is None or vb is None:
            continue
        val = va ^ vb
        if val in val_to_wire2:
            continue
        w = next_wire2
        next_wire2 += 1
        gates2.append([a, b])
        wires2[w] = val
        val_to_wire2[val] = w

    outputs = []
    for i in range(m):
        t = rows[i]
        if t in val_to_wire2:
            outputs.append(val_to_wire2[t])
        else:
            outputs.append(zero_wire)

    return {'gates': gates2, 'outputs': outputs}
