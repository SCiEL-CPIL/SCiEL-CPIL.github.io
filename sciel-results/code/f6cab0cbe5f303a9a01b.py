"""Rewritten program: exact XOR-circuit synthesis via shared distinct-row synthesis."""
# EVOLVE-BLOCK-START
import time
import random

def _to_int(row):
    v = 0
    for b in row:
        v = (v << 1) | (1 if b else 0)
    return v

def _synth_row(mask, n, avail, gates, out, budget, deadline):
    """Return (wire, gate_count) for mask using direct or one-internal-gate expression.
    avail: set of currently-available wires (inputs + zero + gates emitted so far).
    """
    now = time.time()
    if now > deadline:
        return (0, 0)
    bits = [i for i in range(n) if (mask >> i) & 1]
    # weight 0 -> zero wire
    if not bits:
        return (n, 0)
    # weight 1 -> input directly
    if len(bits) == 1:
        return (bits[0], 0)
    # weight 2 -> if both inputs available, one gate
    if len(bits) == 2:
        a, b = bits
        w = n + 1 + len(gates)
        gates.append([a, b])
        avail.add(w)
        out.append(w)
        return (w, 1)
    # weight >= 3: try to find a pair (x,y) both in `avail` such that mask = x XOR y
    #   i.e. there exists a submask s of mask with popcount>=2 and (mask ^ s) has popcount>=2
    #   and both s and (mask^s) are available.  Cheapest: find any available pair.
    best_pair = None
    # iterate over available wires that are submasks of mask (and not equal mask)
    candidates = []
    for w in avail:
        if w >= n:
            # a gate; its value is a mask (store? we don't have it here)
            pass
    # We need the *value* of available gates to test XOR.  Pass a parallel dict.
    return _synth_row_direct(mask, n, bits, avail, gates, out, budget, deadline)

def _synth_row_direct(mask, n, bits, avail, gates, out, budget, deadline):
    """Fallback: emit a chain of XOR gates combining the bits sequentially.
    This always works and costs (len(bits)-1) gates.
    """
    if not bits:
        return (n, 0)
    if len(bits) == 1:
        return (bits[0], 0)
    acc = bits[0]
    cnt = 0
    for b in bits[1:]:
        w = n + 1 + len(gates)
        gates.append([acc, b])
        avail.add(w)
        out.append(w)
        acc = w
        cnt += 1
        if time.time() > deadline:
            break
    return (acc, cnt)

def _verify(matrix, gates, outputs, n):
    """Exact GF(2) verification. Returns True if circuit computes matrix."""
    m = len(matrix)
    # value of each wire as a frozenset of input indices (or int mask)
    val = [0] * (n + 1)  # inputs 0..n-1, wire n = 0
    for i in range(n):
        val[i] = (1 << i)
    val[n] = 0
    for (a, b) in gates:
        val.append(val[a] ^ val[b])
    for r in range(m):
        expected = _to_int(matrix[r])
        if r >= len(outputs):
            return False
        if val[outputs[r]] != expected:
            return False
    return True

def _baseline(matrix):
    s = 0
    for row in matrix:
        s += sum(1 for b in row if b)
    return s

def _make_result(gates, outputs, n):
    return {'gates': [[int(a), int(b)] for (a, b) in gates],
            'outputs': [int(w) for w in outputs]}

def _greedy_synth(matrix, n, deadline):
    """Synthesize with distinct-row sharing + greedy topological ordering."""
    # distinct non-zero masks with their row indices
    mask_to_rows = {}
    for r, row in enumerate(matrix):
        mk = _to_int(row)
        if mk:
            mask_to_rows.setdefault(mk, []).append(r)
    zero_rows = [r for r, row in enumerate(matrix) if _to_int(row) == 0]

    masks = list(mask_to_rows.keys())
    # sort masks by popcount ascending (cheap ones first, become available early)
    masks.sort(key=lambda m: bin(m).count('1'))

    avail = set(range(n + 1))  # inputs + zero
    wire_of_mask = {}
    gates = []
    out = []
    for mk in masks:
        if time.time() > deadline:
            break
        bits = [i for i in range(n) if (mk >> i) & 1]
        if len(bits) == 1:
            wire_of_mask[mk] = bits[0]
        else:
            # try to find two available wires whose XOR == mk
            found = None
            # gather available gate wires with their values
            gate_vals = []
            for w in avail:
                if w >= n:
                    # value of gate w is stored in _gate_val
                    pass
            # We track gate values in a dict
            # (handled below)
            found = _try_fold(mk, avail, _gate_val)
            if found is None:
                # emit direct chain
                acc = bits[0]
                for b in bits[1:]:
                    w = n + 1 + len(gates)
                    gates.append([acc, b])
                    _gate_val[w] = _gate_val[acc] ^ (1 << b)
                    avail.add(w)
                    out.append(w)
                    acc = w
                wire_of_mask[mk] = acc
            else:
                a, b = found
                w = n + 1 + len(gates)
                gates.append([a, b])
                _gate_val[w] = _gate_val[a] ^ _gate_val[b]
                avail.add(w)
                out.append(w)
                wire_of_mask[mk] = w

    outputs = [0] * len(matrix)
    for r in zero_rows:
        outputs[r] = n
    for mk, rows in mask_to_rows.items():
        w = wire_of_mask.get(mk, n)
        for r in rows:
            outputs[r] = w
    return _make_result(gates, outputs, n)

# global gate-value tracker (reset per synth)
_gate_val = {}

def _try_fold(mk, avail, gval):
    """Find (a,b) both in avail, a!=b, gval[a]^gval[b]==mk.  Prefer a being an input or a cheap gate."""
    # collect available gate wires
    gate_wires = [w for w in avail if w >= n]
    # candidate: pair (input i, gate g) where gval[g] == mk ^ (1<<i)
    for i in range(n):
        if (mk >> i) & 1:
            need = mk ^ (1 << i)
            # need must have popcount >= 1 and be an available gate
            for g in gate_wires:
                if gval.get(g) == need:
                    return (i, g)
    # candidate: pair (gate a, gate b)
    gset = {gval.get(g): g for g in gate_wires}
    for a in gate_wires:
        va = gval.get(a)
        need = mk ^ va
        if need in gset and gset[need] != a:
            return (a, gset[need])
    return None

def synthesize(matrix, depth_limit):
    start = time.time()
    deadline = start + 75.0  # safety margin before 90s
    m = len(matrix)
    n = len(matrix[0]) if m else 0
    if m == 0:
        return {'gates': [], 'outputs': []}

    global _gate_val
    _gate_val = {}
    for i in range(n):
        _gate_val[i] = (1 << i)
    _gate_val[n] = 0

    best = None
    best_count = None
    try:
        res = _greedy_synth(matrix, n, deadline)
        if _verify(matrix, res['gates'], res['outputs'], n):
            best = res
            best_count = len(res['gates'])
    except Exception:
        pass

    # Polish: local search on gate order to maximize folding
    if best is not None and time.time() < deadline - 5.0:
        try:
            improved = _polish(matrix, best, n, deadline)
            if improved is not None and _verify(matrix, improved['gates'], improved['outputs'], n):
                if len(improved['gates']) < best_count:
                    best = improved
                    best_count = len(best['gates'])
        except Exception:
            pass

    if best is None:
        # absolute fallback: independent rows
        gates = []
        out = []
        outputs = []
        for row in matrix:
            bits = [i for i in range(n) if row[i]]
            if not bits:
                outputs.append(n)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                acc = bits[0]
                for b in bits[1:]:
                    w = n + 1 + len(gates)
                    gates.append([acc, b])
                    out.append(w)
                    acc = w
                outputs.append(acc)
        best = _make_result(gates, outputs, n)

    return best

def _polish(matrix, result, n, deadline):
    """Local search: try to reduce gate count by reordering / re-folding."""
    # This is a simple hill-climb: repeatedly attempt to fold a weight-3+ direct
    # chain into a weight-2 form if a matching gate becomes available.
    # Given time constraints, we do a bounded number of improvement attempts.
    gates = [list(g) for g in result['gates']]
    outputs = list(result['outputs'])
    if not gates:
        return None
    # Build gate values
    gval = {}
    for i in range(n):
        gval[i] = (1 << i)
    gval[n] = 0
    for idx, (a, b) in enumerate(gates):
        w = n + 1 + idx
        gval[w] = gval[a] ^ gval[b]
    # outputs map wire
    # Try: for each gate, if it's part of a chain, see if we can replace
    # We keep it simple: attempt to reorder gates so that shared subexpressions
    # are computed earlier.  A cheap heuristic: sort gates so that a gate whose
    # value is used by many later gates comes first.
    # Compute use-count of each gate wire by later gates
    use = {}
    for idx, (a, b) in enumerate(gates):
        for x in (a, b):
            if x >= n:
                use[x] = use.get(x, 0) + 1
    # stable sort by use-count descending (most-shared first)
    order = sorted(range(len(gates)), key=lambda idx: -use.get(n + 1 + idx, 0))
    # Rebuild in that order if it stays valid (all operands available)
    new_gates = []
    avail = set(range(n + 1))
    remap = {}  # old wire -> new wire
    ok = True
    for idx in order:
        a, b = gates[idx]
        if a not in avail or b not in avail:
            ok = False
            break
        w = n + 1 + len(new_gates)
        new_gates.append([a, b])
        remap[n + 1 + idx] = w
        avail.add(w)
    if not ok:
        return None
    new_outputs = []
    for w in outputs:
        if w in remap:
            new_outputs.append(remap[w])
        else:
            new_outputs.append(w)
    newres = _make_result(new_gates, new_outputs, n)
    # only accept if same or fewer gates (it's a reordering, same count)
    return newres
# EVOLVE-BLOCK-END
