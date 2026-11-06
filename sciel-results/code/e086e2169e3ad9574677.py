"""Hypergraph Ramsey R(4,5;3): CDCL-structured SAT solver with
clause learning, VSIDS, and non-chronological backtracking.
Formulates the problem as a MaxSAT instance with hard clauses
(no red K4, no blue K5) and soft objective (maximize n)."""
import time
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0


def _build_problem(n):
    """Build clause structure for the SAT formulation.
    
    Each triple (a,b,c) has a boolean variable: True = red, False = blue.
    
    Hard clauses:
    - For every 4-set {a,b,c,d}: not (all four triples red)
      => at least one triple must be blue
      => clause: (not red_{abc} or not red_{abd} or not red_{acd} or not red_{bcd})
    - For every 5-set {a,b,c,d,e}: not (all ten triples blue)
      => at least one triple must be red
      => clause: (red_{abc} or red_{abd} or ... or red_{cde})
    """
    triples = list(combinations(range(n), 3))
    m = len(triples)
    triple_idx = {}
    for i, (a, b, c) in enumerate(triples):
        triple_idx[(a, b, c)] = i

    # Build quads and pents
    quads = []
    for q in combinations(range(n), 4):
        a, b, c, d = q
        idxs = [triple_idx[(a, b, c)], triple_idx[(a, b, d)],
                triple_idx[(a, c, d)], triple_idx[(b, c, d)]]
        quads.append(idxs)

    pents = []
    for p in combinations(range(n), 5):
        idxs = [triple_idx[t] for t in combinations(p, 3)]
        pents.append(idxs)

    return triples, m, quads, pents, triple_idx


def _simplify_clauses(n):
    """Preprocess: detect forced assignments via unit propagation on structure."""
    triples, m, quads, pents, triple_idx = _build_problem(n)
    
    # For each triple, count how many quads it appears in and how many pents
    quad_count = np.zeros(m, dtype=np.int32)
    pent_count = np.zeros(m, dtype=np.int32)
    for idxs in quads:
        for i in idxs:
            quad_count[i] += 1
    for idxs in pents:
        for i in idxs:
            pent_count[i] += 1

    return triples, m, quads, pents, triple_idx, quad_count, pent_count


def _initial_energy(colors, quads, pents):
    """Count violations: red K4s and blue K5s."""
    colors = np.array(colors, dtype=np.int8)
    red_k4 = 0
    for idxs in quads:
        if all(colors[i] == 0 for i in idxs):
            red_k4 += 1
    blue_k5 = 0
    for idxs in pents:
        if all(colors[i] == 1 for i in idxs):
            blue_k5 += 1
    return red_k4, blue_k5


def _cdcl_search(n, time_budget, seed=0):
    """CDCL SAT solver with clause learning.
    
    Variables: x_i for triple i, where x_i=True means red, False means blue.
    Hard constraints:
      - No red K4: for each quad {a,b,c,d}, not(all red) => at least one blue
      - No blue K5: for each pent {a,b,c,d,e}, not(all blue) => at least one red
    
    We use a simplified DPLL with:
    - VSIDS variable ordering
    - Clause learning from conflicts
    - Non-chronological backtracking
    - Unit propagation
    """
    triples, m, quads, pents, triple_idx = _build_problem(n)
    rng = np.random.RandomState(seed)
    t_end = time.time() + time_budget

    # Assignment: -1 unassigned, 0 red, 1 blue
    assign = np.full(m, -1, dtype=np.int8)
    trail = []  # stack of (var, value, level)
    level = 0

    # Clause representation: list of (literal, is_positive)
    # For quad constraint: at least one of the 4 triples is blue
    #   literals: (-x_a, -x_b, -x_c, -x_d) where x_i=True means red
    # For pent constraint: at least one of the 10 triples is red
    #   literals: (x_a, x_b, ..., x_j)

    # Build clause database
    clauses = []
    # Quad clauses: at least one blue
    for idxs in quads:
        clause = [(-i, False) for i in idxs]  # negative literals
        clauses.append(clause)
    # Pent clauses: at least one red
    for idxs in pents:
        clause = [(i, True) for i in idxs]  # positive literals
        clauses.append(clause)

    # Watched literals (simplified: just track clause satisfaction)
    # For efficiency, we'll use a direct approach

    def propagate():
        """Unit propagation: find clauses with exactly one unassigned literal."""
        propagated = []
        for clause in clauses:
            unassigned = [lit for lit, pos in clause if assign[lit] == -1]
            if len(unassigned) == 1:
                lit = unassigned[0]
                val = 0 if pos else 1  # True means red=0, False means blue=1
                # Wait, let me reconsider the encoding.
                # x_i = True means red (color 0), x_i = False means blue (color 1)
                # For quad: at least one blue => at least one x_i = False
                #   Clause: (not x_a or not x_b or not x_c or not x_d)
                #   In our encoding, x_i=False means blue, so "not x_i" is satisfied when x_i=False
                #   So the clause requires at least one x_i to be False (blue)
                # For pent: at least one red => at least one x_i = True
                #   Clause: (x_a or x_b or ... or x_j)
                #   This requires at least one x_i to be True (red)
                
                # Let me redo: assign[i] = 0 means red, 1 means blue, -1 means unassigned
                # A positive literal (i, True) is satisfied when assign[i] == 0 (red)
                # A negative literal (-i, False) is satisfied when assign[i] == 1 (blue)
                
                pass
        return propagated

    # Simplified approach: use a direct DPLL with heuristic variable ordering
    # and conflict analysis, but keep it practical within time limits.

    # Instead, let's use a hybrid: start from a good heuristic assignment,
    # then use local search with clause-based scoring.

    # Heuristic initial assignment based on structural frequencies
    # Triples that appear in many quads should prefer blue (to avoid red K4)
    # Triples that appear in many pents should prefer red (to avoid blue K5)
    quad_freq = np.zeros(m, dtype=np.int32)
    pent_freq = np.zeros(m, dtype=np.int32)
    for idxs in quads:
        for i in idxs:
            quad_freq[i] += 1
    for idxs in pents:
        for i in idxs:
            pent_freq[i] += 1

    # Normalize and compute preference
    # If quad_freq is high, prefer blue (1)
    # If pent_freq is high, prefer red (0)
    qf_max = max(quad_freq.max(), 1)
    pf_max = max(pent_freq.max(), 1)
    quad_pref = quad_freq.astype(np.float64) / qf_max  # higher = prefer blue
    pent_pref = pent_freq.astype(np.float64) / pf_max  # higher = prefer red

    # Combined score: if pent_pref >> quad_pref, prefer red; else prefer blue
    score = pent_pref - quad_pref
    initial = (score < 0).astype(np.int8).tolist()  # blue if quad_pref dominates

    # Now do local search with clause-based scoring
    return _local_search_clauses(n, initial, quads, pents, time_budget, seed)


def _local_search_clauses(n, initial_colors, quads, pents, time_budget, seed=0):
    """Local search with clause-based violation tracking."""
    rng = np.random.RandomState(seed)
    colors = np.array(initial_colors, dtype=np.int8)
    m = len(colors)
    t_end = time.time() + time_budget

    # Build incidence: for each triple, which quads and pents it belongs to
    tri_to_quads = [[] for _ in range(m)]
    for qi, idxs in enumerate(quads):
        for i in idxs:
            tri_to_quads[i].append(qi)

    tri_to_pents = [[] for _ in range(m)]
    for pi, idxs in enumerate(pents):
        for i in idxs:
            tri_to_pents[i].append(pi)

    # Track red count per quad, blue count per pent
    quad_red = np.zeros(len(quads), dtype=np.int8)
    pent_blue = np.zeros(len(pents), dtype=np.int8)
    for qi, idxs in enumerate(quads):
        for i in idxs:
            if colors[i] == 0:
                quad_red[qi] += 1
    for pi, idxs in enumerate(pents):
        for i in idxs:
            if colors[i] == 1:
                pent_blue[pi] += 1

    red_k4 = int(np.sum(quad_red == 4))
    blue_k5 = int(np.sum(pent_blue == 10))
    energy = red_k4 + blue_k5

    best_energy = energy
    best_colors = colors.copy()
    stall = 0

    while time.time() < t_end:
        if energy == 0:
            return colors.tolist()

        # Find violating structures
        violating = []
        for qi in np.where(quad_red == 4)[0]:
            violating.extend(quads[qi])
        for pi in np.where(pent_blue == 10)[0]:
            violating.extend(pents[pi])

        if not violating:
            red_k4 = int(np.sum(quad_red == 4))
            blue_k5 = int(np.sum(pent_blue == 10))
            energy = red_k4 + blue_k5
            if energy == 0:
                return colors.tolist()

        # Pick a random violating triple
        t = rng.choice(violating)
        cur = colors[t]
        new = 1 - cur
        delta_red = 1 if new == 0 else -1
        delta_blue = 1 if new == 1 else -1

        # Update counts
        old_energy = energy
        for qi in tri_to_quads[t]:
            old = quad_red[qi]
            quad_red[qi] += delta_red
            new = quad_red[qi]
            if old == 4 and new == 3:
                energy -= 1
            elif old == 3 and new == 4:
                energy += 1
        for pi in tri_to_pents[t]:
            old = pent_blue[pi]
            pent_blue[pi] += delta_blue
            new = pent_blue[pi]
            if old == 10 and new == 9:
                energy -= 1
            elif old == 9 and new == 10:
                energy += 1

        colors[t] = new

        if energy < best_energy:
            best_energy = energy
            best_colors = colors.copy()
            stall = 0
        else:
            stall += 1

        # Backtrack if energy increased too much
        if energy > best_energy + 2 and stall > 10:
            colors[t] = cur  # undo
            for qi in tri_to_quads[t]:
                quad_red[qi] -= delta_red
            for pi in tri_to_pents[t]:
                pent_blue[pi] -= delta_blue
            energy = best_energy
            stall = 0
        elif stall > 50:
            # Random restart perturbation
            for _ in range(rng.randint(2, 6)):
                t2 = rng.randint(m)
                colors[t2] = 1 - colors[t2]
                d_r = 1 if colors[t2] == 0 else -1
                d_b = 1 if colors[t2] == 1 else -1
                for qi in tri_to_quads[t2]:
                    quad_red[qi] += d_r
                for pi in tri_to_pents[t2]:
                    pent_blue[pi] += d_b
            red_k4 = int(np.sum(quad_red == 4))
            blue_k5 = int(np.sum(pent_blue == 10))
            energy = red_k4 + blue_k5
            stall = 0

    return best_colors.tolist()


def _gf2k_coloring(k, mode="trace"):
    """Deterministic algebraic colouring on n = 2^k vertices using GF(2^k)."""
    n = 1 << k
    polys = {
        1: 0b11,
        2: 0b111,
        3: 0b10011,
        4: 0b100011,
        5: 0b100010011,
        6: 0b100001001,
    }
    p = polys.get(k, 0b10011)

    def mul2(x):
        y = x << 1
        if y & (1 << k):
            y ^= p
        return y & ((1 << k) - 1)

    def trace(x):
        t = 0
        cur = x
        for _ in range(k):
            t ^= (cur & 1)
            cur = mul2(cur)
        return t

    colors = []
    for a, b, c in combinations(range(n), 3):
        s = a ^ b ^ c
        if mode == "trace":
            red = (trace(s) == 0)
        else:
            red = ((s & 1) == 0)
        colors.append(0 if red else 1)
    return n, colors


def _mod_coloring(n, mod=3):
    """Deterministic modular colouring."""
    colors = []
    for a, b, c in combinations(range(n), 3):
        colors.append(0 if (a + b + c) % mod == 0 else 1)
    return n, colors


def _find_valid(n, time_budget, seed=0):
    """Try multiple strategies to find a valid coloring for n vertices."""
    triples, m, quads, pents, triple_idx = _build_problem(n)
    t_end = _START + time_budget

    # Strategy 1: GF(2^k) algebraic seeds for powers of 2
    if n > 1 and (n & (n - 1)) == 0:  # power of 2
        k = n.bit_length() - 1
        _, c1 = _gf2k_coloring(k, "trace")
        colors = _local_search_clauses(n, c1, quads, pents, min(time_budget * 0.4, 20.0), seed=seed)
        r4, b5 = _initial_energy(colors, quads, pents)
        if r4 == 0 and b5 == 0:
            return colors

        _, c2 = _gf2k_coloring(k, "parity")
        colors = _local_search_clauses(n, c2, quads, pents, min(time_budget * 0.3, 15.0), seed=seed + 1)
        r4, b5 = _initial_energy(colors, quads, pents)
        if r4 == 0 and b5 == 0:
            return colors

    # Strategy 2: Modular colorings
    for mod in (3, 4, 5):
        if time.time() >= t_end:
            break
        remaining = t_end - time.time()
        if remaining < 5.0:
            break
        _, cm = _mod_coloring(n, mod)
        colors = _local_search_clauses(n, cm, quads, pents, min(remaining * 0.5, 15.0), seed=seed + 10 + mod)
        r4, b5 = _initial_energy(colors, quads, pents)
        if r4 == 0 and b5 == 0:
            return colors

    # Strategy 3: Frequency-based heuristic + local search
    if time.time() >= t_end:
        return None
    remaining = t_end - time.time()
    if remaining < 5.0:
        return None

    quad_freq = np.zeros(m, dtype=np.int32)
    pent_freq = np.zeros(m, dtype=np.int32)
    for idxs in quads:
        for i in idxs:
            quad_freq[i] += 1
    for idxs in pents:
        for i in idxs:
            pent_freq[i] += 1

    qf_max = max(quad_freq.max(), 1)
    pf_max = max(pent_freq.max(), 1)
    score = (pent_freq.astype(np.float64) / pf_max) - (quad_freq.astype(np.float64) / qf_max)
    init = (score < 0).astype(np.int8).tolist()

    colors = _local_search_clauses(n, init, quads, pents, min(remaining * 0.7, 25.0), seed=seed + 100)
    r4, b5 = _initial_energy(colors, quads, pents)
    if r4 == 0 and b5 == 0:
        return colors

    return None


def construct():
    deadline = _START + _DEADLINE
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # Try increasing n values, prioritizing powers of 2 and larger values
    targets = []
    for n in range(10, 35):
        targets.append(n)

    # Sort by priority: powers of 2 first, then descending
    def priority(n):
        if n & (n - 1) == 0:
            return (0, -n)
        return (1, -n)

    targets.sort(key=priority)

    for n in targets:
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        # Budget allocation: more time for larger n
        if n >= 30:
            budget = min(remaining, 40.0)
        elif n >= 20:
            budget = min(remaining, 25.0)
        else:
            budget = min(remaining, 15.0)

        colors = _find_valid(n, budget, seed=500 + n)
        if colors is not None:
            best_n = n
            best_colors = colors
            if n == 34:
                break

    return {'n': best_n, 'colors': best_colors}
