import time
import random
import math

_DEADLINE = 82.0  # safety margin below 90s
_N = 256


def _validate(S):
    """Return True if S is a valid subset of [1,256]."""
    s = set(S)
    if len(s) != len(S):
        return False
    if any(x < 1 or x > _N for x in s):
        return False
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            if (a * b) % (a + b) == 0:
                return False
    return True


def _build_conflict_bitsets():
    """Build bitset representation of conflicts. conflict_bits[i] has bit j set if i conflicts with j."""
    conflict_bits = [0] * (_N + 1)
    for a in range(1, _N + 1):
        bits = 0
        for b in range(a + 1, _N + 1):
            if (a * b) % (a + b) == 0:
                bits |= (1 << b)
                conflict_bits[b] |= (1 << a)
        conflict_bits[a] = bits
    return conflict_bits


def _greedy_color_bound(verts, conflict_bits):
    """
    Greedy coloring of the subgraph induced by verts.
    Returns a list of colors (one per vertex in verts) such that adjacent vertices
    get different colors. The number of colors used is an upper bound on the
    maximum independent set size within verts.
    verts: list of vertex indices
    Returns (color_count, color_of) where color_of[v] is the color assigned to v.
    """
    color_of = {}
    color = 0
    remaining = list(verts)
    # Process vertices in order of decreasing degree within remaining (for tighter bound)
    # Actually for coloring upper bound, we want to assign colors greedily.
    # Standard Tomita: color in order, assign smallest available color.
    # We'll sort by degree descending for better coloring.
    rem_set = set(remaining)
    order = sorted(remaining, key=lambda v: sum(1 for u in remaining if u != v and (conflict_bits[v] >> u) & 1), reverse=True)
    for v in order:
        # Find smallest color not used by neighbors of v already colored
        used = set()
        for u in order:
            if u == v:
                break
            if u in color_of and (conflict_bits[v] >> u) & 1:
                used.add(color_of[u])
        c = 0
        while c in used:
            c += 1
        color_of[v] = c
        color = max(color, c + 1)
    return color, color_of


def _max_independent_set_tomita(conflict_bits, all_vertices, deadline, initial_best_size):
    """
    Tomita-style branch and bound for maximum independent set.
    all_vertices: list of all vertex indices (1..N)
    Returns (best_set, best_size)
    """
    best_set = []
    best_size = initial_best_size
    nodes = [0]
    
    # Convert to 0-indexed internally for bit operations? No, use 1-indexed bits.
    # vertices are 1..N, bit i in an int represents vertex i.
    
    def _popcount(x):
        return bin(x).count('1')
    
    def _iter_bits(bits):
        """Yield vertex indices from a bitset."""
        while bits:
            lsb = bits & (-bits)
            yield lsb.bit_length() - 1
            bits ^= lsb
    
    def bnb(cand_bits, current_set, current_size):
        """
        cand_bits: bitset of candidate vertices to consider
        current_set: list of vertices in current independent set
        current_size: len(current_set)
        """
        nonlocal best_set, best_size
        nodes[0] += 1
        
        if time.time() > deadline:
            return
        
        if cand_bits == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        # Upper bound via greedy coloring
        verts = list(_iter_bits(cand_bits))
        color_cnt, color_of = _greedy_color_bound(verts, conflict_bits)
        
        if current_size + color_cnt <= best_size:
            return
        
        # Branch: pick a vertex with highest color (last color class)
        # Tomita's algorithm: branch on vertices in the last color class
        # Actually, standard approach: pick a vertex, branch include/exclude
        # Let's use the approach: pick a pivot vertex, branch on include/exclude
        
        # Pick vertex with maximum degree in cand_bits
        max_deg = -1
        pivot = None
        for v in verts:
            deg = _popcount(conflict_bits[v] & cand_bits)
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        # Branch 1: include pivot
        new_cand = cand_bits & ~(1 << pivot) & ~conflict_bits[pivot]
        current_set.append(pivot)
        bnb(new_cand, current_set, current_size + 1)
        current_set.pop()
        
        # Branch 2: exclude pivot
        new_cand2 = cand_bits & ~(1 << pivot)
        bnb(new_cand2, current_set, current_size)
    
    all_bits = 0
    for v in all_vertices:
        all_bits |= (1 << v)
    
    bnb(all_bits, [], 0)
    return best_set, best_size


def _greedy_build(order, conflict_bits):
    """Greedy add elements in given order if they don't conflict with current set."""
    S = []
    S_set = set()
    for x in order:
        # Check if x conflicts with any in S_set
        ok = True
        for y in S_set:
            if (conflict_bits[x] >> y) & 1:
                ok = False
                break
        if ok:
            S.append(x)
            S_set.add(x)
    return S, S_set


def _local_search(S, conflict_bits, time_limit):
    """
    Local search: try 1-for-2 swaps (remove one, add two) to increase size.
    """
    S = list(S)
    S_set = set(S)
    remaining = set(range(1, _N + 1)) - S_set
    start = time.time()
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for r in list(S):
            if time.time() >= time_limit:
                break
            S_set.discard(r)
            # Find candidates that can be added
            candidates = []
            for x in remaining:
                ok = True
                for y in S_set:
                    if (conflict_bits[x] >> y) & 1:
                        ok = False
                        break
                if ok:
                    candidates.append(x)
            
            # Try to add two from candidates
            added = 0
            added_list = []
            for x in candidates:
                if added >= 2:
                    break
                ok = True
                for y in S_set:
                    if (conflict_bits[x] >> y) & 1:
                        ok = False
                        break
                if ok:
                    S.append(x)
                    S_set.add(x)
                    remaining.discard(x)
                    added += 1
                    added_list.append(x)
            
            if added > 1:
                # Remove r
                S.remove(r)
                improved = True
                break
            else:
                # Re-add r
                S_set.add(r)
    
    return S


def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    conflict_bits = _build_conflict_bitsets()
    
    def _deadline_ok():
        return time.time() - t0 < _DEADLINE
    
    # Start with all odds as a strong baseline
    best = list(range(1, _N + 1, 2))
    best_size = len(best)
    
    # 1) Multiple random greedy orderings to get a strong initial lower bound
    if _deadline_ok():
        for _ in range(500):
            if time.time() - t0 > _DEADLINE - 30:
                break
            order = list(range(1, _N + 1))
            rng.shuffle(order)
            S, S_set = _greedy_build(order, conflict_bits)
            if len(S) > best_size:
                best = S
                best_size = len(S)
    
    # 2) Local search on best
    if _deadline_ok():
        S_new = _local_search(best, conflict_bits, _DEADLINE - 20)
        if len(S_new) > best_size:
            best = S_new
            best_size = len(S_new)
    
    # 3) More random restarts with local search
    if _deadline_ok():
        for _ in range(200):
            if time.time() - t0 > _DEADLINE - 15:
                break
            order = list(range(1, _N + 1))
            rng.shuffle(order)
            S, S_set = _greedy_build(order, conflict_bits)
            S = _local_search(S, conflict_bits, time.time() + 0.5)
            if len(S) > best_size:
                best = S
                best_size = len(S)
    
    # 4) Exact branch-and-bound with Tomita coloring bound
    if _deadline_ok():
        deadline = _DEADLINE - 2
        bnb_set, bnb_size = _max_independent_set_tomita(
            conflict_bits, list(range(1, _N + 1)), deadline, best_size
        )
        if bnb_size > best_size and _validate(bnb_set):
            best = bnb_set
            best_size = bnb_size
    
    # Final safety: ensure validity
    if not _validate(best):
        best = list(range(1, _N + 1, 2))
    
    return best
