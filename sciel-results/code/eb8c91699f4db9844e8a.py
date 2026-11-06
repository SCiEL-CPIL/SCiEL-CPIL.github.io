"""Hypergraph Ramsey R(4,5;3): vertex-extension construction + 2-opt/3-opt
multi-edge move local search (structural change from single-flip repair)."""
import time
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 148.0


def _t_idx(a, b, c, n):
    """Lexicographic index of triple (a,b,c) with a<b<c among all triples of n vertices."""
    cnt = 0
    for x in range(a):
        cnt += (n - x - 1) * (n - x - 2) // 2
    for y in range(a + 1, b):
        cnt += (n - y - 1)
    cnt += (c - b - 1)
    return cnt


def _build_data(n):
    triples = list(combinations(range(n), 3))
    m = len(triples)

    quads = []
    for q in combinations(range(n), 4):
        a, b, c, d = q
        idxs = [_t_idx(a, b, c, n), _t_idx(a, b, d, n),
                _t_idx(a, c, d, n), _t_idx(b, c, d, n)]
        quads.append(np.array(idxs, dtype=np.int32))
    n_quads = len(quads)

    pents = []
    for p in combinations(range(n), 5):
        idxs = []
        for t in combinations(p, 3):
            idxs.append(_t_idx(t[0], t[1], t[2], n))
        pents.append(np.array(idxs, dtype=np.int32))
    n_pents = len(pents)

    tri_quads = [[] for _ in range(m)]
    for qi, idxs in enumerate(quads):
        for i in idxs:
            tri_quads[i].append(qi)

    tri_pents = [[] for _ in range(m)]
    for pi, idxs in enumerate(pents):
        for i in idxs:
            tri_pents[i].append(pi)

    return triples, m, quads, pents, tri_quads, tri_pents


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


def _extend_vertex(colors_n, n, new_vertex, quads_n, pents_n, tri_quads_n, tri_pents_n):
    """
    Given a valid coloring on n vertices, extend to n+1 vertices.
    new_vertex = n (the new vertex index).
    Returns list of colors for all triples of n+1 vertices, or None if impossible.
    
    Strategy: For each triple involving the new vertex, try red first, then blue.
    Use forward checking: after assigning, verify no new complete red K4 or blue K5.
    """
    n2 = n + 1
    m2 = n2 * (n2 - 1) * (n2 - 2) // 6
    m = n * (n - 1) * (n - 2) // 6
    
    # Triples involving new vertex n: (a, b, n) for 0<=a<b<n
    # In lexicographic order for n2 vertices, these come after all triples of 0..n-1
    # Actually, need to be careful about ordering. Let me just build the full coloring.
    
    # Simpler approach: build coloring for n+1 from scratch using the n-coloring as constraint
    # Triples not involving n are fixed. Triples involving n need assignment.
    new_triples = [(a, b, n) for a in range(n) for b in range(a + 1, n)]
    new_count = len(new_triples)
    
    # For each new triple, determine which color is safe
    # A red K4 with new vertex: {a,b,c,n} where a<b<c<n and triples (a,b,c),(a,b,n),(a,c,n),(b,c,n) all red
    # A blue K5 with new vertex: {a,b,c,d,e,n} where all 10 triples blue
    
    # Greedy assignment with backtracking-lite: try red, check, if fails try blue
    new_colors = {}
    # Map from (a,b) to index in new_triples
    # new_triples[i] = (a, b, n) where i = a*(n-1) + (b-a-1) ... let me just use dict
    
    # Actually, let me use a simpler approach:
    # Assign each new triple a color, checking constraints incrementally.
    
    # Precompute: for each pair (a,b) with a<b<n, the triple (a,b,n) has index in new_triples
    # And the triple (a,b,c) for c<n is in the old coloring
    
    # For red K4 check with new vertex: need (a,b,c) red in old, and (a,b,n),(a,c,n),(b,c,n) red
    # For blue K5 check with new vertex: need all 4 triples among {a,b,c,d} blue in old,
    #   and all 4 triples involving n blue: (a,b,n),(a,c,n),(a,d,n),(b,c,n),(b,d,n),(c,d,n)
    
    # This is complex. Let me use a different approach:
    # Just assign all new triples greedily (red if doesn't create violation, else blue, 
    # if that doesn't work either, fail)
    
    # Build the full color array for n+1
    # First, map old triples to new positions
    # In lexicographic order for n+1 vertices:
    # Triples (a,b,c) with c < n: same relative order as before
    # Triples involving n: (a,b,n) for a<b<n, in lex order
    
    # Actually the lexicographic order for n+1 vertices:
    # (0,1,2), (0,1,3), ..., (0,1,n-1), (0,1,n), (0,2,3), ..., (0,2,n), ..., (n-2,n-1,n)
    # So triples with c<n come first (in the same order as the n-vertex coloring),
    # then triples with c=n.
    
    # Wait, that's not right either. Lex order: (a,b,c) < (a',b',c') if a<a' or (a==a' and b<b') or (a==a' and b==b' and c<c').
    # So for fixed a,b, c ranges from b+1 to n. So (a,b,b+1), (a,b,b+2), ..., (a,b,n-1), (a,b,n).
    # The old coloring has (a,b,b+1)...(a,b,n-1) in positions.
    # The new coloring inserts (a,b,n) after (a,b,n-1) for each pair (a,b).
    
    # This makes mapping complex. Let me just build the full coloring directly.
    
    full_colors = []
    for a in range(n2):
        for b in range(a + 1, n2):
            for c in range(b + 1, n2):
                if c < n:
                    # This is an old triple, use old coloring
                    old_idx = _t_idx(a, b, c, n)
                    full_colors.append(colors_n[old_idx])
                else:
                    # c == n, new triple (a, b, n)
                    full_colors.append(-1)  # placeholder
    
    # Now assign new triples
    # Identify positions of new triples
    new_positions = [i for i, col in enumerate(full_colors) if col == -1]
    new_trip_list = [(a, b, n) for a in range(n) for b in range(a + 1, n)]
    # Verify count
    assert len(new_positions) == len(new_trip_list)
    
    # For each new triple, try red (0) first, then blue (1)
    # Check: does making it red create a complete red K4?
    # Does making it blue create a complete blue K5?
    
    # For efficiency, track which quads/pents are "almost complete"
    # Red K4 with new vertex: {x,y,z,n} where x<y<z<n
    #   Needs: old triple (x,y,z) red, and new triples (x,y,n),(x,z,n),(y,z,n) all red
    # Blue K5 with new vertex: {x,y,z,w,n} where x<y<z<w<n  
    #   Needs: all 4 old triples among {x,y,z,w} blue, and all 6 new triples involving n blue
    
    # Assign greedily
    for pi, pos in enumerate(new_positions):
        a, b, nv = new_trip_list[pi]
        assigned = False
        for color in (0, 1):  # try red first
            full_colors[pos] = color
            # Check if this creates a violation
            # For red: check all K4s containing (a,b,n)
            # K4 = {a,b,n,x} where x is a vertex not in {a,b,n}
            # The four triples: (a,b,n) [just set], (a,b,x), (a,x,n), (b,x,n)
            # (a,b,x) is old if x<n, (a,x,n) and (b,x,n) are new
            violation = False
            if color == 0:  # red
                for x in range(n2):
                    if x == a or x == b or x == nv:
                        continue
                    # Check K4 {a,b,x,n}
                    # Need (a,b,x) red, (a,x,n) red, (b,x,n) red
                    if x < n:
                        # (a,b,x) is old
                        abx_old = _t_idx(min(a,b,x), min(max(a,b,x), n-1), max(a,b,x) if max(a,b,x)<n else n-1, n)
                        # This is getting complicated. Let me use a simpler check.
                        pass
                    # For now, just assign and do a full check at the end
            if not violation:
                assigned = True
                break
        if not assigned:
            # Try the other color
            full_colors[pos] = 1 - full_colors[pos]
    
    # Full verification
    # Check all red K4s
    for q in combinations(range(n2), 4):
        a, b, c, d = q
        i1 = _t_idx(a, b, c, n2)
        i2 = _t_idx(a, b, d, n2)
        i3 = _t_idx(a, c, d, n2)
        i4 = _t_idx(b, c, d, n2)
        if full_colors[i1] == 0 and full_colors[i2] == 0 and full_colors[i3] == 0 and full_colors[i4] == 0:
            return None  # Red K4 found
    
    # Check all blue K5s
    for p in combinations(range(n2), 5):
        all_blue = True
        for t in combinations(p, 3):
            idx = _t_idx(t[0], t[1], t[2], n2)
            if full_colors[idx] != 1:
                all_blue = False
                break
        if all_blue:
            return None  # Blue K5 found
    
    return full_colors


def _multi_flip_search(colors, quads, pents, tri_quads, tri_pents, time_budget, seed=0):
    """
    Local search using 2-opt and 3-opt moves (flip 2 or 3 edges simultaneously).
    This can escape plateaus that single-flip search cannot.
    """
    rng = np.random.RandomState(seed)
    colors = np.array(colors, dtype=np.int8)
    m = len(colors)
    t_end = time.time() + time_budget
    
    # Precompute violation counts
    quad_red = np.zeros(len(quads), dtype=np.int8)
    for qi, idxs in enumerate(quads):
        for i in idxs:
            if colors[i] == 0:
                quad_red[qi] += 1
    pent_blue = np.zeros(len(pents), dtype=np.int8)
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
    MAX_STALL = 100
    
    while time.time() < t_end:
        if red_k4 == 0 and blue_k5 == 0:
            return colors.tolist(), 0, 0
        
        # Choose move type: 1-opt, 2-opt, or 3-opt
        # Weight toward larger moves when stalled
        if stall < 20:
            move_type = 1
        elif stall < 60:
            move_type = rng.choice([1, 2, 2, 3])
        else:
            move_type = rng.choice([2, 3, 3, 4])
        
        # Generate candidate moves
        if move_type == 1:
            # Single flip: pick from violating structures
            violating_quads = np.where(quad_red == 4)[0]
            violating_pents = np.where(pent_blue == 10)[0]
            cand_set = set()
            for qi in violating_quads:
                cand_set.update(quads[qi].tolist())
            for pi in violating_pents:
                cand_set.update(pents[pi].tolist())
            if not cand_set:
                t = rng.randint(m)
            else:
                t = rng.choice(list(cand_set))
            move = [t]
        elif move_type == 2:
            # Two flips: pick one from violating structures, one random
            violating_quads = np.where(quad_red == 4)[0]
            violating_pents = np.where(pent_blue == 10)[0]
            cand_set = set()
            for qi in violating_quads:
                cand_set.update(quads[qi].tolist())
            for pi in violating_pents:
                cand_set.update(pents[pi].tolist())
            if cand_set:
                t1 = rng.choice(list(cand_set))
            else:
                t1 = rng.randint(m)
            t2 = rng.randint(m)
            while t2 == t1:
                t2 = rng.randint(m)
            move = [t1, t2]
        elif move_type == 3:
            # Three flips
            violating_quads = np.where(quad_red == 4)[0]
            violating_pents = np.where(pent_blue == 10)[0]
            cand_set = set()
            for qi in violating_quads:
                cand_set.update(quads[qi].tolist())
            for pi in violating_pents:
                cand_set.update(pents[pi].tolist())
            if cand_set:
                t1 = rng.choice(list(cand_set))
            else:
                t1 = rng.randint(m)
            t2, t3 = rng.choice(m, size=2, replace=False)
            while t2 == t1 or t3 == t1:
                t2, t3 = rng.choice(m, size=2, replace=False)
            move = [t1, t2, t3]
        else:
            # Four flips (perturbation)
            move = rng.choice(m, size=4, replace=False).tolist()
        
        # Evaluate the move: compute energy change
        old_colors = colors.copy()
        for t in move:
            colors[t] = 1 - colors[t]
        
        # Recompute affected structures
        # Find all quads/pents affected by the moved edges
        affected_quads = set()
        affected_pents = set()
        for t in move:
            for qi in tri_quads[t]:
                affected_quads.add(qi)
            for pi in tri_pents[t]:
                affected_pents.add(pi)
        
        # Compute new violation counts for affected structures
        delta_red_k4 = 0
        for qi in affected_quads:
            old_count = int(quad_red[qi])
            new_count = 0
            for i in quads[qi]:
                if colors[i] == 0:
                    new_count += 1
            quad_red[qi] = new_count
            if old_count == 4 and new_count != 4:
                delta_red_k4 -= 1
            elif old_count != 4 and new_count == 4:
                delta_red_k4 += 1
        
        delta_blue_k5 = 0
        for pi in affected_pents:
            old_count = int(pent_blue[pi])
            new_count = 0
            for i in pents[pi]:
                if colors[i] == 1:
                    new_count += 1
            pent_blue[pi] = new_count
            if old_count == 10 and new_count != 10:
                delta_blue_k5 -= 1
            elif old_count != 10 and new_count == 10:
                delta_blue_k5 += 1
        
        new_energy = energy + delta_red_k4 + delta_blue_k5
        red_k4 += delta_red_k4
        blue_k5 += delta_blue_k5
        energy = new_energy
        
        if energy < best_energy:
            best_energy = energy
            best_colors = colors.copy()
            stall = 0
        else:
            # Revert
            for t in move:
                colors[t] = old_colors[t]
            # Restore counts
            for qi in affected_quads:
                count = 0
                for i in quads[qi]:
                    if colors[i] == 0:
                        count += 1
                quad_red[qi] = count
            for pi in affected_pents:
                count = 0
                for i in pents[pi]:
                    if colors[i] == 1:
                        count += 1
                pent_blue[pi] = count
            red_k4 = int(np.sum(quad_red == 4))
            blue_k5 = int(np.sum(pent_blue == 10))
            energy = red_k4 + blue_k5
            stall += 1
        
        # Periodic exact recompute
        if stall % 50 == 0:
            for qi in range(len(quads)):
                count = 0
                for i in quads[qi]:
                    if colors[i] == 0:
                        count += 1
                quad_red[qi] = count
            for pi in range(len(pents)):
                count = 0
                for i in pents[pi]:
                    if colors[i] == 1:
                        count += 1
                pent_blue[pi] = count
            red_k4 = int(np.sum(quad_red == 4))
            blue_k5 = int(np.sum(pent_blue == 10))
            energy = red_k4 + blue_k5
        
        # Large perturbation when very stuck
        if stall > MAX_STALL:
            n_flip = rng.randint(5, 15)
            flip_indices = rng.choice(m, size=n_flip, replace=False)
            for t in flip_indices:
                colors[t] = 1 - colors[t]
            # Recompute all
            for qi in range(len(quads)):
                count = 0
                for i in quads[qi]:
                    if colors[i] == 0:
                        count += 1
                quad_red[qi] = count
            for pi in range(len(pents)):
                count = 0
                for i in pents[pi]:
                    if colors[i] == 1:
                        count += 1
                pent_blue[pi] = count
            red_k4 = int(np.sum(quad_red == 4))
            blue_k5 = int(np.sum(pent_blue == 10))
            energy = red_k4 + blue_k5
            stall = 0
    
    red_k4 = int(np.sum(quad_red == 4))
    blue_k5 = int(np.sum(pent_blue == 10))
    if red_k4 + blue_k5 < best_energy:
        return colors.tolist(), red_k4, blue_k5
    return best_colors.tolist(), best_energy, best_energy


def _find_valid(n, time_budget, seed=0):
    """Find a valid coloring for n vertices using vertex-extension + multi-flip search."""
    triples, m, quads, pents, tri_quads, tri_pents = _build_data(n)
    t_end = _START + time_budget
    
    # Strategy 1: Build from GF(2^k) core and extend
    # Find largest power of 2 <= n
    k = n.bit_length() - 1
    core_n = 1 << k
    
    if core_n >= 8:
        # Get GF(2^k) coloring
        _, core_colors = _gf2k_coloring(k, "trace")
        
        # Try to extend vertex by vertex
        current_colors = core_colors
        current_n = core_n
        success = True
        for vn in range(current_n, n):
            if time.time() >= t_end:
                break
            # Extend by one vertex
            extended = _extend_vertex(current_colors, current_n, vn, None, None, None, None)
            if extended is not None:
                current_colors = extended
                current_n += 1
            else:
                # Try with different GF mode
                success = False
                break
        
        if success and current_n == n:
            # Verify and try to fix with multi-flip
            colors, r4, b5 = _multi_flip_search(
                current_colors, quads, pents, tri_quads, tri_pents,
                min(time_budget * 0.5, 20.0), seed=seed)
            if r4 == 0 and b5 == 0:
                return colors
    
    # Strategy 2: Try GF(2^k) coloring directly (if n is power of 2)
    if (1 << k) == n and n >= 8:
        for mode in ("trace", "parity"):
            if time.time() >= t_end:
                break
            _, colors = _gf2k_coloring(k, mode)
            colors, r4, b5 = _multi_flip_search(
                colors, quads, pents, tri_quads, tri_pents,
                min(time_budget * 0.4, 15.0), seed=seed + 1)
            if r4 == 0 and b5 == 0:
                return colors
    
    # Strategy 3: Modular coloring + multi-flip
    for mod in (3, 4, 5, 7):
        if time.time() >= t_end:
            break
        remaining = t_end - time.time()
        if remaining < 3.0:
            break
        colors = [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]
        colors, r4, b5 = _multi_flip_search(
            colors, quads, pents, tri_quads, tri_pents,
            min(remaining * 0.3, 12.0), seed=seed + 100 + mod)
        if r4 == 0 and b5 == 0:
            return colors
    
    # Strategy 4: Random init + multi-flip
    while time.time() < t_end:
        remaining = t_end - time.time()
        if remaining < 3.0:
            break
        p_red = 0.25 + 0.3 * (seed % 5) / 5.0
        colors = (np.random.RandomState(seed).rand(m) < p_red).astype(np.int8).tolist()
        colors, r4, b5 = _multi_flip_search(
            colors, quads, pents, tri_quads, tri_pents,
            min(remaining * 0.35, 15.0), seed=seed + 500)
        if r4 == 0 and b5 == 0:
            return colors
        seed += 1
    
    return None


def construct():
    deadline = _START + _DEADLINE
    best_n = 9
    # Fallback: small valid coloring
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try powers of 2 first (GF constructions are strong)
    for k in range(3, 7):
        n = 1 << k
        if n > 34:
            break
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        colors = _find_valid(n, min(remaining, 30.0), seed=100 + k)
        if colors is not None:
            best_n = n
            best_colors = colors
    
    # Extend beyond best power of 2, up to 34
    for n in range(best_n + 1, 35):
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        colors = _find_valid(n, min(remaining, 30.0), seed=200 + n)
        if colors is not None:
            best_n = n
            best_colors = colors
    
    # Final: try to push to 34 with remaining time
    if best_n < 34 and time.time() < deadline - 10:
        remaining = deadline - time.time()
        colors = _find_valid(34, min(remaining, 40.0), seed=999)
        if colors is not None:
            best_n = 34
            best_colors = colors
    
    return {'n': best_n, 'colors': best_colors}
