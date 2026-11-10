import time
from collections import deque
from itertools import combinations

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    if k == 4 and g == 10:
        # Try bipartite circulant on Z_160 with connection set of size 4
        # For no 4-cycles: all pair sums (s_i + s_j) for i<=j must be distinct mod N
        # This is a necessary condition. Let's find all valid 4-sets first, then check 6 and 8 cycles.
        
        N = 160
        # For 4-cycles: need all C(4,2)+4 = 10 sums s_i+s_j (i<=j) distinct mod N
        # Generate candidate sets: fix s_0=1 (WLOG by scaling if gcd(1,N)=1), 
        # but actually we can't scale since N=160. Let's just search.
        
        # Better approach: enumerate all 4-subsets of {1,...,159} with the no-4-cycle property
        # That's C(159,4) ~ 63M which is too many.
        
        # Smarter: use the structure. For no 4-cycles in bipartite circulant,
        # the set S must be a "sum-free" set in the sense that all pairwise sums are distinct.
        # This is equivalent to S being a Sidon set in Z_N (with the convention that we include repeated pairs).
        
        # Let's try a more targeted search. Known results suggest the (4,10)-cage
        # might use a specific algebraic structure.
        
        # Try: S = {1, a, b, c} where a,b,c are chosen to avoid sum collisions.
        # First filter: all 10 sums {1+1, 1+a, 1+b, 1+c, a+a, a+b, a+c, b+b, b+c, c+c} mod 160 must be distinct.
        
        # Let's do a smart enumeration: fix s0=1, then try s1, s2, s3 with pruning.
        # For each (s1,s2), check if sums so far are distinct, then try s3.
        
        found = False
        # To keep runtime manageable, limit the search space
        # Try s1 in a reasonable range, s2 in a reasonable range, s3 in a reasonable range
        
        # Actually, let's first try known good structures.
        # The (4,10)-cage might be related to the Hoffman-Singleton or other known cages.
        
        # Let me try a different approach: search over connection sets of the form
        # {1, a, b, c} with 1 < a < b < c < N, checking the 4-cycle condition first (fast),
        # then 6-cycle, then 8-cycle.
        
        # For efficiency, precompute which pairs give sum collisions.
        # Two pairs (x,y) and (x',y') with x<=y, x'<=y' collide if x+y = x'+y' mod N.
        
        # Let's enumerate more carefully with early termination.
        # Fix s0 = 1.
        # For s1: must have 1+1 != 1+s1, i.e., s1 != 0 mod N. Also 1+s1 != 1+s1 (trivially same).
        # Actually the 10 sums are: 2, 1+s1, 1+s2, 1+s3, 2*s1, s1+s2, s1+s3, 2*s2, s2+s3, 2*s3
        # All mod N must be distinct.
        
        # Let's try a bounded search with a time limit.
        best_result = None
        
        # Try structured families first
        # Family 1: {1, a, b, c} where b = 2a, c = 2b (geometric-like)
        for a in range(2, N//2):
            if time.time() > deadline - 10:
                break
            b = 2 * a % N
            c = 4 * a % N
            if b == 0 or c == 0 or b == c or a == b or a == c:
                continue
            S = [1, a, b, c]
            if _check_no_4cycle(N, S):
                if _check_no_6cycle(N, S):
                    if _check_no_8cycle(N, S):
                        return _build_circulant_edges(N, S)
        
        # Family 2: {1, a, b, c} where c = N/2 - a (symmetric)
        for a in range(2, N//2):
            if time.time() > deadline - 10:
                break
            c = N // 2 - a
            if c <= a:
                continue
            for b in range(a + 1, c):
                if time.time() > deadline - 10:
                    break
                S = [1, a, b, c]
                if _check_no_4cycle(N, S):
                    if _check_no_6cycle(N, S):
                        if _check_no_8cycle(N, S):
                            return _build_circulant_edges(N, S)
        
        # Family 3: Broader search with s0=1, s1 in [2,60], s2 in [s1+1,120], s3 in [s2+1, 159]
        for s1 in range(2, 60):
            if time.time() > deadline - 10:
                break
            for s2 in range(s1 + 1, 120):
                if time.time() > deadline - 10:
                    break
                for s3 in range(s2 + 1, N):
                    if time.time() > deadline - 5:
                        break
                    S = [1, s1, s2, s3]
                    if _check_no_4cycle(N, S):
                        if _check_no_6cycle(N, S):
                            if _check_no_8cycle(N, S):
                                return _build_circulant_edges(N, S)
        
        # Family 4: Try s0=1, s1 in [2,100], s2 in [s1+1,159], s3 in [s2+1,159]
        # but only if we have time remaining
        if time.time() < deadline - 30:
            for s1 in range(2, 100):
                if time.time() > deadline - 10:
                    break
                for s2 in range(s1 + 1, N):
                    if time.time() > deadline - 10:
                        break
                    for s3 in range(s2 + 1, N):
                        if time.time() > deadline - 5:
                            break
                        S = [1, s1, s2, s3]
                        if _check_no_4cycle(N, S):
                            if _check_no_6cycle(N, S):
                                if _check_no_8cycle(N, S):
                                    return _build_circulant_edges(N, S)
        
        # If no bipartite circulant found, try non-bipartite circulant on Z_320
        # Connection set S with S = -S, |S|=4, so S = {a, -a, b, -b} = {a, N-a, b, N-b}
        N2 = 320
        for a in range(1, N2 // 4):
            if time.time() > deadline - 5:
                break
            for b in range(a + 1, N2 // 2):
                if time.time() > deadline - 5:
                    break
                S2 = [a, N2 - a, b, N2 - b]
                if _check_girth_nonbipartite_circulant(N2, S2, g):
                    return _build_nonbipartite_circulant_edges(N2, S2)
        
        # Fallback: LUW construction
        return _construct_luw(k, g)
    
    return _construct_luw(k, g)

def _check_no_4cycle(N, S):
    """Check if bipartite circulant with connection set S has no 4-cycles."""
    # 4-cycle: A0 -> B_s1 -> A_{s1-s2} -> B_{s1-s2+s3} -> A0
    # Need s1-s2+s3-s4 = 0 mod N, i.e., s1+s3 = s2+s4
    # Non-backtracking: s1!=s2, s2!=s3, s3!=s4, s4!=s1
    # A-vertex distinct: s1-s2 != 0 (guaranteed by s1!=s2)
    # B-vertex distinct: s1 != s1-s2+s3, i.e., s2 != s3 (guaranteed)
    sums = {}
    for s1 in S:
        for s3 in S:
            key = (s1 + s3) % N
            if key not in sums:
                sums[key] = []
            sums[key].append((s1, s3))
    
    for key, pairs in sums.items():
        if len(pairs) > 1:
            for i in range(len(pairs)):
                for j in range(i + 1, len(pairs)):
                    s1, s3 = pairs[i]
                    s2, s4 = pairs[j]
                    if s1 != s2 and s2 != s3 and s3 != s4 and s4 != s1:
                        return False
    return True

def _check_no_6cycle(N, S):
    """Check if bipartite circulant with connection set S has no 6-cycles."""
    # 6-cycle: A0 -> B_s1 -> A_{s1-s2} -> B_{s1-s2+s3} -> A_{s1-s2+s3-s4} -> B_{s1-s2+s3-s4+s5} -> A0
    # Condition: s1+s3+s5 = s2+s4+s6 (mod N)
    # Non-backtracking: s1!=s2, s2!=s3, s3!=s4, s4!=s5, s5!=s6, s6!=s1
    # A distinct: a1=s1-s2, a2=s1-s2+s3-s4; need a1!=0, a2!=0, a1!=a2
    # B distinct: b1=s1, b2=s1-s2+s3, b3=s1-s2+s3-s4+s5; need all distinct
    sums3 = {}
    for s1 in S:
        for s3 in S:
            for s5 in S:
                key = (s1 + s3 + s5) % N
                if key not in sums3:
                    sums3[key] = []
                sums3[key].append((s1, s3, s5))
    
    for key, triples in sums3.items():
        if len(triples) > 1:
            for i in range(len(triples)):
                for j in range(i + 1, len(triples)):
                    s1, s3, s5 = triples[i]
                    s2, s4, s6 = triples[j]
                    if (s1 != s2 and s2 != s3 and s3 != s4 and s4 != s5 and
                        s5 != s6 and s6 != s1):
                        a1 = (s1 - s2) % N
                        a2 = (s1 - s2 + s3 - s4) % N
                        if a1 != 0 and a2 != 0 and a1 != a2:
                            b1 = s1
                            b2 = (s1 - s2 + s3) % N
                            b3 = (s1 - s2 + s3 - s4 + s5) % N
                            if b1 != b2 and b2 != b3 and b1 != b3:
                                return False
    return True

def _check_no_8cycle(N, S):
    """Check if bipartite circulant with connection set S has no 8-cycles."""
    # 8-cycle: A0 -> B_s1 -> A_{s1-s2} -> B_{s1-s2+s3} -> A_{s1-s2+s3-s4}
    #            -> B_{s1-s2+s3-s4+s5} -> A_{s1-s2+s3-s4+s5-s6}
    #            -> B_{s1-s2+s3-s4+s5-s6+s7} -> A0
    # Condition: s1+s3+s5+s7 = s2+s4+s6+s8 (mod N)
    # Non-backtracking: all consecutive pairs differ
    # A distinct: a1, a2, a3 all nonzero and pairwise distinct
    # B distinct: b1, b2, b3, b4 all pairwise distinct
    sums4 = {}
    for s1 in S:
        for s3 in S:
            for s5 in S:
                for s7 in S:
                    key = (s1 + s3 + s5 + s7) % N
                    if key not in sums4:
                        sums4[key] = []
                    sums4[key].append((s1, s3, s5, s7))
    
    for key, quads in sums4.items():
        if len(quads) > 1:
            for i in range(len(quads)):
                for j in range(i + 1, len(quads)):
                    s1, s3, s5, s7 = quads[i]
                    s2, s4, s6, s8 = quads[j]
                    if (s1 != s2 and s2 != s3 and s3 != s4 and s4 != s5 and
                        s5 != s6 and s6 != s7 and s7 != s8 and s8 != s1):
                        a1 = (s1 - s2) % N
                        a2 = (s1 - s2 + s3 - s4) % N
                        a3 = (s1 - s2 + s3 - s4 + s5 - s6) % N
                        if (a1 != 0 and a2 != 0 and a3 != 0 and
                            a1 != a2 and a2 != a3 and a1 != a3):
                            b1 = s1
                            b2 = (s1 - s2 + s3) % N
                            b3 = (s1 - s2 + s3 - s4 + s5) % N
                            b4 = (s1 - s2 + s3 - s4 + s5 - s6 + s7) % N
                            if (b1 != b2 and b1 != b3 and b1 != b4 and
                                b2 != b3 and b2 != b4 and b3 != b4):
                                return False
    return True

def _check_girth_nonbipartite_circulant(N, S, g):
    """Check girth of non-bipartite circulant graph on Z_N with connection set S."""
    # BFS from vertex 0 to find shortest cycle back to 0
    # But we need to check all vertices. For circulant, by symmetry, check from 0.
    # Actually for girth, we need the shortest cycle anywhere, but by transitivity of circulant,
    # the shortest cycle through 0 gives the girth.
    
    # BFS with non-backtracking constraint
    # State: (current_vertex, previous_vertex)
    # Start: (0, None) - but we need to handle the start specially
    
    # Actually, let's do BFS from (0, -1) where -1 means "no previous"
    # We track (vertex, prev) states
    from collections import deque
    visited = {}  # (vertex, prev) -> distance
    q = deque()
    q.append((0, -1))
    visited[(0, -1)] = 0
    
    while q:
        v, prev = q.popleft()
        dist = visited[(v, prev)]
        if dist >= g:
            continue
        for s in S:
            nv = (v + s) % N
            if nv == prev:
                continue  # non-backtracking
            if (nv, v) not in visited:
                visited[(nv, v)] = dist + 1
                if nv == 0 and dist + 1 >= 3:
                    # Found a cycle of length dist+1
                    if dist + 1 < g:
                        return False
                q.append((nv, v))
    
    return True

def _build_nonbipartite_circulant_edges(N, S):
    """Build edge list for non-bipartite circulant on Z_N."""
    edges = []
    for i in range(N):
        for s in S:
            j = (i + s) % N
            if i < j:
                edges.append((i, j))
    return N, edges

def _build_circulant_edges(N, S):
    """Build edge list for bipartite circulant with N vertices per part."""
    edges = []
    for i in range(N):
        for s in S:
            j = (i + s) % N
            a = i
            b = N + j
            if a < b:
                edges.append((a, b))
    return 2 * N, edges

def _construct_luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
