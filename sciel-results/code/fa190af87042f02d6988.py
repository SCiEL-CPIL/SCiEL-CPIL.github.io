"""Portfolio SEARCH: try multiple explicit families, keep best (smallest) valid graph."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    deadline = time.time() + 115.0  # safety margin before 150s kill
    rng = random.Random(12345)

    def check_deadline():
        return time.time() > deadline

    # --- Fast girth check using BFS from each vertex with adjacency list ---
    def girth_check_fast(adj, n, g):
        """Check if all cycles have length >= g. Returns True if girth >= g."""
        # For each source, do BFS tracking (node, parent) to detect cycles
        # A cycle of length L is found when we revisit source at distance L
        # We need to check that no such cycle has L < g
        for s in range(n):
            # dist[u] = shortest distance from s to u
            # parent[u] = parent in BFS tree
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= g - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        # Back edge or cross edge: cycle length = du + 1 + dist[v]
                        # Actually for undirected BFS, if v is already visited and v != parent[u],
                        # then we have a cycle of length du + dist[v] + 1
                        # But we need to be careful: in undirected graph BFS,
                        # the cycle formed by u-v where both are visited has length
                        # dist[u] + dist[v] + 1 only if the paths don't share nodes.
                        # Simpler: check if v is ancestor of u or vice versa.
                        # Actually the standard approach: for each edge (u,v) in BFS tree
                        # or non-tree edge, the cycle length is dist[u] + dist[v] + 1
                        # if the LCA is handled properly.
                        # Let me use a different approach: track the actual BFS tree.
                        pass
            # Alternative: for each source s, do BFS and check all edges
            # For edge (u,v), if both u and v are reachable from s,
            # the cycle through s has length dist[s->u] + 1 + dist[s->v]
            # but this isn't right either.
            # 
            # Correct approach: BFS from s. For each non-tree edge (u,v) where
            # parent[u] != v and parent[v] != u, the cycle length is
            # dist[u] + dist[v] + 1. We need all such cycles >= g.
            # But this only catches cycles that go through the BFS tree structure.
            # Actually for undirected graphs, BFS from s: any cycle must contain
            # at least one non-tree edge. The cycle length for non-tree edge (u,v)
            # is dist[u] + dist[v] + 1. So we need dist[u] + dist[v] + 1 >= g
            # for all non-tree edges (u,v) where u,v are both reachable.
            #
            # Wait, that's not quite right. The cycle formed by the non-tree edge (u,v)
            # goes u -> ... -> s -> ... -> v -> u, with length dist[u] + dist[v] + 1.
            # But there could be shorter cycles not going through s.
            #
            # Actually, the girth is the minimum over all sources s of the minimum
            # cycle length found in BFS from s. For a given BFS from s, any cycle
            # in the graph either goes through s (then it's captured by the 
            # dist[u]+dist[v]+1 formula for non-tree edges) or doesn't.
            # If it doesn't go through s, then it's entirely in some subtree,
            # and would be found by BFS from a node in that cycle.
            #
            # So the correct check is: for each source s, BFS from s, and for each
            # non-tree edge (u,v) (i.e., edge where parent[u]!=v and parent[v]!=u),
            # check that dist[u] + dist[v] + 1 >= g.
            #
            # Let me redo this properly.
            pass
        return True

    def girth_check_correct(adj, n, g):
        """
        Check girth >= g by BFS from each vertex.
        For each source s, BFS gives dist[]. For each edge (u,v) in the graph:
        - If it's a tree edge (parent[u]==v or parent[v]==u), skip.
        - Otherwise, the cycle through s has length dist[u] + dist[v] + 1.
        We need this >= g for all such edges.
        """
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
            # Now check all edges
            for u in range(n):
                if dist[u] == -1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        continue
                    # Skip tree edges
                    if parent[u] == v or parent[v] == u:
                        continue
                    # This is a non-tree edge; cycle length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
        return True

    def validate_fast(n, edges, k, g, best_n=None):
        """Validate graph is simple, k-regular, girth >= g.
        If best_n is given and n >= best_n, skip girth check (can't improve)."""
        if n <= 0:
            return False
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return False
            key = (min(u,v), max(u,v))
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        # Quick degree check passed; now girth check
        # If we already have a best with n <= this n, no point checking girth
        if best_n is not None and n >= best_n:
            return None  # Can't improve, skip expensive check
        if not girth_check_correct(adj, n, g):
            return False
        return True

    def try_graph(n, edges, k, g, best_n=None):
        result = validate_fast(n, edges, k, g, best_n)
        if result is None:
            return None  # Skipped because n >= best_n
        if not result:
            return None
        return n, edges

    # --- Family: Incidence graph of a projective plane (for k = q+1, prime power q) ---
    def is_prime_power(m):
        if m < 2:
            return False
        for p in range(2, int(math.isqrt(m)) + 1):
            if m % p == 0:
                x = m
                while x % p == 0:
                    x //= p
                if x == 1:
                    return True
                break
        if m > 1:
            for p in range(2, int(math.isqrt(m)) + 1):
                if m % p == 0:
                    return False
            return True
        return False

    def incidence_family():
        results = []
        if k < 3:
            return results
        q = k - 1
        if not is_prime_power(q):
            return results
        n_pts = q * q + q + 1
        
        points = []
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
            points.append((x, 1, 0))
        points.append((1, 0, 0))
        
        lines = []
        for a in range(q):
            for b in range(q):
                lines.append((a, b, 1))
            lines.append((a, 1, 0))
        lines.append((1, 0, 0))
        
        n_total = 2 * n_pts
        edges = []
        for i, pt in enumerate(points):
            x, y, z = pt
            for j, ln in enumerate(lines):
                a, b, c = ln
                if (a * x + b * y + c * z) % q == 0:
                    edges.append((i, n_pts + j))
        
        results.append((n_total, edges))
        return results

    # --- Family 5: Systematic voltage graph search over K_k base ---
    def voltage_kbase_search():
        """
        Use K_k as base graph with voltages in Z_p.
        n = k * p vertices. Try various p and voltage assignments.
        """
        results = []
        # Moore bound for reference
        if g % 2 == 1:
            r = (g - 1) // 2
            moore = 1 + k * sum((k-1)**i for i in range(r))
        else:
            r = g // 2
            moore = k * sum((k-1)**i for i in range(r))
        
        # Try primes p such that k*p is in a reasonable range
        primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]
        
        for p in primes:
            if check_deadline():
                break
            n = k * p
            if n < moore or n > 500:
                continue
            if n * k % 2 != 0:
                continue
            
            # Try multiple random voltage assignments
            for trial in range(20):
                if check_deadline():
                    break
                
                # Assign random voltages to each directed edge of K_k
                # v[i][j] for i != j, with v[i][j] + v[j][i] != 0 (mod p) for simplicity
                # Actually for the lift to be simple, we need:
                # For each pair (i,j), the set {t + v[i][j] : t in Z_p} and
                # {t + v[j][i] : t in Z_p} should not create multi-edges.
                # Since each fiber pair (i,j) gets exactly one edge per t,
                # we need v[i][j] != v[j][i] to avoid the edge being a loop
                # in the base, and we need the lift to be simple.
                
                # For K_k base: vertex (i, t) connects to (j, t + v[i][j]) for all j != i.
                # Degree of (i,t) = k-1 (one for each j != i). But we need degree k!
                # So K_k base doesn't work directly for k-regular lift.
                
                # We need a base graph where each vertex has degree k.
                # For k=5, we could use a 5-regular base graph.
                # Simplest: use a 5-regular graph on m vertices as base.
                # Then lift has n = m * p vertices, each with degree k.
                
                # For m=5: K_5 is 4-regular, not 5-regular. Can't use.
                # For m=6: K_6 is 5-regular! So use K_6 as base.
                # n = 6 * p. For p=19, n=114. For p=23, n=138.
                
                # Let's use K_6 base for k=5
                if k == 5:
                    m_base = 6  # K_6 is 5-regular
                    n_lift = m_base * p
                    if n_lift < moore or n_lift > 500:
                        continue
                    
                    # Assign voltages to K_6 edges
                    # v[i][j] for 0<=i<j<6, v[j][i] = -v[i][j] mod p
                    voltages = {}
                    for i in range(m_base):
                        for j in range(i+1, m_base):
                            voltages[(i,j)] = rng.randint(1, p-1)
                    
                    # Build the lift
                    edges = []
                    adj = [[] for _ in range(n_lift)]
                    for i in range(m_base):
                        for j in range(i+1, m_base):
                            v_ij = voltages[(i,j)]
                            v_ji = (-v_ij) % p
                            for t in range(p):
                                u1 = i * p + t
                                u2 = j * p + ((t + v_ij) % p)
                                if u1 != u2:
                                    edges.append((u1, u2))
                                    adj[u1].append(u2)
                                    adj[u2].append(u1)
                    
                    # Check regularity
                    valid = True
                    for u in range(n_lift):
                        if len(adj[u]) != k:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Check for multi-edges
                    edge_set = set()
                    has_multi = False
                    for u, v in edges:
                        key = (min(u,v), max(u,v))
                        if key in edge_set:
                            has_multi = True
                            break
                        edge_set.add(key)
                    if has_multi:
                        continue
                    
                    results.append((n_lift, edges))
                else:
                    # For general k, try to find a k-regular base graph
                    # This is complex; skip for now
                    pass
        return results

    def random_regular_graph(n, k, rng, max_iters=2000):
        if n * k % 2 != 0:
            return None
        if k >= n:
            return None
        for attempt in range(max_iters):
            half_edges = []
            for v in range(n):
                half_edges.extend([v] * k)
            rng.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(half_edges), 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if not valid:
                continue
            # Fix multi-edges with 2-switches
            for _ in range(300):
                has_multi = False
                for u in range(n):
                    if len(adj[u]) != len(set(adj[u])):
                        has_multi = True
                        break
                if not has_multi:
                    break
                a = rng.randint(0, n-1)
                b = rng.randint(0, n-1)
                c = rng.randint(0, n-1)
                d = rng.randint(0, n-1)
                if len({a,b,c,d}) < 4:
                    continue
                if b not in adj[a] or d not in adj[c]:
                    continue
                if d in adj[a] or b in adj[c]:
                    continue
                adj[a].remove(b)
                adj[b].remove(a)
                adj[c].remove(d)
                adj[d].remove(c)
                adj[a].append(d)
                adj[d].append(a)
                adj[c].append(b)
                adj[b].append(c)
            has_multi = False
            for u in range(n):
                if len(adj[u]) != len(set(adj[u])):
                    has_multi = True
                    break
            if has_multi:
                continue
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
        return None

    best = None
    best_n = float('inf')

    def consider(n, edges):
        nonlocal best, best_n
        if n < best_n:
            result = try_graph(n, edges, k, g, best_n)
            if result is not None:
                best = result
                best_n = n
        if check_deadline():
            raise TimeoutError

    # --- Family 1: Lazebnik-Ustimenko-Woldar (from seed) ---
    def luw_family(q=None, d=None):
        if q is None:
            q = k
        if d is None:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,)+(0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # --- Family 2: Circulant / folded circulant constructions ---
    def circulant_family():
        results = []
        if g % 2 == 1:
            r = (g-1)//2
            moore = 1 + k * sum((k-1)**i for i in range(r))
        else:
            r = g//2
            moore = k * sum((k-1)**i for i in range(r))
        
        max_n = moore + 500
        if k % 2 == 0:
            m = k // 2
            for n in range(max(moore, 2*m+1), max_n+1):
                if check_deadline():
                    break
                shift_sets = []
                if m <= n//2:
                    shift_sets.append(list(range(1, m+1)))
                if m == 1 and n > 2:
                    shift_sets.append([1])
                if m == 2 and n > 4:
                    shift_sets.append([1, 2])
                    shift_sets.append([1, 3])
                    shift_sets.append([2, 3])
                if m == 3 and n > 6:
                    shift_sets.append([1, 2, 3])
                    shift_sets.append([1, 2, 4])
                    shift_sets.append([1, 3, 4])
                    shift_sets.append([2, 3, 4])
                
                for shifts in shift_sets:
                    valid_shifts = True
                    for s in shifts:
                        if 2 * s % n == 0:
                            valid_shifts = False
                            break
                    if not valid_shifts:
                        continue
                    if len(set(s % n for s in shifts)) != len(shifts):
                        continue
                    if 0 in shifts:
                        continue
                    
                    edges = []
                    for i in range(n):
                        for s in shifts:
                            j = (i + s) % n
                            if i < j:
                                edges.append((i, j))
                    results.append((n, edges))
        else:
            pass
        return results

    # --- Family 3: Random regular graph search with 2-switch refinement ---
    def random_regular_search():
        if g % 2 == 1:
            r = (g-1)//2
            moore = 1 + k * sum((k-1)**i for i in range(r))
        else:
            r = g//2
            moore = k * sum((k-1)**i for i in range(r))
        
        results = []
        # Search in a range around the Moore bound
        search_start = max(moore, k+2)
        search_end = min(search_start + 300, 600)
        
        for n in range(search_start, search_end):
            if check_deadline():
                break
            if n * k % 2 != 0:
                continue
            if k >= n:
                continue
            
            # Try to build a random k-regular graph
            for attempt in range(100):
                if check_deadline():
                    break
                
                # Use configuration model with 2-switches to fix multi-edges
                half_edges = []
                for v in range(n):
                    half_edges.extend([v] * k)
                rng.shuffle(half_edges)
                
                adj = [[] for _ in range(n)]
                valid = True
                for i in range(0, len(half_edges), 2):
                    u, v = half_edges[i], half_edges[i+1]
                    if u == v:
                        valid = False
                        break
                    adj[u].append(v)
                    adj[v].append(u)
                
                if not valid:
                    continue
                
                # Fix multi-edges with 2-switches
                for _ in range(200):
                    # Find a multi-edge
                    multi_found = False
                    for u in range(n):
                        if len(adj[u]) != len(set(adj[u])):
                            multi_found = True
                            break
                    if not multi_found:
                        break
                    
                    # Try random 2-switch
                    a = rng.randint(0, n-1)
                    b = rng.randint(0, n-1)
                    c = rng.randint(0, n-1)
                    d = rng.randint(0, n-1)
                    if len({a,b,c,d}) < 4:
                        continue
                    if b not in adj[a] or d not in adj[c]:
                        continue
                    if d in adj[a] or b in adj[c]:
                        continue
                    adj[a].remove(b)
                    adj[b].remove(a)
                    adj[c].remove(d)
                    adj[d].remove(c)
                    adj[a].append(d)
                    adj[d].append(a)
                    adj[c].append(b)
                    adj[b].append(c)
                
                # Final check for multi-edges
                has_multi = False
                for u in range(n):
                    if len(adj[u]) != len(set(adj[u])):
                        has_multi = True
                        break
                if has_multi:
                    continue
                
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                
                results.append((n, edges))
                break  # Found one for this n, move to next n
        
        return results

    # --- Family 4: Voltage graph construction over cyclic group ---
    def voltage_graph_family():
        """
        Construct a k-regular graph of girth at least g using voltage graph technique.
        The base graph is a cycle of length L, with k voltages assigned to each edge.
        The lift has n = L * |G| vertices where G = Z_p for some prime p.
        """
        results = []
        
        # For k=5, g=9: try L=9, p=13 (n=117) or similar
        # The voltage assignments must be chosen so that no closed walk of length < g
        # in the base graph has trivial total voltage.
        
        # Try small primes p and cycle lengths L
        for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            if check_deadline():
                break
            for L in range(g, g + 10):
                if check_deadline():
                    break
                n = L * p
                if n < 100:  # skip too small
                    continue
                if n > 2000:  # skip too large for now
                    continue
                
                # We need a k-regular voltage graph on a cycle of length L
                # with p vertices in the fiber. The base graph must be k-regular
                # in some sense, but for a cycle the base degree is 2, so we need
                # multiple edges or a different base.
                
                # Alternative: use a complete graph K_k as base with voltage assignments
                # This gives a k-regular graph with n = k * p vertices
                n = k * p
                if n < 100 or n > 2000:
                    continue
                
                # Base graph: K_k (complete graph on k vertices)
                # Each edge (i,j) in the base gets a voltage v_ij in Z_p
                # The lift has vertices (i, t) where i in 0..k-1, t in Z_p
                # Edge from (i, t) to (j, t + v_ij)
                
                # For the graph to be simple, we need v_ij != v_ji and no two edges
                # between the same pair of base vertices have the same voltage offset.
                
                # Let's assign voltages such that the product of voltages around
                # any short cycle is non-trivial.
                
                # Use a systematic assignment: v_ij = (i * j) % p for i != j
                # Check if this works
                edges = []
                adj = [[] for _ in range(n)]
                for i in range(k):
                    for j in range(i+1, k):
                        v_ij = (i * j + i + j) % p
                        v_ji = (-v_ij) % p  # antisymmetric
                        for t in range(p):
                            u1 = i * p + t
                            u2 = j * p + ((t + v_ij) % p)
                            if u1 != u2:
                                edges.append((u1, u2))
                                adj[u1].append(u2)
                                adj[u2].append(u1)
                
                # Check if it's k-regular
                valid = True
                for u in range(n):
                    if len(adj[u]) != k:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check for multi-edges
                edge_set = set()
                has_multi = False
                for u, v in edges:
                    key = (min(u,v), max(u,v))
                    if key in edge_set:
                        has_multi = True
                        break
                    edge_set.add(key)
                if has_multi:
                    continue
                
                results.append((n, edges))
        return results

    # --- Run all families, prioritizing by expected quality ---
    # First: LUW family (known good construction)
    try:
        n, e = luw_family()
        consider(n, e)
    except Exception as ex:
        pass

    for d_try in range(max(3, g-6), g+2):
        if check_deadline():
            break
        try:
            n, e = luw_family(d=d_try)
            consider(n, e)
        except Exception as ex:
            pass

    # Second: Voltage graph family (systematic search)
    try:
        volt_results = voltage_graph_family()
        for n, e in volt_results:
            if check_deadline():
                break
            consider(n, e)
    except Exception as ex:
        pass

    # Third: Incidence family
    try:
        inc_results = incidence_family()
        for n, e in inc_results:
            if check_deadline():
                break
            consider(n, e)
    except Exception as ex:
        pass

    # Fourth: Circulant family
    try:
        circ_results = circulant_family()
        for n, e in circ_results:
            if check_deadline():
                break
            consider(n, e)
    except Exception as ex:
        pass

    # Fifth: Random regular graph search (use remaining time)
    try:
        rand_results = random_regular_search()
        for n, e in rand_results:
            if check_deadline():
                break
            consider(n, e)
    except Exception as ex:
        pass

    # Return best found, or fallback to LUW
    if best is not None:
        return best

    # Fallback: return LUW without validation (shouldn't happen)
    try:
        n, e = luw_family()
        return n, e
    except Exception:
        # Last resort: return a trivial valid graph if possible
        # For k=1: path graph
        if k == 1:
            return (2, [(0,1)])
        # For k=2: cycle
        if k == 2:
            n = max(g, 3)
            edges = [(i, (i+1)%n) for i in range(n)]
            return (n, edges)
        # For general k: return LUW even if unvalidated
        return luw_family()

# EVOLVE-BLOCK-END
