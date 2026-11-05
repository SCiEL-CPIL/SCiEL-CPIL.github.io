import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def is_c4_free(n, adj):
        # adj is list of sets
        for i in range(n):
            for j in range(i + 1, n):
                if len(adj[i] & adj[j]) > 1:
                    return False
        return True

    def find_c4(n, adj):
        # Returns (u, v, a, b) such that u-a-v-b-u is a C4
        # i.e., a, b are common neighbors of u, v
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if len(common) >= 2:
                    common_list = list(common)
                    a = common_list[0]
                    b = common_list[1]
                    return (i, j, a, b)
        return None

    def would_create_c4(n, adj, u, v):
        # Check if adding edge (u,v) creates a C4
        # C4 exists if there is a path u-a-b-v of length 3
        # a in N(u), b in N(v), a-b in E
        # Also need a != b, a != v, b != u
        for a in adj[u]:
            if a == v:
                continue
            for b in adj[v]:
                if b == u or b == a:
                    continue
                if b in adj[a]:
                    return True
        return False

    def count_c4s_involving_edge(n, adj, u, v):
        """Count how many C4s contain edge (u,v)."""
        if v not in adj[u]:
            return 0
        count = 0
        for w in adj[u]:
            if w == v:
                continue
            for x in adj[v]:
                if x == u or x == w:
                    continue
                if x in adj[w]:
                    count += 1
        return count

    def build_projective_plane_incidence(q):
        """
        Build the incidence graph of the projective plane of order q.
        This is a C4-free bipartite graph with n = q^2 + q + 1 points + q^2 + q + 1 lines = 2(q^2+q+1) vertices.
        Each point is on q+1 lines, each line has q+1 points.
        For q=4: n=42, min degree=5.
        """
        if q < 2:
            return None
            
        n_points = q * q + q + 1
        n_lines = q * q + q + 1
        total_n = n_points + n_lines
        
        # Generate projective plane of order q
        # Points: (x, y, 1) for x,y in GF(q), plus (x, 1, 0) for x in GF(q), plus (1, 0, 0)
        # Lines: ax + by + cz = 0 for (a,b,c) not all zero, up to scalar
        
        # For small q, we can enumerate
        points = []
        # Affine points: (x, y, 1)
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        # Points at infinity: (x, 1, 0)
        for x in range(q):
            points.append((x, 1, 0))
        # Point at infinity: (1, 0, 0)
        points.append((1, 0, 0))
        
        # Lines: ax + by + cz = 0
        # Normalize: if c != 0, set c=1: ax + by = -z
        # If c == 0 and b != 0, set b=1: ax + y = 0
        # If c == 0 and b == 0, set a=1: x = 0
        lines = []
        # Lines with c=1: ax + by = -z, for a,b in GF(q)
        for a in range(q):
            for b in range(q):
                lines.append((a, b, 1))
        # Lines with c=0, b=1: ax + y = 0, for a in GF(q)
        for a in range(q):
            lines.append((a, 1, 0))
        # Line with c=0, b=0, a=1: x = 0
        lines.append((1, 0, 0))
        
        # Build incidence: point i is on line j if a*x + b*y + c*z = 0
        adj = [set() for _ in range(total_n)]
        edges = []
        
        for i, p in enumerate(points):
            x, y, z = p
            for j, l in enumerate(lines):
                a, b, c = l
                if (a * x + b * y + c * z) % q == 0:
                    adj[i].add(j + n_points)
                    adj[j + n_points].add(i)
                    edges.append((i, j + n_points))
        
        return {'n': total_n, 'adj': adj, 'edges': edges}

    def try_construct_from_seed(n, target_deg, seed_adj, seed_edges, max_time=10.0):
        """Try to extend a seed graph to meet target degree while staying C4-free."""
        t_start = time.time()
        
        # Copy the seed
        adj = [set(s) for s in seed_adj[:n]]
        deg = [len(adj[i]) for i in range(n)]
        edges = list(seed_edges)
        
        if min(deg) >= target_deg and is_c4_free(n, adj):
            return {'n': n, 'edges': [[u, v] for u, v in edges]}
        
        # Try to add edges to increase minimum degree
        all_possible = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in adj[u]]
        random.shuffle(all_possible)
        
        # Priority: add edges incident to low-degree vertices
        for u, v in all_possible:
            if time.time() - t_start > max_time:
                break
            if min(deg) >= target_deg:
                break
            # Only add if it helps a low-degree vertex
            if deg[u] < target_deg or deg[v] < target_deg:
                if not would_create_c4(n, adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                    edges.append((u, v))
        
        if min(deg) >= target_deg and is_c4_free(n, adj):
            return {'n': n, 'edges': [[u, v] for u, v in edges]}
        
        return None

    def try_construct(n, target_deg, max_time=10.0):
        t_start = time.time()
        
        # Strategy 1: Use projective plane seed if applicable
        # For n around 42, use q=4 projective plane
        # For other n, use random initialization
        
        best_result = None
        
        # Try projective plane seeds
        for q in [3, 4, 5, 6]:
            if time.time() - t_start > max_time:
                break
            pp = build_projective_plane_incidence(q)
            if pp is None:
                continue
            pp_n = pp['n']
            
            # If pp_n <= n, we can try to extend it
            if pp_n <= n:
                # Use the projective plane as seed and try to add vertices/edges
                # For simplicity, if pp_n == n, just use it directly
                if pp_n == n:
                    if min([len(pp['adj'][i]) for i in range(n)]) >= target_deg:
                        if is_c4_free(n, pp['adj']):
                            return {'n': n, 'edges': [[u, v] for u, v in pp['edges']]}
                
                # Try to extend by adding vertices
                # This is complex, so for now just check if pp_n == n works
                
        # Strategy 2: Random initialization with local search
        start_deg = target_deg + 4
        
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        
        # Generate random edges until min degree is start_deg
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if min(deg) >= start_deg:
                break
            if v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges.append((u, v))
        
        if min(deg) < start_deg:
            pass

        # Local search: remove edges to break C4s, then try to add back
        iterations = 0
        stuck_count = 0
        while time.time() - t_start < max_time:
            c4 = find_c4(n, adj)
            if c4 is None:
                if min(deg) >= target_deg:
                    return {'n': n, 'edges': [[u, v] for u, v in edges]}
                else:
                    # Need to add edges
                    added = False
                    # Try to add edges that help low-degree vertices
                    candidates = [(u, v) for u, v in all_edges if v not in adj[u]]
                    random.shuffle(candidates)
                    for u, v in candidates:
                        if min(deg) >= target_deg:
                            break
                        # Prioritize edges incident to low-degree vertices
                        if deg[u] < target_deg or deg[v] < target_deg:
                            if not would_create_c4(n, adj, u, v):
                                adj[u].add(v)
                                adj[v].add(u)
                                deg[u] += 1
                                deg[v] += 1
                                edges.append((u, v))
                                added = True
                                if min(deg) >= target_deg:
                                    if is_c4_free(n, adj):
                                        return {'n': n, 'edges': [[u, v] for u, v in edges]}
                                break
                    if not added:
                        stuck_count += 1
                        if stuck_count > 50:
                            break
                    else:
                        stuck_count = 0
            else:
                u, v, a, b = c4
                # Remove the edge that's in the most C4s (greedy)
                candidates = [(u, a), (a, v), (v, b), (b, u)]
                # Count C4s for each candidate edge
                edge_c4_counts = []
                for ru, rv in candidates:
                    if rv in adj[ru]:
                        c4_count = count_c4s_involving_edge(n, adj, ru, rv)
                        edge_c4_counts.append((c4_count, ru, rv))
                
                if edge_c4_counts:
                    # Remove the edge with the most C4s
                    edge_c4_counts.sort(reverse=True)
                    _, ru, rv = edge_c4_counts[0]
                    if rv in adj[ru]:
                        adj[ru].remove(rv)
                        adj[rv].remove(ru)
                        deg[ru] -= 1
                        deg[rv] -= 1
                        try:
                            edges.remove((ru, rv))
                        except ValueError:
                            try:
                                edges.remove((rv, ru))
                            except ValueError:
                                pass
                else:
                    # Fallback: remove random edge
                    random.shuffle(candidates)
                    for ru, rv in candidates:
                        if rv in adj[ru]:
                            adj[ru].remove(rv)
                            adj[rv].remove(ru)
                            deg[ru] -= 1
                            deg[rv] -= 1
                            try:
                                edges.remove((ru, rv))
                            except ValueError:
                                try:
                                    edges.remove((rv, ru))
                                except ValueError:
                                    pass
                            break
            
            iterations += 1
            if iterations > 20000:
                break
        
        return None

    best_result = None
    
    # Try n=47 first, then 46, etc.
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline:
            break
            
        target_deg = n - 39
        
        # Multiple attempts
        for attempt in range(15):
            if time.time() > deadline:
                break
            # Vary the random seed for diversity
            res = try_construct(n, target_deg, max_time=4.0)
            if res is not None:
                # Verify
                adj = [set() for _ in range(n)]
                for u, v in res['edges']:
                    adj[u].add(v)
                    adj[v].add(u)
                if is_c4_free(n, adj):
                    degs = [len(adj[i]) for i in range(n)]
                    if min(degs) >= target_deg:
                        best_result = res
                        break
        
        if best_result is not None:
            break

    if best_result is None:
        # Fallback: use projective plane of order 4 (n=42, min degree 5 >= 42-39=3)
        pp = build_projective_plane_incidence(4)
        if pp is not None:
            best_result = {'n': pp['n'], 'edges': [[u, v] for u, v in pp['edges']]}
        else:
            # Ultimate fallback: n=39 empty graph
            best_result = {'n': 39, 'edges': []}
        
    return best_result
