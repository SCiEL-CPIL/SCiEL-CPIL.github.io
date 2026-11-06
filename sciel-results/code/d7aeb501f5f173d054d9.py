import time
import random
import numpy as np
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin under 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _milp_decompose(edges, vertices, time_limit=30):
        """
        Formulate edge-disjoint path decomposition as a MILP.
        Each undirected edge {u,v} becomes two directed arcs u->v and v->u, each with capacity 1.
        We minimize the number of paths (number of path starts).
        """
        if not edges:
            return []

        # Build edge list and arc index mapping
        # Each undirected edge (u, v) with u < v creates arcs: (u,v) and (v,u)
        arcs = []  # list of (u, v) directed arcs
        arc_index = {}  # (u,v) -> index
        for u, v in edges:
            if u == v:
                continue
            arcs.append((u, v))
            arcs.append((v, u))
            arc_index[(u, v)] = len(arcs) - 2
            arc_index[(v, u)] = len(arcs) - 1

        m = len(arcs)  # number of arc variables
        n = len(vertices)
        v_list = sorted(vertices)
        v_index = {v: i for i, v in enumerate(v_list)}
        nv = len(v_list)

        # Binary variables: x_i for each arc i (1 if arc is used in some path)
        # Objective: minimize number of path starts
        # A vertex v is a path start if: outflow(v) - inflow(v) = 1
        # A vertex v is a path end if: inflow(v) - outflow(v) = 1
        # Number of paths = number of path starts = sum over v of max(0, outflow(v) - inflow(v))
        # To linearize: introduce s_v >= outflow(v) - inflow(v), s_v >= 0, minimize sum s_v
        # Actually simpler: number of paths = (number of odd-degree vertices)/2 if we pair them,
        # but in general for a path decomposition, #paths = #vertices with (out-in)=1
        
        # Let's define: for each vertex v, let d_v = outflow(v) - inflow(v)
        # d_v can be -1, 0, or 1 (since each arc has capacity 1 and we use at most one arc per direction per edge)
        # Actually with capacity 1 per directed arc, and each undirected edge contributing at most 2 arcs (one each way),
        # but we can't use both arcs of the same undirected edge in a simple path decomposition... 
        # Wait, we CAN use both arcs? No - a path is simple (no repeated vertices), so it can't traverse both u->v and v->u.
        # But in the flow formulation, we need to ensure we don't use both arcs of the same undirected edge.
        # Add constraint: for each undirected edge (u,v): x_{u,v} + x_{v,u} <= 1
        
        # Number of paths = number of vertices with d_v = 1 (path starts)
        # = sum_v max(0, d_v)
        # Since d_v in {-1, 0, 1}, max(0, d_v) = (d_v + |d_v|)/2
        # But |d_v| is tricky. Alternatively: since sum_v d_v = 0 (flow conservation globally),
        # #starts = #ends = (1/2) * sum_v |d_v|
        # And since d_v in {-1,0,1}, sum_v |d_v| = sum_v (d_v^2) but that's quadratic.
        
        # Simpler: introduce binary variables s_v indicating v is a path start.
        # Constraint: outflow(v) - inflow(v) = s_v - e_v where e_v is path end indicator
        # s_v + e_v <= 1 (can't be both start and end unless single vertex, but paths have >= 1 edge)
        # Minimize sum s_v
        
        # Variables:
        # x_i for i in range(m): binary, arc i used
        # s_v for v in range(nv): binary, vertex v is path start
        # e_v for v in range(nv): binary, vertex v is path end
        
        total_vars = m + 2 * nv
        # x: 0..m-1
        # s: m..m+nv-1
        # e: m+nv..m+2*nv-1
        
        def x_idx(i):
            return i
        
        def s_idx(v):
            return m + v_index[v]
        
        def e_idx(v):
            return m + nv + v_index[v]
        
        # Objective: minimize sum of s_v
        c = np.zeros(total_vars)
        for v in v_list:
            c[s_idx(v)] = 1.0
        
        # Constraints:
        # 1. Flow conservation: for each vertex v:
        #    sum_{arcs (v,w)} x_{v,w} - sum_{arcs (w,v)} x_{w,v} = s_v - e_v
        #    => sum_{arcs (v,w)} x_{v,w} - sum_{arcs (w,v)} x_{w,v} - s_v + e_v = 0
        
        # 2. For each undirected edge (u,v): x_{u,v} + x_{v,u} <= 1
        #    (can't use both directions of same edge)
        
        # 3. Bounds: all variables in [0, 1], integer
        
        # Build constraint matrix
        # We'll use LinearConstraint with lb and ub arrays
        
        # Constraint 1: flow conservation (equality)
        # For each vertex v: sum_out - sum_in - s_v + e_v = 0
        A_rows = []
        lb = []
        ub = []
        
        for v in v_list:
            row = np.zeros(total_vars)
            vi = v_index[v]
            for i, (a, b) in enumerate(arcs):
                if a == v:
                    row[x_idx(i)] = 1.0  # outgoing
                elif b == v:
                    row[x_idx(i)] = -1.0  # incoming
            row[s_idx(v)] = -1.0
            row[e_idx(v)] = 1.0
            A_rows.append(row)
            lb.append(0.0)
            ub.append(0.0)
        
        # Constraint 2: for each undirected edge, x_uv + x_vu <= 1
        for u, v in edges:
            row = np.zeros(total_vars)
            idx_uv = arc_index[(u, v)]
            idx_vu = arc_index[(v, u)]
            row[x_idx(idx_uv)] = 1.0
            row[x_idx(idx_vu)] = 1.0
            A_rows.append(row)
            lb.append(-np.inf)
            ub.append(1.0)
        
        if A_rows:
            A = np.array(A_rows)
            lb_arr = np.array(lb)
            ub_arr = np.array(ub)
        else:
            A = None
            lb_arr = None
            ub_arr = None
        
        # Variable bounds and integrality
        integrality = np.ones(total_vars)  # all binary
        lbounds = np.zeros(total_vars)
        ubounds = np.ones(total_vars)
        
        # Solve MILP
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A, lb_arr, ub_arr) if A is not None else None,
                integrality=integrality,
                bounds=Bounds(lbounds, ubounds),
            )
            
            if res.status == 0 and res.x is not None:
                # Extract solution
                x_sol = res.x[:m]
                
                # Reconstruct paths
                # Build adjacency from selected arcs
                adj = defaultdict(list)
                used_arcs = set()
                for i, (a, b) in enumerate(arcs):
                    if x_sol[i] > 0.5:
                        adj[a].append(b)
                        used_arcs.add((a, b))
                
                # Find path starts: vertices where s_v = 1
                s_sol = res.x[m:m+nv]
                path_starts = []
                for v in v_list:
                    if s_sol[v_index[v]] > 0.5:
                        path_starts.append(v)
                
                # Trace paths from each start
                paths = []
                for start in path_starts:
                    if start not in adj:
                        continue
                    path = [start]
                    current = start
                    visited = {start}
                    while current in adj and adj[current]:
                        # Find next vertex
                        next_v = None
                        for nb in adj[current]:
                            if nb not in visited:
                                next_v = nb
                                break
                        if next_v is None:
                            break
                        path.append(next_v)
                        visited.add(next_v)
                        current = next_v
                        # Remove the arc to avoid reusing
                        if current in adj and next_v in adj[current]:
                            adj[current].remove(next_v)
                    if len(path) > 1:
                        paths.append(path)
                
                # Verify coverage
                if len(paths) > 0:
                    # Check all edges are covered
                    covered = set()
                    for path in paths:
                        for i in range(len(path) - 1):
                            covered.add(tuple(sorted((path[i], path[i+1]))))
                    if covered == set(edges):
                        return paths
                
                # Fallback: try to reconstruct differently
                # Use the flow to trace paths more carefully
                # Build directed graph from used arcs
                out_adj = defaultdict(list)
                for i, (a, b) in enumerate(arcs):
                    if x_sol[i] > 0.5:
                        out_adj[a].append(b)
                
                # Find starts
                starts = []
                for v in v_list:
                    vi = v_index[v]
                    if s_sol[vi] > 0.5:
                        starts.append(v)
                
                paths = []
                for start in starts:
                    path = [start]
                    current = start
                    used_arcs_local = set()
                    while True:
                        found = False
                        for nb in out_adj.get(current, []):
                            arc = (current, nb)
                            if arc not in used_arcs_local:
                                path.append(nb)
                                used_arcs_local.add(arc)
                                current = nb
                                found = True
                                break
                        if not found:
                            break
                    if len(path) > 1:
                        paths.append(path)
                
                # Verify
                covered = set()
                for path in paths:
                    for i in range(len(path) - 1):
                        covered.add(tuple(sorted((path[i], path[i+1]))))
                if covered == set(edges):
                    return paths
                
        except Exception:
            pass
        
        return None

    def _greedy_decompose(edges):
        """Greedy path decomposition as fallback."""
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining = set(edges)
        paths = []
        
        while remaining:
            # Find a vertex with max remaining degree
            degs = defaultdict(int)
            for u, v in remaining:
                degs[u] += 1
                degs[v] += 1
            
            if not degs:
                break
            
            start = max(degs, key=lambda x: (degs[x], -x))
            
            path = [start]
            current = start
            visited = {start}
            
            while True:
                avail = [nb for nb in adj[current] 
                         if tuple(sorted((current, nb))) in remaining 
                         and nb not in visited]
                if not avail:
                    break
                # Pick neighbor with max remaining degree
                avail.sort(key=lambda x: -degs.get(x, 0))
                next_v = avail[0]
                edge = tuple(sorted((current, next_v)))
                remaining.discard(edge)
                path.append(next_v)
                visited.add(next_v)
                current = next_v
            
            paths.append(path)
        
        return paths

    def _lower_bound(edges):
        """Lower bound on number of paths = max(1, num_odd/2) if edges exist."""
        if not edges:
            return 0
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd = sum(1 for d in deg.values() if d % 2 == 1)
        return max(1, odd // 2)

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for i, spec in enumerate(specs):
        if time.time() > deadline - 5:
            edges = instance(*spec)
            paths = _greedy_decompose(edges)
            results.append(paths)
            continue
        
        edges = instance(*spec)
        vertices = set()
        for u, v in edges:
            vertices.add(u)
            vertices.add(v)
        
        # Try MILP
        lb = _lower_bound(edges)
        paths = None
        
        # Give more time to larger instances
        time_left = deadline - time.time() - 5
        milp_time = min(35, max(10, time_left / 3))
        
        paths = _milp_decompose(edges, vertices, time_limit=milp_time)
        
        if paths is None or len(paths) > lb + 3:
            # Fallback to greedy
            greedy_paths = _greedy_decompose(edges)
            if paths is None or len(greedy_paths) < len(paths):
                paths = greedy_paths
        
        results.append(paths)
    
    return results

if __name__ == "__main__":
    res = construct()
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
    print(f"Total: {sum(len(r) for r in res)} paths")
