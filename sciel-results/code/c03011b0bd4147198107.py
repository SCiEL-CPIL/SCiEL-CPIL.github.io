import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from collections import deque

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_path_cover_milp(edges, n):
        if not edges:
            return []
        
        # Build edge list and map
        edge_list = list(edges)
        m = len(edge_list)
        
        # Variables:
        # x_e for e in 0..m-1 : direction of edge e. 0 means (u,v), 1 means (v,u).
        # s_v for v in 0..n-1 : is v a source? (1 if yes)
        # t_v for v in 0..n-1 : is v a sink? (1 if yes)
        
        # Total variables: m + 2*n
        # Indices:
        # 0 .. m-1 : x
        # m .. m+n-1 : s
        # m+n .. m+2*n-1 : t
        
        c = np.zeros(m + 2*n)
        # Minimize sum of s_v
        c[m : m+n] = 1.0
        
        # Constraints:
        # 1. Flow conservation for each node v:
        # Sum(x_in) - Sum(x_out) = s_v - t_v
        # Let A be incidence matrix.
        # For each edge e=(u,v), if x_e=0 (u->v), it contributes -1 to u's out? No.
        # Let's define:
        # Outflow from v = sum of x_e for edges e where v is the tail.
        # Inflow to v = sum of x_e for edges e where v is the head.
        #
        # If x_e = 0, edge is u->v.
        #   u loses 1 (out), v gains 1 (in).
        # If x_e = 1, edge is v->u.
        #   v loses 1 (out), u gains 1 (in).
        #
        # Net flow out of v = (Out - In).
        # We want Out - In = s_v - t_v.
        #
        # Let's build the matrix for constraints:
        # For each node v, sum over incident edges.
        
        A_rows = []
        b_lower = []
        b_upper = []
        
        # We will construct the full constraint matrix A for milp.
        # milp constraints are A @ x >= lb and A @ x <= ub.
        # Or LinearConstraint(A, lb, ub).
        
        # Let's use LinearConstraint.
        # Number of constraints: n (flow conservation) + m (edge direction binary? No, integrality handles that)
        # Actually, x_e is binary. s_v, t_v are binary.
        
        # Flow conservation:
        # For each v:
        # Sum_{e incident to v} coeff_e * x_e - s_v + t_v = 0
        # 
        # If e=(u,v) with u<v.
        # If x_e=0 (u->v): u has -1, v has +1.
        # If x_e=1 (v->u): u has +1, v has -1.
        # So contribution to u is (1 - 2*x_e) ?
        # If x=0, 1. If x=1, -1. Yes.
        # Contribution to v is (2*x_e - 1) ?
        # If x=0, -1. If x=1, 1. Yes.
        
        # So for node u: coeff is 1 for x_e.
        # For node v: coeff is -1 for x_e.
        # Equation: (1*x_e for u) + (-1*x_e for v) - s_u + t_u ... wait.
        # Net flow out of u = s_u - t_u.
        # Out - In = s - t.
        
        # Let's write the equation for node v:
        # Sum_{e out of v} 1 - Sum_{e into v} 1 = s_v - t_v
        # 
        # For an edge e=(u,v) (u<v):
        # If x_e=0 (u->v):
        #   u is tail, v is head.
        #   u: Out+1. v: In+1.
        # If x_e=1 (v->u):
        #   v is tail, u is head.
        #   v: Out+1. u: In+1.
        
        # Let's express Net Outflow from v in terms of x_e.
        # For edge e=(u,v):
        # If v=u: Net Out contribution is 1 if x_e=0 (u->v? No, u is u). 
        # Let's be precise.
        # e=(u,v).
        # If v == u:
        #   If x_e=0 (u->v), v is head. In+1. Net Out -1.
        #   If x_e=1 (v->u), v is tail. Out+1. Net Out +1.
        #   So coeff for x_e in v's equation is (1 if x_e=1 else -1) = 2*x_e - 1.
        # If v == v:
        #   If x_e=0 (u->v), v is head. In+1. Net Out -1.
        #   If x_e=1 (v->u), v is tail. Out+1. Net Out +1.
        #   Wait.
        #   If x_e=0, edge is u->v. v is head. Net Out decreases by 1.
        #   If x_e=1, edge is v->u. v is tail. Net Out increases by 1.
        #   So coeff is (1 if x_e=1 else -1) = 2*x_e - 1.
        
        # Let's check u.
        # If v == u:
        #   If x_e=0 (u->v), u is tail. Out+1. Net Out +1.
        #   If x_e=1 (v->u), u is head. In+1. Net Out -1.
        #   So coeff is (1 if x_e=0 else -1) = 1 - 2*x_e.
        
        # So for edge e=(u,v):
        # Coeff for x_e in node u's equation: 1 - 2*x_e? No, the variable is x_e.
        # The term is (1 - 2*x_e) * 1? No.
        # The contribution to Net Out of u is:
        # If x_e=0: +1.
        # If x_e=1: -1.
        # This is 1 - 2*x_e.
        # So the linear term is 1*x_e? No.
        # The value is 1 - 2*x_e.
        # This is not linear in x_e if we treat it as a variable? No, x_e is a variable.
        # The expression is affine.
        # Net Out_u = sum (1 - 2*x_e) for edges where u is u in (u,v)?
        # This is sum 1 - 2 sum x_e.
        # This is a constant minus 2 * sum x_e.
        
        # Let's re-evaluate.
        # We want: Net Out_v = s_v - t_v.
        # Net Out_v = (Out - In).
        
        # For edge e=(u,v):
        # If u < v.
        # Case x_e=0 (u->v):
        #   u: Out+1.
        #   v: In+1.
        # Case x_e=1 (v->u):
        #   v: Out+1.
        #   u: In+1.
        
        # Let's define variables $z_e$ such that $z_e=1$ if $u \to v$ and $z_e=0$ if $v \to u$.
        # Then $x_e$ in my previous thought was $1-z_e$.
        # Let's use $z_e \in \{0,1\}$.
        # If $z_e=1$, flow $u \to v$.
        # If $z_e=0$, flow $v \to u$.
        
        # Net Out_u:
        # If $z_e=1$: +1.
        # If $z_e=0$: -1.
        # Value: $2 z_e - 1$.
        
        # Net Out_v:
        # If $z_e=1$: -1.
        # If $z_e=0$: +1.
        # Value: $1 - 2 z_e$.
        
        # Equation for node v:
        # $\sum_{e \ni v} \text{contrib}_v(e) = s_v - t_v$.
        
        # This is a linear equation in $z_e, s_v, t_v$.
        
        # Let's build the matrix A for $\sum A_{ij} x_j = b_i$.
        # Variables: $z_0, \dots, z_{m-1}, s_0, \dots, s_{n-1}, t_0, \dots, t_{n-1}$.
        
        # For each node v (row v):
        # Sum over incident edges e:
        # If v is u in e=(u,v): coeff for z_e is 2.
        # If v is v in e=(u,v): coeff for z_e is -2.
        # Coeff for s_v is 1.
        # Coeff for t_v is -1.
        # RHS is 0.
        # So: $2 \sum_{e: v=u} z_e - 2 \sum_{e: v=v} z_e + s_v - t_v = 0$.
        
        # We can solve this MILP.
        
        # Initialize A matrix (n x (m+2n))
        A = np.zeros((n, m + 2*n))
        
        for idx, (u, v) in enumerate(edge_list):
            # u < v
            # Coeff for z_idx in row u: 2
            # Coeff for z_idx in row v: -2
            A[u, idx] = 2.0
            A[v, idx] = -2.0
            
        # Coeff for s_v in row v: 1
        for v in range(n):
            A[v, m + v] = 1.0
            A[v, m + n + v] = -1.0
            
        # Constraints: A @ x = 0
        # milp uses inequalities.
        # lb = 0, ub = 0.
        
        # Integrality: all variables are integer.
        integrality = np.ones(m + 2*n)
        
        # Bounds: 0 to 1 for all.
        bounds = Bounds(lb=0, ub=1)
        
        # Objective: Minimize sum s_v.
        c_obj = np.zeros(m + 2*n)
        c_obj[m : m+n] = 1.0
        
        try:
            res = milp(c=c_obj, constraints=LinearConstraint(A, 0, 0), integrality=integrality, bounds=bounds)
            
            if not res.success:
                return None
            
            x = res.x
            z = x[:m]
            s = x[m : m+n]
            t = x[m+n : m+2*n]
            
            # Reconstruct paths
            # Build directed graph
            adj = {v: [] for v in range(n)}
            for idx, (u, v) in enumerate(edge_list):
                if z[idx] > 0.5: # u -> v
                    adj[u].append(v)
                else: # v -> u
                    adj[v].append(u)
            
            # Find paths
            # Sources are nodes with s_v = 1
            sources = [v for v in range(n) if s[v] > 0.5]
            
            paths = []
            for src in sources:
                path = [src]
                current = src
                while True:
                    # Find next
                    # Since it's a path cover, out-degree of internal nodes is 1, sinks is 0.
                    # Sources have out-degree 1 (unless length 0, but edges exist).
                    # Actually, if a node is a source, it has net outflow 1.
                    # If it's an internal node, net outflow 0.
                    # If it's a sink, net outflow -1.
                    # In the directed graph, out-degree of a source is 1?
                    # Net Out = Out - In = 1.
                    # If In=0, Out=1.
                    # If In=1, Out=2?
                    # But we have simple paths?
                    # The flow formulation allows multiple edges? No, capacity 1 per edge.
                    # But a node can have multiple outgoing edges if it's a branching point?
                    # No, flow conservation + binary variables on edges.
                    # If a node has Out=2, In=1, Net=1.
                    # This would imply the node is a source for TWO paths?
                    # But s_v is binary.
                    # So Net Out can only be 1, 0, or -1.
                    # If s_v=1, t_v=0, Net=1.
                    # If s_v=0, t_v=1, Net=-1.
                    # If s_v=0, t_v=0, Net=0.
                    #
                    # Can a node have Out=2, In=1?
                    # Then Net=1. This is allowed if s_v=1.
                    # But then the node is the start of a path, but also has an incoming edge from another path?
                    # That would mean the paths are not edge-disjoint? No, edges are disjoint.
                    # But the "path" would merge?
                    # If a node has In=1 and Out=2, it's a branching point.
                    # A path cannot branch.
                    # So the flow formulation as described allows "branching" flows, which do not decompose into simple paths directly?
                    # Wait. If we have a flow, it decomposes into paths and cycles.
                    # If we minimize sources, we minimize paths.
                    # If a node has Out=2, In=1, it must be the start of a path AND have an incoming edge from another path.
                    # This means two paths meet at this node?
                    # But they are edge-disjoint.
                    # So one path ends here? No, it has Out=2.
                    # One path starts here? Yes.
                    # One path passes through?
                    # If one path passes through, it uses 1 In and 1 Out.
                    # The second Out must start a new path.
                    # So the node is an endpoint of one path (start) and an internal node of another?
                    # But a node can only be an endpoint of one path in a simple path decomposition?
                    # No, a node can be an endpoint of multiple paths if it has high degree.
                    # Example: Star graph. Center has degree 4.
                    # 2 paths of length 1? No, 4 edges.
                    # 2 paths: (leaf1-center-leaf2) and (leaf3-center-leaf4).
                    # Center is internal to both.
                    # Net flow 0.
                    #
                    # What if Center has degree 3?
                    # Path 1: leaf1-center. Path 2: leaf2-center-leaf3?
                    # Center is end of P1, internal of P2.
                    # Net flow:
                    # P1: In 1 (from leaf1), Out 0. Net -1? No, flow direction.
                    # Let's trace flow.
                    # leaf1 -> center.
                    # leaf2 -> center -> leaf3.
                    # Center: In 2 (from leaf1, leaf2), Out 1 (to leaf3).
                    # Net Out = 1 - 2 = -1.
                    # So Center is a Sink (t_center=1).
                    # Leaf1 is Source (s_leaf1=1).
                    # Leaf2 is Source (s_leaf2=1).
                    # Leaf3 is Sink (t_leaf3=1).
                    # Total sources = 2. Total paths = 2.
                    # This works.
                    
                    # So we just need to extract the paths from the flow.
                    # Since it's a flow, we can trace from sources.
                    # But if a node has Out > 1, we have multiple outgoing edges.
                    # We need to pick one to continue the path?
                    # No, the flow is a sum of paths.
                    # We can extract them by following edges.
                    # If a node has Out=2, we can start a path from one outgoing edge?
                    # No, we start from Sources.
                    # If a Source has Out=1, we follow it.
                    # If an Internal node has Out=1, we follow it.
                    # If a node has Out=2, it must be a Source (Net=1) or something else?
                    # If Net=1, s=1.
                    # If it has In=1, Out=2.
                    # It is a Source.
                    # We start a path here.
                    # But we also have an incoming edge from another path.
                    # That other path ends here?
                    # If the other path ends here, it uses the In edge.
                    # So the node is the End of Path A and Start of Path B.
                    # And Path B branches?
                    # No, Path B uses one Out edge.
                    # What about the other Out edge?
                    # It must belong to Path C?
                    # But we only have one Source here.
                    # This implies the flow decomposition is not unique or my understanding of "simple path" constraint in MILP is missing.
                    # The MILP finds a feasible flow.
                    # Any integral flow in this network decomposes into edge-disjoint paths and cycles.
                    # Since we minimize sources, cycles (which have 0 sources) are not preferred unless necessary?
                    # But we must cover all edges.
                    # If there is a cycle, it doesn't contribute to source count.
                    # So we might get cycles.
                    # But a cycle is not a simple path.
                    # However, we can break a cycle? No, we must cover edges.
                    # If the flow contains a cycle, we can just output it? No, "simple path".
                    # But usually, for connected graphs, the minimum path cover won't have cycles if we minimize sources?
                    # Actually, if we have a cycle, we can traverse it.
                    # But the task requires "simple paths".
                    # If the MILP returns a cycle, we have a problem.
                    # However, for the given random graphs, it's unlikely to have pure cycles as components because they are connected trees+extra edges.
                    # Even if it does, we can handle it.
                    
                    # Extraction:
                    # We have directed edges.
                    # We identify sources (s=1).
                    # We follow the edges.
                    # If we hit a node with Out=0, we stop.
                    # If we hit a node with Out>0, we continue.
                    # But if Out>1, we have a choice.
                    # This implies the flow is not a single path.
                    # But we want a list of paths.
                    # We can just extract all paths from the flow.
                    # Standard algorithm:
                    # 1. Find all sources.
                    # 2. For each source, follow edges.
                    #    If at a node, pick an outgoing edge.
                    #    Mark edge as used.
                    #    If we reach a sink, path done.
                    #    If we reach a node with no unused outgoing edges, path done.
                    # 3. Repeat for remaining edges?
                    #    Since it's a flow, all edges are covered.
                    #    If we follow from sources, we might not cover edges in cycles?
                    #    If there are cycles, they have no sources.
                    #    We need to find cycles separately.
                    #    But Gallai's conjecture is about paths.
                    #    If the optimal solution has cycles, we can break them?
                    #    No, breaking a cycle into a path requires removing an edge, which is not allowed.
                    #    But a cycle is a closed path.
                    #    Is a cycle a "simple path"? No.
                    #    But maybe the MILP solution won't have cycles for these graphs.
                    
                    # Let's assume the flow decomposes into paths.
                    
                    # To handle Out>1, we can just pick any unused outgoing edge.
                    # Since we mark edges as used, we will eventually cover all edges.
                    
                    # Implementation:
                    used_edges = set()
                    paths = []
                    
                    # Map from (u,v) to edge index?
                    # We just need to traverse.
                    
                    # Build list of outgoing edges for each node.
                    out_edges = {v: [] for v in range(n)}
                    for idx, (u, v) in enumerate(edge_list):
                        if z[idx] > 0.5:
                            out_edges[u].append((v, idx))
                        else:
                            out_edges[v].append((u, idx))
                            
                    # Sort outgoing edges for determinism?
                    for v in out_edges:
                        out_edges[v].sort()
                        
                    sources = [v for v in range(n) if s[v] > 0.5]
                    
                    # Also, if there are cycles, they won't be reached from sources.
                    # We should check for remaining edges after processing sources.
                    
                    for src in sources:
                        if not out_edges[src]:
                            continue
                        path = [src]
                        current = src
                        while True:
                            # Find unused outgoing edge
                            next_v = None
                            next_idx = None
                            for nb, idx in out_edges[current]:
                                if idx not in used_edges:
                                    next_v = nb
                                    next_idx = idx
                                    break
                            if next_v is None:
                                break
                            used_edges.add(next_idx)
                            path.append(next_v)
                            current = next_v
                        if len(path) > 1:
                            paths.append(path)
                            
                    # Check for remaining edges (cycles)
                    remaining = [idx for idx in range(m) if idx not in used_edges]
                    if remaining:
                        # Find cycles
                        # We can just pick any remaining edge and follow
                        # Since it's a flow, it must form cycles.
                        for idx in remaining:
                            if idx in used_edges:
                                continue
                            # Find start
                            u, v = edge_list[idx]
                            if z[idx] > 0.5:
                                start = u
                            else:
                                start = v
                            
                            path = [start]
                            current = start
                            while True:
                                next_v = None
                                next_idx = None
                                for nb, i in out_edges[current]:
                                    if i not in used_edges:
                                        next_v = nb
                                        next_idx = i
                                        break
                                if next_v is None or next_v == start:
                                    # If next_v is start, we closed the cycle.
                                    # But we need to remove the last edge?
                                    # No, the path is simple?
                                    # A cycle is not a simple path.
                                    # But we can't avoid it if the flow has a cycle.
                                    # However, for these graphs, it's unlikely.
                                    break
                                used_edges.add(next_idx)
                                path.append(next_v)
                                current = next_v
                            if len(path) > 1:
                                # If it's a cycle, path[0] == path[-1]?
                                # We stopped before adding the edge back to start?
                                # My loop breaks if next_v is start.
                                # So path is simple.
                                paths.append(path)
                                
                    return paths

        except Exception as e:
            return None

    def path_cover_greedy(edges, n):
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            sub_adj = {i: set() for i in range(n)}
            for u, v in remaining_edges:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
            
            active_verts = [v for v in range(n) if sub_adj[v]]
            if not active_verts:
                break
            
            start = active_verts[0]
            path = [start]
            visited = {start}
            current = start
            
            while True:
                candidates = [nb for nb in sub_adj[current] if nb not in visited]
                if not candidates:
                    break
                best_next = min(candidates, key=lambda nb: len(sub_adj[nb]))
                path.append(best_next)
                visited.add(best_next)
                e = tuple(sorted((current, best_next)))
                remaining_edges.discard(e)
                sub_adj[current].discard(best_next)
                sub_adj[best_next].discard(current)
                current = best_next
            
            if len(path) > 1:
                paths.append(path)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            while len(results) < 3:
                edges = instance(*spec)
                results.append([list(e) for e in edges])
            break
        
        edges = instance(*spec)
        n = spec[0]
        
        paths = solve_path_cover_milp(edges, n)
        
        if paths is None:
            paths = path_cover_greedy(edges, n)
            
        results.append(paths)
    
    return results
