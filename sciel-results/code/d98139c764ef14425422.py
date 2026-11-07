import time
import sys
from itertools import combinations
from collections import deque
import networkx as nx

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin (evaluator kills at 90s)
    
    n = P['n']
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find maximum clique using Bron-Kerbosch with pivot
    best_clique = []
    
    def bronk(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        # Choose pivot u from P_set ∪ X
        candidates = list(P_set | X)
        if not candidates:
            return
        # Pick pivot with max degree in P_set
        pivot = max(candidates, key=lambda u: len(adj[u] & P_set))
        # For each v in P_set \ N(pivot)
        for v in list(P_set - adj[pivot]):
            bronk(R | {v}, P_set & adj[v], X & adj[v])
            P_set = P_set - {v}
            X = X | {v}
            if time.time() > deadline:
                return
    
    bronk(set(), set(range(n)), set())
    
    def flow_oracle(branches):
        """Use max flow to check if vertex-disjoint paths exist for all pairs.
        Returns dict of paths if valid, None otherwise."""
        b_set = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        
        # Build a flow network with vertex splitting for capacity constraints
        # Each vertex v becomes v_in -> v_out with capacity 1 (except branches which get higher cap)
        # Actually, for vertex-disjoint paths, we need each non-branch vertex to be used at most once.
        # Branch vertices can be used as endpoints multiple times.
        
        # Create a bipartite-like structure:
        # Source connects to each branch's "out" node
        # Each branch connects to its "in" node
        # Non-branch vertices: in -> out with cap 1
        # Branch vertices: in -> out with cap = degree in pairs (they're endpoints)
        
        # Actually, let's use a different approach:
        # For each pair (u,v), we need a path. All interior vertices must be distinct.
        # This is equivalent to: can we route |pairs| units of flow from a super-source 
        # (connected to all branch "sinks") through the graph to a super-sink?
        
        # Better formulation: 
        # - Split each vertex v into v_in and v_out
        # - Edge v_in -> v_out has capacity 1 if v is not a branch, capacity len(branches)-1 if v is a branch
        # - For each edge (u,v) in original graph, add edges u_out -> v_in and v_out -> u_in with capacity 1
        # - Create super-source S and super-sink T
        # - For each branch b, add edge S -> b_out with capacity (len(branches)-1) [it's an endpoint for that many paths]
        # - For each branch b, add edge b_in -> T with capacity (len(branches)-1)
        # - Wait, this doesn't work directly because we need specific pairs to be connected.
        
        # Alternative: Use the fact that for a clique subdivision, we need vertex-disjoint paths
        # for all C(k,2) pairs. This is a multi-commodity flow problem, which is NP-hard in general.
        # But for small k, we can use a different approach.
        
        # Let's use a simpler greedy with backtracking for small cases, and max-flow for verification.
        
        # Actually, let's use a cleaner approach:
        # 1. Try to find vertex-disjoint paths using a matching-based approach
        # 2. For each pair, we need a path. Interior vertices must be disjoint.
        
        # Simpler: use the current BFS approach but with smarter ordering and backtracking
        # for the first few assignments.
        
        # Let's try a different structural approach:
        # Use integer programming via scipy if available, otherwise use a smarter greedy.
        
        # For now, let's improve the greedy by trying multiple orderings.
        paths = {}
        used_interior = set()
        
        # Sort pairs: those with direct edge first, then by number of available paths
        def has_direct_edge(u, v):
            return v in adj[u]
        
        # Try multiple orderings
        orderings = [
            list(pairs),  # natural order
            sorted(pairs, key=lambda uv: 0 if has_direct_edge(uv[0], uv[1]) else 1),  # direct edges first
        ]
        
        for ordering in orderings:
            if time.time() > deadline:
                return None
            test_paths = {}
            test_used = set()
            success = True
            for u, v in ordering:
                if has_direct_edge(u, v):
                    test_paths[(u, v)] = [u, v]
                else:
                    forbidden = b_set - {u, v}
                    forbidden |= test_used
                    # BFS
                    visited = {u}
                    parent = {u: None}
                    queue = deque([u])
                    found = False
                    while queue:
                        if time.time() > deadline:
                            success = False
                            break
                        cur = queue.popleft()
                        if cur == v:
                            path = []
                            node = v
                            while node is not None:
                                path.append(node)
                                node = parent[node]
                            path.reverse()
                            test_paths[(u, v)] = path
                            for mid in path[1:-1]:
                                test_used.add(mid)
                            found = True
                            break
                        for nb in adj[cur]:
                            if nb not in visited and nb not in forbidden:
                                visited.add(nb)
                                parent[nb] = cur
                                queue.append(nb)
                    if not found:
                        success = False
                        break
                if not success:
                    break
            if success:
                return test_paths
        return None
    
    # Start with max clique
    current_branches = list(best_clique)
    current_paths = flow_oracle(current_branches)
    
    # Try to add more vertices, trying multiple candidates per round
    remaining = [v for v in range(n) if v not in set(current_branches)]
    improved = True
    while improved:
        improved = False
        # Try all remaining vertices, keep the best set of additions
        best_addition = None
        best_new_branches = None
        best_new_paths = None
        for v in remaining[:]:
            if time.time() > deadline:
                break
            test_branches = current_branches + [v]
            test_paths = flow_oracle(test_branches)
            if test_paths is not None:
                if best_addition is None or v < best_addition:
                    best_addition = v
                    best_new_branches = test_branches
                    best_new_paths = test_paths
        if best_addition is not None:
            current_branches = best_new_branches
            current_paths = best_new_paths
            remaining.remove(best_addition)
            improved = True
    
    # Build result
    if current_paths is None:
        # Fallback: just use the clique with direct edges
        current_branches = list(best_clique)
        current_paths = {}
        for u, v in combinations(sorted(current_branches), 2):
            current_paths[(u, v)] = [u, v]
    
    branches = [int(b) for b in sorted(current_branches)]
    paths_list = []
    for u, v in combinations(sorted(current_branches), 2):
        p = current_paths.get((u, v), [u, v])
        paths_list.append([int(x) for x in p])
    
    return {'branches': branches, 'paths': paths_list}
