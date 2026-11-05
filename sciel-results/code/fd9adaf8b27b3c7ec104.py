"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import sys
from collections import defaultdict

# Deadline guard: evaluator kills after 150s, keep safety margin.
_DEADLINE = 118.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = 118.0

    n = 16
    total_edges_k16 = n * (n - 1) // 2  # 120 edges in K16
    
    # Precompute Q3 structure
    # Q3 vertices: 0..7 (3-bit strings)
    # Edge exists if Hamming distance = 1
    q3_edges = []
    for u in range(8):
        for v in range(u + 1, 8):
            if bin(u ^ v).count('1') == 1:
                q3_edges.append((u, v))
    # 12 edges in Q3
    
    # Precompute all Q3 embeddings in K16
    # An embedding is a mapping from Q3 vertices {0..7} to distinct vertices of {0..15}
    # such that all 12 Q3 edges map to edges in K16 (which is always true since K16 is complete)
    # So we just need all injective mappings from 8 Q3 vertices to 16 graph vertices.
    # Number: 16P8 = 16!/8! = 518,918,400 - too many to enumerate directly.
    
    # Instead, for the conflict graph approach:
    # Two edges e1=(a,b) and e2=(c,d) in K16 conflict if there exists a Q3 embedding
    # that uses both e1 and e2.
    # 
    # Key insight: Two edges conflict if they can both appear in some Q3 copy.
    # An edge (u,v) can appear in a Q3 if u and v are at Hamming distance 1 in the Q3 labeling.
    # But since we're mapping to arbitrary vertices, any pair of vertices can be adjacent in Q3.
    # 
    # Actually, let's think differently. For a given set of edges E in our graph,
    # E contains a Q3 iff there exist 8 distinct vertices v0,...,v7 such that
    # for each Q3 edge (i,j), (vi,vj) is in E.
    #
    # Conflict graph approach:
    # Nodes: the 120 edges of K16
    # Two nodes (edges) conflict if they "cannot coexist" in a Q3-free graph.
    # But that's not quite right. Two edges conflict if EVERY Q3 embedding that uses
    # one edge also uses the other? No, that's not right either.
    #
    # Correct formulation: We want to select a maximum subset of edges from K16
    # such that no Q3 is fully contained. This is equivalent to: for every Q3 embedding
    # (which specifies 12 edges), we must NOT select all 12 edges.
    #
    # This is a "maximum independent set in a hypergraph" problem, or equivalently,
    # a hitting set / set cover dual problem.
    #
    # Alternative: Think of it as - we have a family F of 12-edge sets (each Q3 embedding).
    # We want the largest subset S of the 120 edges such that no member of F is a subset of S.
    #
    # This is the "maximum F-free subset" problem. For large F, this is hard.
    #
    # Let's use a different structural approach: branch-and-bound on the edge set.
    # State: (edges_included, edges_excluded)
    # Upper bound: |edges_included| + |edges not in either set|
    # If upper bound <= best, prune.
    # Branch: pick an undecided edge, try including it, then excluding it.
    # When including an edge, check if it completes any Q3. If so, must exclude.
    #
    # To check if adding edge e completes a Q3: we need to find if there's a Q3 embedding
    # that uses e and has all other 11 edges already in edges_included.
    #
    # Let's precompute: for each edge e, the list of Q3 embeddings that contain e.
    # Then when we add e, we check if any such embedding has all other edges already included.
    
    # First, enumerate all Q3 embeddings more efficiently.
    # A Q3 embedding is determined by choosing 8 vertices and a bijection to Q3 vertices.
    # But we can think of it as: choose an ordered 8-tuple of distinct vertices,
    # and check if the induced subgraph contains all 12 Q3 edges.
    #
    # For the conflict detection, let's precompute for each pair of edges whether
    # they appear together in some Q3. But that's still complex.
    
    # Let's use a practical approach: 
    # 1. Start with a good initial solution (K3,13 with 39 edges)
    # 2. Use local search / simulated annealing to improve
    # 3. Fast Q3 check using bitmasks
    
    # Fast Q3 check:
    # For a graph given by adjacency bitmasks, check if it contains Q3.
    # Use the fact that Q3 is 3-regular bipartite.
    
    def has_q3_fast(adj):
        """Check if graph with adjacency bitmasks contains Q3. adj[i] is bitmask of neighbors of i."""
        # Q3 is bipartite with parts A={0,3,5,6} and B={1,2,4,7} (by parity)
        # Actually let me recompute: vertex i in Q3 has neighbors i^1, i^2, i^4
        # Parity (sum of bits mod 2):
        # 0=000(0), 1=001(1), 2=010(1), 3=011(0), 4=100(1), 5=101(0), 6=110(0), 7=111(1)
        # Part A (even): {0, 3, 5, 6}
        # Part B (odd): {1, 2, 4, 7}
        
        # For each 8-subset S, check if it can host Q3.
        # Optimization: the subgraph on S must have at least 12 edges.
        
        # Precompute Q3 adjacency
        q3_adj = [0] * 8
        for u, v in q3_edges:
            q3_adj[u] |= (1 << v)
            q3_adj[v] |= (1 << u)
        
        # Enumerate 8-subsets
        for S in itertools.combinations(range(n), 8):
            # Quick check: count edges in G[S]
            ec = 0
            for i in range(8):
                for j in range(i + 1, 8):
                    if adj[S[i]] & (1 << S[j]):
                        ec += 1
            if ec < 12:
                continue
            
            # Build local adjacency
            la = [0] * 8
            for i in range(8):
                for j in range(i + 1, 8):
                    if adj[S[i]] & (1 << S[j]):
                        la[i] |= (1 << j)
                        la[j] |= (1 << i)
            
            # Check if la contains Q3 as subgraph using backtracking
            # Assign Q3 vertices 0..7 to local positions 0..7
            assigned = [-1] * 8
            used = [False] * 8
            
            def bt(qv):
                if qv == 8:
                    return True
                for lv in range(8):
                    if used[lv]:
                        continue
                    # Check consistency with already-assigned neighbors
                    ok = True
                    for qu in range(qv):
                        if assigned[qu] != -1 and (q3_adj[qv] & (1 << qu)):
                            if not (la[lv] & (1 << assigned[qu])):
                                ok = False
                                break
                    if ok:
                        assigned[qv] = lv
                        used[lv] = True
                        if bt(qv + 1):
                            return True
                        assigned[qv] = -1
                        used[lv] = False
                return False
            
            if bt(0):
                return True
        
        return False
    
    # Build initial solution: K3,13
    def build_initial():
        adj = [0] * 16
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        return adj
    
    adj = build_initial()
    best_adj = adj[:]
    best_edges = 39
    
    # Local search: try to add edges, remove edges that create Q3
    # Use a queue of candidate edges to try
    
    # Get all edges not in current graph
    def get_missing_edges(adj):
        missing = []
        for i in range(16):
            for j in range(i + 1, 16):
                if not (adj[i] & (1 << j)):
                    missing.append((i, j))
        return missing
    
    # Greedy addition with Q3 check
    random.seed(42)
    missing = get_missing_edges(adj)
    random.shuffle(missing)
    
    for u, v in missing:
        if _time_left() < 10.0:
            break
        
        # Try adding edge (u, v)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        if has_q3_fast(adj):
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            best_edges += 1
            best_adj = adj[:]
    
    # Now try edge swaps: remove one edge, add another
    # This can escape local optima
    for iteration in range(200):
        if _time_left() < 15.0:
            break
        
        # Try random edge swaps
        current_edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                if adj[i] & (1 << j):
                    current_edges.append((i, j))
        
        if not current_edges:
            break
        
        # Pick a random edge to remove
        rem_u, rem_v = random.choice(current_edges)
        
        # Pick a random missing edge to add
        missing_now = get_missing_edges(adj)
        if not missing_now:
            break
        add_u, add_v = random.choice(missing_now)
        
        # Remove rem edge
        adj[rem_u] &= ~(1 << rem_v)
        adj[rem_v] &= ~(1 << rem_u)
        
        # Add add edge
        adj[add_u] |= (1 << add_v)
        adj[add_v] |= (1 << add_u)
        
        # Check Q3
        if has_q3_fast(adj):
            # Revert: put back rem edge, remove add edge
            adj[add_u] &= ~(1 << add_v)
            adj[add_v] &= ~(1 << add_u)
            adj[rem_u] |= (1 << rem_v)
            adj[rem_v] |= (1 << rem_u)
        else:
            # Accept if same or better edge count
            # (swap keeps same count, so just accept)
            pass
    
    # Final greedy pass
    missing = get_missing_edges(adj)
    random.shuffle(missing)
    for u, v in missing:
        if _time_left() < 5.0:
            break
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if has_q3_fast(adj):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            best_edges += 1
            best_adj = adj[:]
    
    # Build final edge list
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            if adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
