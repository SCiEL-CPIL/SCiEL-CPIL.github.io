import time
import random
from itertools import combinations
from collections import deque

def _is_q3_free(adj_sets, n=16):
    """Check if graph on n vertices (adjacency as list of sets) is Q3-free."""
    # Q3 has 8 vertices, 12 edges, is bipartite with parts of size 4,
    # each vertex has degree 3 in Q3.
    # Strategy: for every 8-vertex subset, check if induced subgraph contains Q3.
    # But 12870 * (check cost) may be slow.
    # Better: enumerate all 8-vertex subsets, but use a faster test.
    #
    # A subgraph isomorphic to Q3 exists iff there is a set of 8 vertices
    # whose induced subgraph contains a spanning subgraph isomorphic to Q3.
    #
    # Precompute: for a labeled 8-vertex graph, check if it contains Q3 as subgraph.
    # Q3 on vertices {0..7} with edges:
    # 0-1, 0-2, 0-4, 1-3, 2-3, 4-5, 4-6, 5-7, 6-7, 1-5, 2-6, 3-7
    # (standard binary labeling: vertex v connected to v^(1<<i) for i=0,1,2)
    
    # For a given 8-vertex subset, we can try all 8! / 48 = 840 labelings? 
    # That's too slow.
    
    # Alternative: use the fact that Q3 is the unique 3-regular bipartite graph on 8 vertices
    # with girth 6 (actually girth of Q3 is 4? No, Q3 has 4-cycles).
    # Actually Q3 has girth 4.
    
    # Let's use a different approach: enumerate all 8-vertex subsets, and for each,
    # check if the induced subgraph has a spanning subgraph isomorphic to Q3.
    # We can do this by checking if there exists a 3-regular spanning subgraph
    # that is bipartite with both parts of size 4.
    
    # For an 8-vertex induced subgraph with edge set E:
    # 1. Check if it has at least 12 edges (Q3 has 12).
    # 2. Try to find a 3-regular spanning subgraph: this means each vertex has
    #    exactly 3 neighbors within the 8. So the induced subgraph must have
    #    each vertex degree >= 3 in the induced subgraph? No, we need a SUBGRAPH,
    #    so each vertex needs at least 3 neighbors in the 8-set.
    # 3. If all 8 vertices have degree >= 3 in the induced subgraph, then we
    #    need to check if there's a 3-regular spanning subgraph that is Q3.
    
    # Actually, let's just precompute all edge masks of Q3 on 8 labeled vertices.
    # There are 840 distinct labelings (8!/48 = 840).
    # For each 8-vertex subset, map to 0..7, compute the 28-bit edge mask,
    # and check if any of the 840 Q3 masks is a submask.
    
    # Precompute Q3 edge masks:
    q3_masks = set()
    from itertools import permutations
    # Generate all permutations of 0..7, but modulo automorphisms of Q3.
    # Simpler: generate all 8! permutations, compute edge mask, add to set.
    # 40320 permutations, each takes ~12 edge checks. That's fine at startup.
    # But we want unique masks. 40320/48 = 840 unique masks.
    
    # Actually, let's just generate them:
    base_q3_edges = set()
    for v in range(8):
        for i in range(3):
            u = v ^ (1 << i)
            if u > v:
                base_q3_edges.add((v, u))
    
    # Now generate all distinct edge masks by permuting vertex labels:
    # For a permutation p (a bijection 0..7 -> 0..7), the edge mask is:
    # For each (a,b) in base_q3_edges, edge (p(a), p(b)) in the permuted graph.
    # The 28-bit mask: edge (i,j) with i<j is bit (i*7 + j - i*(i+1)//2) or simpler:
    # Use a canonical edge index: for i<j, index = i*8 + j (but that's 64 bits).
    # Better: use a 28-bit mask where edge (i,j) for i<j is at position idx(i,j).
    
    def edge_index(i, j):
        if i > j:
            i, j = j, i
        # Triangular number: number of edges (k,l) with k<l, k<i or (k==i and l<j)
        # Simpler: precompute
        return i * 8 + j  # 64-bit mask, but we'll use a set of frozensets or tuple
    
    # Let's use a frozenset of edges as the key. 840 frozensets of 12 edges each.
    q3_edge_sets = set()
    for perm in permutations(range(8)):
        edges = set()
        for (a, b) in base_q3_edges:
            u, v = perm[a], perm[b]
            edges.add(frozenset((u, v)))
        q3_edge_sets.add(frozenset(edges))
    
    # Now check: for each 8-vertex subset S, compute the induced edge set,
    # and check if any Q3 edge set is a subset of it.
    
    # To speed up: only check if the induced subgraph has at least 12 edges
    # and all 8 vertices have degree >= 3 in the induced subgraph.
    
    for S in combinations(range(n), 8):
        S_set = set(S)
        # Compute induced edges
        induced_edges = set()
        deg = [0] * 8
        for i, vi in enumerate(S):
            for j in range(i + 1, 8):
                vj = S[j]
                if vj in adj_sets[vi]:
                    induced_edges.add(frozenset((vi, vj)))
                    deg[i] += 1
                    deg[j] += 1
        if len(induced_edges) < 12:
            continue
        if any(d < 3 for d in deg):
            continue
        # Check if any Q3 edge set is a subset
        for q3_e in q3_edge_sets:
            if q3_e.issubset(induced_edges):
                return False
    return True


def _count_edges(adj_sets):
    count = 0
    for i in range(len(adj_sets)):
        count += len(adj_sets[i])
    return count // 2


def construct():
    n = 16
    deadline = time.time() + 110  # safety margin
    
    # Start with K_{3,13}
    A = list(range(3))
    B = list(range(3, 16))
    adj = [set() for _ in range(n)]
    for a in A:
        for b in B:
            adj[a].add(b)
            adj[b].add(a)
    
    best_adj = [s.copy() for s in adj]
    best_count = _count_edges(best_adj)
    
    # Candidate edges to add: all edges within A and within B
    candidate_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                candidate_edges.append((i, j))
    
    # Local search: iteratively try to add edges, check validity, remove if invalid
    random.seed(42)
    
    # Sort candidate edges: prioritize internal edges (within A or within B)
    def edge_priority(e):
        i, j = e
        if i in A and j in A:
            return 0
        if i in B and j in B:
            return 1
        return 2
    
    candidate_edges.sort(key=edge_priority)
    
    current_adj = [s.copy() for s in adj]
    current_count = _count_edges(current_adj)
    
    # Add edges greedily one at a time
    for i, j in candidate_edges:
        if time.time() > deadline - 10:
            break
        if j not in current_adj[i]:
            current_adj[i].add(j)
            current_adj[j].add(i)
            # Check if still Q3-free
            # Fast check: only check if the new edge could be part of a Q3
            # For now, do full check but optimize later
            if _is_q3_free(current_adj, n):
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = [s.copy() for s in current_adj]
            else:
                # Remove the edge
                current_adj[i].discard(j)
                current_adj[j].discard(i)
    
    # Now do local search: try swapping edges
    # Or: try to add more edges by first removing some
    while time.time() < deadline - 5:
        # Try random edge additions
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            random.shuffle(candidate_edges)
            added = False
            for i, j in candidate_edges:
                if time.time() > deadline - 5:
                    break
                if j not in current_adj[i]:
                    current_adj[i].add(j)
                    current_adj[j].add(i)
                    if _is_q3_free(current_adj, n):
                        current_count += 1
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = [s.copy() for s in current_adj]
                        added = True
                        break
                    else:
                        current_adj[i].discard(j)
                        current_adj[j].discard(i)
            if not added:
                break
    
    # Return best found
    edges = []
    for i in range(n):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])
    return edges
