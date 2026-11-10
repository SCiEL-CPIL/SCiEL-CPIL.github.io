"""
Dense Q3-free graph on 16 vertices.

Strategy:
1. Precompute all Q3 embeddings as frozensets of 12 edge indices (20,384 total).
2. Greedy init + kick-and-add local search with fast Q3 detection.
3. The key insight: a Q3 in the graph corresponds to a set of 8 vertices where
   the induced subgraph contains all 12 edges of Q3. We check this by verifying
   that the 12 specific edge-pairs (under some vertex mapping) are all present.

For efficiency, we precompute for each 8-subset of vertices all Q3 mappings and
store them as frozensets of the 12 edge indices. Then checking if adding edge e
creates a Q3 is: does any precomputed Q3 hyperedge H satisfy e in H and H ⊆ current_edges?

Actually, that's still heavy. Let's use a different approach:

A graph on 16 vertices contains Q3 as a subgraph iff there exist 8 vertices
and a bijection from Q3 vertices to them such that all 12 edges map to present edges.

We can precompute all such "forbidden configurations": for each 8-subset S and each
Q3 embedding on S, record the frozenset of 12 edge-indices. There are C(16,8) * (8!/48)
= 12870 * 1344 = 17,300,880 such configurations. Too many to precompute fully.

Better: use a fast detection function. When adding edge (u,v), a Q3 containing it
must have u and v as adjacent vertices in Q3. In Q3, each pair of adjacent vertices
appears in exactly 4 different Q3s (since Q3 has 12 edges, and each edge is in 4
different "copies"... actually let me think more carefully).

Actually, let's just use the direct check: after adding (u,v), check if the resulting
graph contains Q3. We can do this by checking all 8-subsets containing u and v.
C(14,6) = 3003 subsets. For each, check if the induced subgraph on those 8 vertices
contains Q3 as a subgraph. This is a small graph isomorphism check on 8 vertices.

But even 3003 * (isomorphism check) might be slow. Let's think of a smarter way.

Alternative: Use the fact that Q3 is a 3-regular bipartite graph on 8 vertices.
When we add edge (u,v), any Q3 containing it must have u and v as endpoints of an
edge in Q3. So we need to find 6 more vertices x1..x6 such that the 8 vertices
{u,v,x1,...,x6} induce a Q3 with (u,v) as one of the 12 edges.

Let's precompute: for each unordered pair (u,v), what are all the Q3 embeddings
where u and v are adjacent? For a given 8-set containing u and v, the number of Q3
embeddings where u,v are adjacent is (8!/48) * (12/28) ... this is getting complicated.

Let me just use a practical approach: fast Q3 check via backtracking on small candidate sets.
"""

import time
import random
import math
from itertools import combinations

def construct():
    deadline = time.time() + 115
    n = 16
    
    # All edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_set = set()
    for u, v in all_edges:
        edge_set.add((u, v))
    m = len(all_edges)  # 120
    
    # Map edge to index
    edge_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_idx[(u, v)] = i
    
    def count_edges(adj):
        c = 0
        for u in range(n):
            c += adj[u].bit_count()
        return c // 2
    
    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result
    
    # Fast Q3 detection: check if graph (adj) contains Q3 as subgraph
    # Use the fact that Q3 is bipartite with parts of size 4
    # Q3 vertices can be labeled as binary 3-tuples
    # Edges connect vertices differing in exactly 1 bit
    
    # Precompute Q3 structure
    # Vertices: 0,1,2,3,4,5,6,7 (binary: 000,001,010,011,100,101,110,111)
    Q3_ADJ = []
    for i in range(8):
        neighbors = []
        for j in range(8):
            if i == j:
                continue
            diff = i ^ j
            if diff in (1, 2, 4):  # differ in exactly 1 bit
                neighbors.append(j)
        Q3_ADJ.append(neighbors)
    
    # Bipartition: even parity vs odd parity
    # Even: 0,3,5,6 (binary: 000,011,101,110)
    # Odd: 1,2,4,7 (binary: 001,010,100,111)
    Q3_EVEN = [0, 3, 5, 6]
    Q3_ODD = [1, 2, 4, 7]
    
    def has_q3(adj):
        """Check if the graph on vertices 0..15 contains Q3 as a subgraph."""
        # For each 8-subset, check if it induces a Q3
        # But that's expensive. Instead, use the structure.
        
        # Q3 is bipartite with parts A and B, |A|=|B|=4
        # Each vertex in A has degree 3, each in B has degree 3
        # The bipartite adjacency matrix is 4x4 with each row/col having 3 ones
        
        # Strategy: try all pairs of 4-sets (A, B) that are disjoint,
        # and check if the bipartite subgraph between A and B contains Q3 as a subgraph.
        
        # Actually, let's just check all 8-subsets but with early termination.
        # C(16,8) = 12870. For each, check if induced subgraph contains Q3.
        
        # For an 8-vertex induced subgraph, it contains Q3 iff there exists a
        # bijection from Q3 vertices to the 8 vertices preserving edges.
        
        # Since Q3 has 12 edges, and the induced subgraph on 8 vertices has at most 28 edges,
        # we can check by trying all 8! / 48 = 1344 isomorphisms... too many.
        
        # Better: use the bipartite structure.
        # In Q3, the bipartition is unique (up to swap).
        # So for each 8-subset S, we need to check if there exist disjoint subsets A, B ⊆ S
        # with |A|=|B|=4 such that the bipartite graph between A and B in G[S]
        # contains Q3 as a subgraph.
        
        # Actually, the bipartition of Q3 is unique. So for a given 8-set S,
        # we need to check if G[S] has a bipartite subgraph isomorphic to Q3.
        
        # Let me just do a direct check: for each 8-subset, try to map Q3 to it.
        # But 12870 * (check) might be too slow.
        
        # Alternative fast check: 
        # A graph contains Q3 iff it contains a cycle of length 4 in the "line graph" sense...
        # This is getting complicated. Let me just do the direct 8-subset check but optimize.
        
        for S in combinations(range(n), 8):
            S_set = set(S)
            # Check if the induced subgraph on S contains Q3
            # Build adjacency within S
            # Try all bijections from Q3 vertices to S
            # Use backtracking with pruning
            
            # Map Q3 vertices to S vertices
            mapping = [-1] * 8
            inverse = {}
            
            def backtrack(q3_idx):
                if q3_idx == 8:
                    # Check if all Q3 edges are present
                    for i in range(8):
                        for j in Q3_ADJ[i]:
                            if j > i:  # only check once
                                u, v = mapping[i], mapping[j]
                                if u > v:
                                    u, v = v, u
                                if not (adj[u] & (1 << v)):
                                    return False
                    return True
                
                for s in S:
                    if s in inverse:
                        continue
                    # Pruning: check if assigning s to q3_idx is consistent
                    # Check edges from q3_idx to already-mapped Q3 vertices
                    for j in range(q3_idx):
                        if j in Q3_ADJ[q3_idx]:
                            u = mapping[j]
                            v = s
                            if u > v:
                                u, v = v, u
                            if not (adj[u] & (1 << v)):
                                break
                    else:
                        mapping[q3_idx] = s
                        inverse[s] = q3_idx
                        if backtrack(q3_idx + 1):
                            return True
                        del inverse[s]
                        mapping[q3_idx] = -1
                
                return False
            
            if backtrack(0):
                return True
        
        return False
    
    def creates_q3(u, v, adj):
        """Check if adding edge (u,v) to the graph with adjacency adj creates a Q3."""
        # Only check 8-subsets containing both u and v
        others = [x for x in range(n) if x != u and x != v]
        for six in combinations(others, 6):
            S = tuple(sorted(six + (u, v)))
            S_set = set(S)
            
            # Check if induced subgraph on S contains Q3
            # Use backtracking
            mapping = [-1] * 8
            
            def backtrack(q3_idx):
                if q3_idx == 8:
                    for i in range(8):
                        for j in Q3_ADJ[i]:
                            if j > i:
                                a, b = mapping[i], mapping[j]
                                if a > b:
                                    a, b = b, a
                                if not (adj[a] & (1 << b)):
                                    return False
                    return True
                
                for s in S:
                    # Pruning: for already-mapped neighbors
                    for j in range(q3_idx):
                        if j in Q3_ADJ[q3_idx]:
                            a = mapping[j]
                            b = s
                            if a > b:
                                a, b = b, a
                            if not (adj[a] & (1 << b)):
                                break
                    else:
                        mapping[q3_idx] = s
                        if backtrack(q3_idx + 1):
                            return True
                        mapping[q3_idx] = -1
                
                return False
            
            if backtrack(0):
                return True
        
        return False
    
    def local_search(adj, deadline):
        """Kick-and-add local search."""
        current_adj = adj[:]
        current_count = count_edges(adj)
        best_adj = adj[:]
        best_count = current_count
        
        # Precompute list of current edges
        def get_edges(adj):
            result = []
            for u in range(n):
                mask = adj[u]
                while mask:
                    bit = mask & (-mask)
                    v = bit.bit_length() - 1
                    if v > u:
                        result.append((u, v))
                    mask -= bit
            return result
        
        for kick_iter in range(200):
            if time.time() > deadline - 5:
                break
            
            trial_adj = current_adj[:]
            
            # Get current edges
            cur_edges = get_edges(trial_adj)
            
            # Kick: remove 3-8 random edges
            k = random.randint(3, 8)
            if len(cur_edges) < k:
                k = len(cur_edges)
            to_remove = random.sample(cur_edges, k)
            for u, v in to_remove:
                trial_adj[u] &= ~(1 << v)
                trial_adj[v] &= ~(1 << u)
            
            # Add all possible missing edges greedily
            for u, v in all_edges:
                if not (trial_adj[u] & (1 << v)):
                    if not creates_q3(u, v, trial_adj):
                        trial_adj[u] |= (1 << v)
                        trial_adj[v] |= (1 << u)
            
            trial_count = count_edges(trial_adj)
            
            if trial_count > current_count:
                current_adj = trial_adj[:]
                current_count = trial_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current_adj[:]
            elif trial_count == current_count and random.random() < 0.3:
                # Accept same-count moves with some probability for exploration
                current_adj = trial_adj[:]
        
        return best_adj, best_count
    
    random.seed(42)
    best_adj = [0] * n
    best_count = -1
    
    for seed in range(20):
        if time.time() > deadline - 20:
            break
        
        random.seed(seed * 1000 + 7)
        adj = [0] * n
        
        # Greedy init: add edges in random order
        edge_order = all_edges[:]
        random.shuffle(edge_order)
        for u, v in edge_order:
            if time.time() > deadline - 10:
                break
            if not (adj[u] & (1 << v)) and not creates_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        greedy_count = count_edges(adj)
        
        # Local search
        adj, cnt = local_search(adj, deadline)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
        
        # Try a few more kicks from the best
        if time.time() < deadline - 10:
            adj2, cnt2 = local_search(best_adj, deadline)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = adj2[:]
    
    # Final polish: try adding any remaining edges
    if time.time() < deadline - 2:
        for u, v in all_edges:
            if not (best_adj[u] & (1 << v)):
                if not creates_q3(u, v, best_adj):
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
                    best_count += 1
    
    return edges_from_adj(best_adj)
