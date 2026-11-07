import time
import sys

_DEADLINE = 140.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() > _DEADLINE

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency for K (0 to k-1)
        # adj_K[u] is a bitset of neighbors of u in K
        adj_K = [0] * k
        for u in range(k):
            for v in range(u + 1, k):
                adj_K[u] |= (1 << v)
                adj_K[v] |= (1 << u)
        
        # Cross edges: for each v in I, N(v) is a bitset of neighbors in K
        N_I = [0] * (n - k)
        edge_set = set()
        
        # We need to know which edges exist.
        # Since K is a clique, all edges within K exist.
        # Cross edges are probabilistic.
        
        # Let's reconstruct the graph structure properly.
        # The instance function returns a list of edges.
        # Let's just use the edges list to build the structure.
        
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))

        # Bitset representation for K vertices
        # k is small (<=30), so we can use Python integers as bitsets.
        
        # Uncovered edges in K
        # We represent the set of uncovered edges in K as a list of bitsets?
        # Or simpler: a set of tuples (u, v) with u < v < k.
        # Since k is small, set of tuples is fine.
        
        uncovered_K = set((u, v) for u in range(k) for v in range(u + 1, k) if (u, v) in edge_set)
        
        # For each v in I, we need to track which cross edges (v, u) are uncovered.
        # Initially all are uncovered.
        # We will process v in I.
        
        independent = list(range(k, n))
        # Sort by degree in K (number of neighbors in K)
        independent.sort(key=lambda v: -len(adj[v] & set(range(k))))
        
        cliques = []
        
        for v in independent:
            # Get neighbors of v in K
            Nv = set(adj[v]) & set(range(k))
            
            # We need to cover edges (v, u) for u in Nv.
            # We will form cliques {v} + S where S is a clique in K (using uncovered edges).
            
            remaining_Nv = set(Nv)
            while remaining_Nv:
                if _deadline_hit():
                    # Fallback: 2-cliques
                    for u in list(remaining_Nv):
                        cliques.append([v, u])
                    break
                
                # Find a large clique S in remaining_Nv such that all edges within S are in uncovered_K.
                # This is the Maximum Clique problem on the induced subgraph of remaining_Nv
                # using only edges that are in uncovered_K.
                
                # Greedy approach for Max Clique:
                # Order vertices by degree in the induced subgraph.
                # Pick the one with highest degree, add to S, remove non-neighbors.
                
                # Let's use a simple greedy:
                # 1. Compute degree of each vertex in remaining_Nv within the graph of uncovered_K.
                # 2. Pick vertex with max degree.
                # 3. Restrict candidates to its neighbors.
                # 4. Repeat.
                
                S = []
                cand = set(remaining_Nv)
                
                # To make it faster, we can pre-compute the induced subgraph.
                # But k is small, so O(k^2) per step is fine.
                
                while cand:
                    # Calculate degree of each node in cand within the graph defined by uncovered_K
                    deg = {}
                    for u in cand:
                        deg[u] = 0
                    for u in cand:
                        for w in cand:
                            if u < w and (u, w) in uncovered_K:
                                deg[u] += 1
                                deg[w] += 1
                            elif w < u and (w, u) in uncovered_K:
                                deg[u] += 1
                                deg[w] += 1
                            
                    if not cand:
                        break
                        
                    # Pick node with max degree
                    best = max(cand, key=lambda x: deg[x])
                    S.append(best)
                    
                    # New candidates must be neighbors of best in uncovered_K
                    new_cand = set()
                    for w in cand:
                        if w == best:
                            continue
                        # Check if (best, w) is in uncovered_K
                        a, b = sorted((best, w))
                        if (a, b) in uncovered_K:
                            new_cand.add(w)
                    cand = new_cand
                
                if not S:
                    # Should not happen if remaining_Nv is not empty
                    S = [next(iter(remaining_Nv))]
                
                clique = [v] + S
                cliques.append(clique)
                
                # Update uncovered_K
                for u in S:
                    for w in S:
                        if u < w:
                            uncovered_K.discard((u, w))
                
                # Update remaining_Nv
                for u in S:
                    remaining_Nv.discard(u)
        
        # Cover remaining edges in K
        rem_edges = set(uncovered_K)
        while rem_edges:
            if _deadline_hit():
                for e in list(rem_edges):
                    cliques.append(list(e))
                break
            
            # Find max clique in rem_edges
            # Pick a vertex with max degree
            deg = {}
            for (u, v) in rem_edges:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            if not deg:
                break
                
            start = max(deg, key=lambda x: deg[x])
            
            S = [start]
            cand = set()
            for (u, v) in rem_edges:
                if u == start:
                    cand.add(v)
                elif v == start:
                    cand.add(u)
            
            while cand:
                # Pick best from cand
                # Calculate degree within cand
                cdeg = {}
                for u in cand:
                    cdeg[u] = 0
                for u in cand:
                    for w in cand:
                        if u < w and (u, w) in rem_edges:
                            cdeg[u] += 1
                            cdeg[w] += 1
                        elif w < u and (w, u) in rem_edges:
                            cdeg[u] += 1
                            cdeg[w] += 1
                
                if not cand:
                    break
                
                best = max(cand, key=lambda x: cdeg[x])
                S.append(best)
                
                new_cand = set()
                for w in cand:
                    if w == best:
                        continue
                    a, b = sorted((best, w))
                    if (a, b) in rem_edges:
                        new_cand.add(w)
                cand = new_cand
            
            if S:
                cliques.append(S)
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        a, b = sorted((S[i], S[j]))
                        rem_edges.discard((a, b))
        
        # Final check and cleanup
        # Ensure all edges are covered.
        # If any edge is missing, add it as a 2-clique (should not happen).
        seen = set()
        for c in cliques:
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    a, b = sorted((c[i], c[j]))
                    seen.add((a, b))
        
        for e in edge_set:
            if e not in seen:
                cliques.append(list(e))
        
        # Convert to lists of ints
        result = []
        for c in cliques:
            result.append([int(x) for x in c])
        return result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    out = []
    for spec in specs:
        if _deadline_hit():
            # Fallback
            out.append([list(e) for e in instance(*spec)])
        else:
            out.append(_solve_split(*spec))
    return out

# Test
if __name__ == "__main__":
    res = construct()
    total = sum(len(x) for x in res)
    print(f"Total cliques: {total}")
