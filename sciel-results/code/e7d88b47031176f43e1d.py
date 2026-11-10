import time
from random import Random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Build adjacency
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))
        
        # Initial Partition Strategy:
        # 1. For each I-vertex v (k..n-1), form clique {v} U N(v)
        # 2. Cover remaining C-C edges with maximal cliques in the remaining graph
        
        cliques = []
        covered_edges = set()
        
        # Step 1: I-vertex cliques
        for v in range(k, n):
            neighbors = sorted(adj[v])
            if neighbors:
                clique = [v] + neighbors
                cliques.append(clique)
                # Mark edges covered by this clique
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        covered_edges.add(e)
        
        # Step 2: Cover remaining C-C edges
        # The remaining edges are a subset of C-C edges
        rem_edges = [e for e in edge_set if e not in covered_edges]
        
        # Group remaining edges into cliques
        # Since they are all C-C edges, and C is a clique, any set of C-vertices is a clique.
        # We want to partition the set of remaining edges into the fewest cliques.
        # This is equivalent to coloring the line graph? Or just finding a clique cover of the subgraph.
        # Since the subgraph is a subgraph of a complete graph, we can just take the maximal cliques?
        # Actually, if we have remaining edges E_rem, we can just cover them one by one? 
        # Better: Find the vertices involved in rem_edges. 
        # Since it's a subgraph of C, we can just add one clique containing all vertices involved in rem_edges?
        # Wait, if we add one clique K_rem containing all vertices in rem_edges, it covers ALL C-C edges between those vertices.
        # But some of those edges might be already covered? No, rem_edges are the UNCOVERED ones.
        # So if we take the set of vertices V_rem involved in rem_edges, the clique V_rem covers all edges in V_rem.
        # But V_rem might include edges that are NOT in rem_edges (i.e., already covered).
        # If we add V_rem as a clique, it will double-cover those edges.
        # So we cannot just take the vertex set.
        
        # We must partition the edges in rem_edges.
        # Since rem_edges is a subgraph of C, we can use a greedy clique cover on rem_edges.
        # But rem_edges is small? 
        # Let's just cover each edge in rem_edges as a separate clique for the baseline.
        # This is valid but suboptimal. The local search will improve it.
        
        for e in rem_edges:
            cliques.append([e[0], e[1]])
            covered_edges.add(e)
            
        # Data structures for Local Search
        # edge_to_clique: dict from edge to clique index
        edge_to_clique = {}
        for idx, clique in enumerate(cliques):
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    edge_to_clique[e] = idx
        
        # clique_to_edges: dict from clique index to set of edges
        clique_to_edges = defaultdict(set)
        for e, idx in edge_to_clique.items():
            clique_to_edges[idx].add(e)
            
        # clique_to_vertices: list of sets
        clique_to_vertices = [set(c) for c in cliques]
        
        def is_clique(vertices):
            for i, u in enumerate(vertices):
                for v in vertices[i+1:]:
                    if v not in adj[u]:
                        return False
            return True

        def try_merge(i, j):
            # Try to merge clique i and j
            if i == j: return False
            if time.time() > deadline - 5: return False
            
            V_i = clique_to_vertices[i]
            V_j = clique_to_vertices[j]
            V_new = V_i | V_j
            
            # Check if V_new is a clique
            if not is_clique(V_new):
                return False
            
            # Check cross edges
            # Cross edges are edges between V_i \ V_j and V_j \ V_i
            # These edges must NOT be covered by any clique other than i or j
            # Actually, if they are covered by i or j, that's fine (they are internal to the new clique? No, they are cross)
            # Wait, if an edge e is between u in V_i and v in V_j.
            # If e is covered by i, then e is in clique_to_edges[i].
            # If e is covered by j, then e is in clique_to_edges[j].
            # If e is covered by k (k!=i,j), then we have a conflict.
            
            valid = True
            for u in V_i:
                for v in V_j:
                    if u == v: continue
                    e = (min(u, v), max(u, v))
                    if e in edge_to_clique:
                        idx = edge_to_clique[e]
                        if idx != i and idx != j:
                            valid = False
                            break
                if not valid: break
            if not valid: return False
            
            # Perform merge
            # 1. Update clique_to_vertices
            clique_to_vertices[i] = V_new
            clique_to_vertices[j] = set() # Mark as empty
            
            # 2. Update edges
            # All edges in clique_to_edges[j] are now in i
            for e in clique_to_edges[j]:
                edge_to_clique[e] = i
            clique_to_edges[i].update(clique_to_edges[j])
            clique_to_edges[j] = set()
            
            # 3. Handle cross edges
            # Cross edges are now internal to i.
            # They were already in edge_to_clique (mapped to i or j).
            # If they were mapped to i or j, they are now in i.
            # We don't need to add them if they weren't there?
            # Wait, the cross edges MUST be in the partition.
            # If a cross edge e was covered by i, it's in clique_to_edges[i].
            # If it was covered by j, it's in clique_to_edges[j] (now moved to i).
            # So all cross edges are now in i.
            
            # 4. Remove j from cliques
            # We don't physically remove from list to keep indices stable, but we can skip empty cliques in count.
            # Or we can physically remove. Let's physically remove to keep count accurate.
            # But indices change.
            # Let's just keep a list of active cliques.
            
            return True

        # Local Search
        # We will iterate and try to merge random pairs.
        # To avoid index shifting issues, we will use a list of active clique indices.
        active_cliques = list(range(len(cliques)))
        
        improvements = 0
        max_iterations = 10000
        it = 0
        while it < max_iterations and time.time() < deadline - 10:
            it += 1
            if len(active_cliques) < 2:
                break
                
            # Pick two random active cliques
            i_idx = Random().choice(active_cliques)
            j_idx = Random().choice(active_cliques)
            if i_idx == j_idx: continue
            
            if try_merge(i_idx, j_idx):
                # Success
                # Remove j_idx from active
                active_cliques.remove(j_idx)
                improvements += 1
                # Reset iteration counter on success?
                # it = 0 # Maybe not, to prevent infinite loops on small gains
            else:
                # Failure
                pass
                
        # Final cleanup: Build final list of cliques
        final_cliques = []
        for idx in active_cliques:
            if clique_to_vertices[idx]:
                final_cliques.append(sorted(list(clique_to_vertices[idx])))
                
        return final_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue
        
        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)
        
        # Convert to lists of Python ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
        
        results.append(final_cliques)
    
    return results
# EVOLVE-BLOCK-END
