"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    def gen_instance(n, p, seed):
        rng = np.random.RandomState(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random_sample() < p:
                    edges.add((u, v))
        return edges

    def solve(n, edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        unassigned = set(edge_set)
        cliques = []
        
        # Process edges in sorted order
        for u, v in sorted(edge_set):
            if (u, v) not in unassigned:
                continue
            
            # Greedily build maximal clique containing (u,v) using only unassigned edges
            clique = [u, v]
            clique_set = {u, v}
            
            # Candidate vertices to add
            for w in range(n):
                if w in clique_set:
                    continue
                # Check if (w, x) is unassigned for all x in clique
                can_add = True
                for x in clique:
                    e = (min(w, x), max(w, x))
                    if e not in unassigned:
                        can_add = False
                        break
                if can_add:
                    clique.append(w)
                    clique_set.add(w)
            
            # Remove all edges in this clique from unassigned
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    unassigned.discard(e)
            
            cliques.append(clique)
        
        return cliques
    
    edges1 = gen_instance(32, 0.6, 601)
    edges2 = gen_instance(64, 0.55, 602)
    
    cliques1 = solve(32, edges1)
    cliques2 = solve(64, edges2)
    
    return [cliques1, cliques2]
# EVOLVE-BLOCK-END
