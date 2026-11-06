# Inside solve, after greedy partition:
    improved = True
    while improved:
        improved = False
        for i in range(len(partition)):
            for j in range(i + 1, len(partition)):
                # Try to merge if possible
                union = set(partition[i]) | set(partition[j])
                if is_clique(union):
                    partition[i] = sorted(union)
                    partition.pop(j)
                    improved = True
                    break
            if improved:
                break
        
        # Try to move edges to empty a clique
        for i in range(len(partition)):
            if len(partition[i]) < 2:
                partition.pop(i)
                improved = True
                break
        
        # Try moving an edge from Ci to Cj to see if Ci becomes empty or smaller
        # This is O(N^2 * E) which might be slow.
        # Let's just do the merge check and a simple "shrink" check.
        
        # Actually, the merge check is the most direct way to reduce count.
        # The greedy algorithm already produces maximal cliques, so merging two maximal cliques is rare unless they were disjoint in the greedy step but actually form a larger clique together? No, if they are maximal in the *remaining* graph, their union might not be a clique because edges between them were covered by other cliques? No, edges between vertices in Ci and Cj are either in Ci, Cj, or another clique. If they are in another clique, they are not in the "uncovered" set when Ci and Cj were formed? 
        # Wait, the partition covers *all* edges. 
        # If Ci and Cj are in the partition, all edges within Ci are covered by Ci, all within Cj by Cj.
        # Edges between Ci and Cj are covered by *some* clique in the partition.
        # If we merge Ci and Cj, we require that ALL edges between Ci and Cj exist in the original graph.
        # If they exist, they must be in the partition. If they are in Ci or Cj, that's fine. If they are in another clique Ck, then Ck must contain vertices from both Ci and Cj? No, Ck is a set of vertices. If Ck contains u in Ci and v in Cj, then Ck intersects both.
        # So merging Ci and Cj is only valid if the subgraph induced by Ci U Cj is a complete graph.
        # This check is worth doing.
