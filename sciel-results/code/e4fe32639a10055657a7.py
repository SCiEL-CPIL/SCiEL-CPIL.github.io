import random
import time

def construct():
    """
    Constructs clique partitions for two fixed dense random graphs.
    Returns a list of two lists, where each inner list contains cliques (as lists of vertices).
    """
    
    def random_graph_edges(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_clique_partition(edges, n, num_trials=200, time_limit=10):
        """
        Runs a randomized greedy algorithm to partition edges into cliques.
        Returns the partition with the fewest cliques found.
        """
        best_partition = None
        best_count = float('inf')
        start_time = time.time()
        
        # Pre-compute adjacency sets for fast lookup
        # adj[v] = set of neighbors u such that (min(u,v), max(u,v)) in edges
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # We will simulate the removal of edges. 
        # To make it efficient, we can maintain a set of remaining edges.
        
        for trial in range(num_trials):
            if time.time() - start_time > time_limit:
                break
                
            current_edges = set(edges)
            partition = []
            
            # Shuffle vertices or edges to introduce randomness?
            # The randomness comes from the choice of expansion vertex.
            
            while current_edges:
                # Pick a random edge from current_edges
                # To avoid O(N) pick from set repeatedly which is fine for small N, 
                # but we can convert to list if needed. For N~64, |E|~1000, it's fine.
                if not current_edges:
                    break
                    
                # Pick random edge
                # Using list(current_edges)[random.randint(0, len(current_edges)-1)] is O(E)
                # Better: maintain a list? Or just use set iteration?
                # For simplicity and speed in Python, picking from a list is faster than set if we convert, 
                # but converting every step is O(E).
                # Let's just pick one.
                
                # Optimization: If we pick a random edge, we might pick a "bad" one that leads to a small clique.
                # A better heuristic: Pick an edge incident to a high-degree vertex in the remaining graph.
                # However, maintaining degrees dynamically is costly.
                # Let's stick to random edge + greedy expansion with random tie-breaking.
                
                e = random.choice(list(current_edges))
                u, v = e
                
                # Start clique with {u, v}
                clique = {u, v}
                
                # Candidates are common neighbors of u and v in current_edges
                # common_neighbors = adj[u] & adj[v]
                # But we must ensure the edge (x, y) exists in current_edges for all pairs in clique.
                # Since we build the clique incrementally, we only need to check connectivity to the new vertex.
                
                # Initial candidates: vertices w such that (u,w) and (v,w) are in current_edges
                # Note: adj is static. We must check current_edges.
                
                cand = set()
                for w in adj[u]:
                    if (min(u, w), max(u, w)) in current_edges and (min(v, w), max(v, w)) in current_edges:
                        cand.add(w)
                        
                while cand:
                    # Pick a candidate w
                    # To encourage larger cliques, we could pick w with max degree in cand?
                    # Let's try random pick first, as it's faster and diversity helps.
                    w = random.choice(list(cand))
                    clique.add(w)
                    
                    # Update candidates: must be connected to w in current_edges
                    new_cand = set()
                    for x in cand:
                        if x == w:
                            continue
                        if (min(w, x), max(w, x)) in current_edges:
                            new_cand.add(x)
                    cand = new_cand
                
                # Add clique to partition
                partition.append(list(clique))
                
                # Remove all edges in this clique from current_edges
                clique_list = list(clique)
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        a, b = clique_list[i], clique_list[j]
                        edge_to_remove = (min(a, b), max(a, b))
                        if edge_to_remove in current_edges:
                            current_edges.remove(edge_to_remove)
                            
            if len(partition) < best_count:
                best_count = len(partition)
                best_partition = partition
                
        return best_partition

    # Instance 1
    n1, p1, seed1 = 32, 0.6, 601
    edges1 = random_graph_edges(n1, p1, seed1)
    partition1 = greedy_clique_partition(edges1, n1, num_trials=100, time_limit=5)
    
    # Instance 2
    n2, p2, seed2 = 64, 0.55, 602
    edges2 = random_graph_edges(n2, p2, seed2)
    partition2 = greedy_clique_partition(edges2, n2, num_trials=200, time_limit=10)
    
    # Fallback if something goes wrong (should not happen)
    if partition1 is None:
        partition1 = [[list(e)] for e in edges1] # Wait, clique must be list of vertices. Edge is 2 vertices.
        partition1 = [[u, v] for u, v in edges1]
    if partition2 is None:
        partition2 = [[u, v] for u, v in edges2]

    return [partition1, partition2]

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(f"Instance 1: {len(res[0])} cliques")
    print(f"Instance 2: {len(res[1])} cliques")
