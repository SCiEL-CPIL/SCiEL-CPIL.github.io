import random
import time
import itertools

def construct():
    n = 20
    vertices = list(range(n))
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edges = []
    best_count = 0
    
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    def is_k33_free(adj):
        # Check all triples
        for a1, a2, a3 in itertools.combinations(range(n), 3):
            common = adj[a1] & adj[a2] & adj[a3]
            if len(common) >= 3:
                return False
        return True

    def find_k33(adj):
        for a1, a2, a3 in itertools.combinations(range(n), 3):
            common = adj[a1] & adj[a2] & adj[a3]
            if len(common) >= 3:
                # Pick 3 neighbors
                b1, b2, b3 = list(common)[:3]
                # Return edges of this K3,3
                edges = [
                    (a1, b1), (a1, b2), (a1, b3),
                    (a2, b1), (a2, b2), (a2, b3),
                    (a3, b1), (a3, b2), (a3, b3)
                ]
                return edges
        return None

    def greedy_add():
        random.shuffle(all_edges)
        adj = [set() for _ in range(n)]
        current_edges = set()
        
        for u, v in all_edges:
            safe = True
            nvs = list(adj[v])
            nus = list(adj[u])
            
            if len(nvs) >= 2 and len(nus) >= 2:
                for i in range(len(nvs)):
                    a1 = nvs[i]
                    for j in range(i + 1, len(nvs)):
                        a2 = nvs[j]
                        common_12 = adj[a1] & adj[a2]
                        candidates = common_12 & set(nus)
                        if len(candidates) >= 2:
                            safe = False
                            break
                    if not safe:
                        break
            
            if safe:
                current_edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
        
        return current_edges

    def random_remove():
        # Start with random graph
        adj = [set() for _ in range(n)]
        current_edges = set()
        
        # p = 0.5
        for u, v in all_edges:
            if random.random() < 0.5:
                current_edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
        
        # Remove edges to fix K3,3
        # Limit iterations to avoid infinite loops
        for _ in range(1000):
            edges_in_k33 = find_k33(adj)
            if not edges_in_k33:
                break
            # Pick a random edge from the K3,3 to remove
            edge_to_remove = random.choice(edges_in_k33)
            u, v = edge_to_remove
            if u > v: u, v = v, u
            
            if (u, v) in current_edges:
                current_edges.remove((u, v))
                adj[u].remove(v)
                adj[v].remove(u)
        
        return current_edges

    while time.time() < deadline:
        if random.random() < 0.5:
            edges = greedy_add()
        else:
            edges = random_remove()
        
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = [[u, v] for u, v in edges]
            
    return best_edges
