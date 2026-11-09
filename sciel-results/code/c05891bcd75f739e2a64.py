import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 145
    
    adj = [set() for _ in range(n)]
    edges = set()
    
    def add_edge(u, v):
        edges.add((u, v))
        edges.add((v, u))
        adj[u].add(v)
        adj[v].add(u)

    def remove_edge(u, v):
        edges.discard((u, v))
        edges.discard((v, u))
        adj[u].discard(v)
        adj[v].discard(u)

    # Initial 8 K5s
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                add_edge(u, v)
    
    current_edges = len(edges) // 2
    best_edges = current_edges
    best_adj = [set(s) for s in adj]
    best_edge_set = set(edges)
    
    def can_add_edge(u, v):
        if (u, v) in edges:
            return False
        nu = adj[u]
        nv = adj[v]
        for a in nu:
            for b in adj[a]:
                for c in adj[b]:
                    if not adj[c].isdisjoint(nv):
                        return False
        return True

    no_improvement = 0
    
    while time.time() < deadline:
        added = False
        # Try to add edges
        for _ in range(50):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or (u, v) in edges:
                continue
            if can_add_edge(u, v):
                add_edge(u, v)
                current_edges += 1
                added = True
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [set(s) for s in adj]
                    best_edge_set = set(edges)
                no_improvement = 0
                break
        
        if not added:
            no_improvement += 1
            if no_improvement > 5:
                # Perturb
                if edges:
                    e = random.choice(list(edges))
                    u, v = e
                    remove_edge(u, v)
                    current_edges -= 1
                    
                    # Try to add back or others
                    for _ in range(50):
                        u2 = random.randint(0, n - 1)
                        v2 = random.randint(0, n - 1)
                        if u2 == v2 or (u2, v2) in edges:
                            continue
                        if can_add_edge(u2, v2):
                            add_edge(u2, v2)
                            current_edges += 1
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_adj = [set(s) for s in adj]
                                best_edge_set = set(edges)
                no_improvement = 0

    final_edges = []
    for u, v in best_edge_set:
        if u < v:
            final_edges.append([u, v])
    return final_edges
