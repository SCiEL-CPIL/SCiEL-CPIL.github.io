import random

def construct():
    n = 18
    # Initialize with K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = set()
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    edges.add((u, v))
    
    def creates_k4(u, v, edges):
        common = []
        for w in range(n):
            if w == u or w == v: continue
            if (u, w) in edges or (w, u) in edges:
                if (v, w) in edges or (w, v) in edges:
                    common.append(w)
        for i in range(len(common)):
            for j in range(i+1, len(common)):
                w1, w2 = common[i], common[j]
                if (w1, w2) in edges or (w2, w1) in edges:
                    return True
        return False

    def greedy_tf_size(edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
            
        best_greedy = 0
        for _ in range(20):
            order = list(range(n))
            random.shuffle(order)
            S = []
            for v in order:
                ok = True
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        u1, u2 = S[i], S[j]
                        if adj[v][u1] and adj[v][u2] and adj[u1][u2]:
                            ok = False
                            break
                    if not ok: break
                if ok:
                    S.append(v)
            best_greedy = max(best_greedy, len(S))
        return best_greedy

    current_edges = set(edges)
    best_score = 12
    best_graph = current_edges
    
    for step in range(500):
        score = greedy_tf_size(current_edges)
        if score < best_score:
            best_score = score
            best_graph = set(current_edges)
            if best_score <= 10:
                break
        
        if random.random() < 0.5:
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            non_edges = [p for p in all_pairs if p not in current_edges]
            if non_edges:
                u, v = random.choice(non_edges)
                if not creates_k4(u, v, current_edges):
                    current_edges.add((u, v))
        else:
            if current_edges:
                u, v = random.choice(list(current_edges))
                current_edges.remove((u, v))
                
    return [[u, v] for u, v in best_graph]
