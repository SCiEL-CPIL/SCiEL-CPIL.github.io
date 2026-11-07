import time
import random

def _all_simple_cycles(adj, n):
    cycles = set()
    def dfs(start, current, path, visited):
        for nei in adj[current]:
            if nei == start and len(path) >= 3:
                cycles.add(tuple(path))
            elif nei > start and nei not in visited:
                visited.add(nei)
                path.append(nei)
                dfs(start, nei, path, visited)
                path.pop()
                visited.remove(nei)
    for start in range(n):
        dfs(start, start, [start], {start})
    return cycles

def _evaluate_edges(edges, n=12):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for i in range(n):
        if len(adj[i]) < 3:
            return None
    
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k > j and k in adj[i]:
                    return None
    
    cycles = _all_simple_cycles(adj, n)
    if not cycles:
        return None
    
    lengths = set(len(c) for c in cycles)
    return len(lengths)

def construct():
    n = 12
    deadline = time.time() + 80

    edges = set()
    for i in range(6):
        edges.add(tuple(sorted((i, (i + 1) % 6))))
        edges.add(tuple(sorted((i + 6, (i + 1) % 6 + 6))))
        edges.add(tuple(sorted((i, i + 6))))

    best_edges = list(edges)
    best_score = _evaluate_edges(best_edges, n)

    all_possible = set()
    for u in range(n):
        for v in range(u + 1, n):
            all_possible.add((u, v))

    current_edges = set(best_edges)
    current_score = best_score

    random.seed(42)

    while time.time() < deadline:
        if random.random() < 0.5:
            if not current_edges:
                continue
            edge = random.choice(list(current_edges))
            new_edges = current_edges - {edge}
        else:
            candidates = all_possible - current_edges
            if not candidates:
                continue
            edge = random.choice(list(candidates))
            new_edges = current_edges | {edge}

        score = _evaluate_edges(list(new_edges), n)

        if score is not None:
            if current_score is None or score < current_score:
                current_edges = new_edges
                current_score = score
                if score < best_score:
                    best_score = score
                    best_edges = list(new_edges)
                    if best_score == 1:
                        break
            elif random.random() < 0.05:
                current_edges = new_edges
                current_score = score

    return [list(e) for e in sorted(best_edges)]
