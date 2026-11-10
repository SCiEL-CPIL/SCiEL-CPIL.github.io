import random
import time

def construct():
    n = 15
    deadline = time.time() + 80  # safety margin
    
    def get_adjacency(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def get_degrees(adj):
        return [len(adj[i]) for i in range(n)]
    
    def count_violations(adj):
        degs = get_degrees(adj)
        violations = 0
        for u in range(n):
            for v in adj[u]:
                for t in adj[v]:
                    if t != u and t != u:  # t != u already since adj is symmetric, need s != t
                        for s in adj[u]:
                            if s != t and s != v:
                                if degs[s] == degs[t]:
                                    violations += 1
        return violations
