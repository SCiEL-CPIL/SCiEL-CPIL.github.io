"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    deadline = time.time() + 85.0
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)

    def build_signed_adj(signs):
        S = np.zeros((n, n), dtype=np.float64)
        for idx in range(num_edges):
            u, v = edges[idx]
            s = float(signs[idx])
            S[u, v] = s
            S[v, u] = s
        return S

    def check_connectivity(signs):
        lift_n = 2 * n
        adj = [[] for _ in range(lift_n)]
        for idx in range(num_edges):
            u, v = edges[idx]
            s = signs[idx]
            if s == 1:
                adj[u*2].append(v*2)
                adj[v*2].append(u*2)
                adj[u*2+1].append(v*2+1)
                adj[v*2+1].append(u*2+1)
            else:
                adj[u*2].append(v*2+1)
                adj[v*2+1].append(u*2)
                adj[u*2+1].append(v*2)
                adj[v*2].append(u*2+1)
        visited = [False] * lift_n
        visited[0] = True
        q = deque([0])
        count = 1
        while q:
            node = q.popleft()
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    q.append(nb)
        return count == lift_n

    def evaluate(signs):
        if not check_connectivity(signs):
            return float('inf')
        S = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(eigvals)))

    rng = np.random.default_rng(42)
    best_signs = None
    best_radius = float('inf')

    def try_candidate(signs):
        nonlocal best_signs, best_radius
        r = evaluate(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()

    # Initial candidates
    try_candidate(np.ones(num_edges, dtype=int))
    try_candidate(-np.ones(num_edges, dtype=int))
    for _ in range(30):
        try_candidate(rng.choice([-1, 1], size=num_edges))

    # Local search
    stall = 0
    while time.time() < deadline:
        num_flips = int(rng.integers(1, 4))
        new_signs = best_signs.copy()
        flip_idx = rng.choice(num_edges, size=num_flips, replace=False)
        new_signs[flip_idx] *= -1

        r = evaluate(new_signs)
        if r < best_radius:
            best_radius = r
            best_signs = new_signs
            stall = 0
        else:
            stall += 1

        if stall > 2000:
            try_candidate(rng.choice([-1, 1], size=num_edges))
            stall = 0

    return best_signs
# EVOLVE-BLOCK-END
