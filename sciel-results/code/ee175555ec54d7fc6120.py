import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin

    best_n = 39
    best_edges = []

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break

        # Target min degree
        min_deg_req = n - 39

        # Try multiple random restarts
        found = False
        attempts = 0
        while not found and time.time() < deadline:
            attempts += 1

            # Random graph initialization
            p = (min_deg_req + 2) / (n - 1)

            # Use numpy adjacency matrix (symmetric, no self-loops)
            adj_mat = np.zeros((n, n), dtype=np.int8)
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        adj_mat[u, v] = 1
                        adj_mat[v, u] = 1

            # Remove C4s using numpy
            # A C4 exists if there are two vertices u,v with >=2 common neighbors
            # We iteratively find and break C4s by removing edges
            c4_found = True
            max_iters = 200
            iters = 0
            while c4_found and iters < max_iters and time.time() < deadline:
                c4_found = False
                iters += 1

                # For each pair (u,v), count common neighbors
                # common[u,v] = sum over w of adj[u,w]*adj[v,w]
                # Use matrix multiply: A @ A gives (A^2)[u,v] = number of length-2 paths from u to v
                # For u != v, (A^2)[u,v] counts common neighbors
                A2 = adj_mat @ adj_mat
                # (A^2)[u,v] for u!=v is the number of common neighbors
                # We need to check if any off-diagonal entry >= 2
                # Set diagonal to 0 to ignore self-loops
                np.fill_diagonal(A2, 0)

                # Find pairs with >= 2 common neighbors
                mask = A2 >= 2
                if not mask.any():
                    break

                # Get all violating pairs
                rows, cols = np.where(mask)
                if len(rows) == 0:
                    break

                c4_found = True

                # Process violating pairs - pick one and remove an edge
                # Try to remove an edge that minimally damages min degree
                # For pair (u,v) with common neighbors, find them
                idx = random.randint(0, len(rows) - 1)
                u = int(rows[idx])
                v = int(cols[idx])

                # Find common neighbors of u and v
                common = np.where((adj_mat[u] == 1) & (adj_mat[v] == 1))[0]
                if len(common) < 2:
                    continue

                # Pick two common neighbors
                w1 = int(common[0])
                w2 = int(common[1])

                # Candidate edges to remove: (u,w1), (u,w2), (v,w1), (v,w2)
                # Choose the one whose removal minimally affects min degree
                # Prefer removing edge from higher-degree vertex
                candidates = []
                for w in [w1, w2]:
                    if adj_mat[u, w] == 1:
                        candidates.append((u, w))
                    if adj_mat[v, w] == 1:
                        candidates.append((v, w))

                if not candidates:
                    continue

                # Sort by degree of the endpoint we'd be removing from
                # (remove from higher degree vertex to preserve min degree)
                candidates.sort(key=lambda e: -int(adj_mat[e[0]].sum()))
                eu, ev = candidates[0]

                adj_mat[eu, ev] = 0
                adj_mat[ev, eu] = 0

            # Check min degree
            degrees = adj_mat.sum(axis=1)
            min_deg = int(degrees.min())
            if min_deg >= min_deg_req:
                found = True
                best_n = n
                # Rebuild edges list
                best_edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj_mat[u, v] == 1:
                            best_edges.append([u, v])
                # If we found n=47, we can stop.
                if n == 47:
                    break

        if found and n == 47:
            break

    return {'n': best_n, 'edges': best_edges}
