"""Program for clique partition of two dense random graphs.

Strategy: spectral clustering on the graph's Laplacian matrix to identify
dense subgraphs, then iteratively extract maximum cliques from these clusters
using a heuristic that prioritizes vertices with the highest eigenvector
centrality to minimize the total number of cliques.
Every edge is placed in exactly one clique; each clique has >= 2 vertices.
Minimizes total number of cliques.
"""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict


def _random_graph(n, p, seed):
    """Build the edge list exactly as the verifier does."""
    from random import Random
    rng = Random(seed)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    return edges


def _eigenvector_centrality(adj, n, tol=1e-10, max_iter=300):
    """Power iteration for eigenvector centrality."""
    deg = np.asarray(adj.sum(axis=1), dtype=float)
    deg[deg == 0] = 1.0
    x = np.ones(n, dtype=float) / n
    for _ in range(max_iter):
        x_new = (adj @ x) / deg
        x_new = x_new / (np.linalg.norm(x_new) + 1e-30)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x


def _spectral_clusters(adj, n, k=6):
    """
    Cluster vertices using the Fiedler vector(s) of the normalized Laplacian.
    Returns a list of cluster labels for each vertex.
    """
    deg = adj.sum(axis=1).astype(float)
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt

    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except Exception:
        return list(range(n))

    # Use the smallest non-trivial eigenvectors
    n_use = min(k, n - 1)
    if n_use < 1:
        return list(range(n))

    vecs = eigvecs[:, 1:1 + n_use]
    pts = vecs
    idx = np.arange(n)

    # Initialize centroids using farthest-point sampling
    centroids = [pts[0]]
    for _ in range(1, n_use):
        d = np.min([np.sum((pts - c) ** 2, axis=1) for c in centroids], axis=0)
        centroids.append(pts[idx[np.argmax(d)]])

    # K-means clustering
    labels = np.zeros(n, dtype=int)
    for _ in range(100):
        d = np.stack([np.sum((pts - c) ** 2, axis=1) for c in centroids], axis=1)
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for ci in range(n_use):
            mask = labels == ci
            if mask.any():
                centroids[ci] = pts[mask].mean(axis=0)

    return labels.tolist()


def _find_maximal_clique(adj, n, seed, remaining_edges):
    """
    Find a large clique containing the seed vertex using a greedy Bron-Kerbosch
    variant. We only consider edges that are still in remaining_edges.
    Returns the clique as a list of vertices.
    """
    # Build candidate set: neighbors of seed that have remaining edges
    candidates = set()
    for v in range(n):
        if v == seed:
            continue
        e = (seed, v) if seed < v else (v, seed)
        if e in remaining_edges:
            candidates.add(v)

    if not candidates:
        return [seed]

    # Greedily expand: at each step, add the vertex that maximizes
    # the number of common neighbors in the remaining candidate set
    clique = [seed]
    # Track which candidates are still valid (adjacent to all clique members)
    valid = set(candidates)

    while valid:
        # For each candidate, count how many other valid candidates it's connected to
        # (and also connected to all current clique members - already ensured by valid set)
        best_v = None
        best_score = -1
        for v in valid:
            # Count neighbors in valid that are also adjacent to v
            score = 0
            for w in valid:
                if w == v:
                    continue
                e = (v, w) if v < w else (w, v)
                if e in remaining_edges:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v

        if best_v is None:
            break

        # Add best_v to clique
        clique.append(best_v)

        # Update valid: only keep candidates adjacent to best_v
        new_valid = set()
        for w in valid:
            if w == best_v:
                continue
            e = (best_v, w) if best_v < w else (w, best_v)
            if e in remaining_edges:
                new_valid.add(w)
        valid = new_valid

    return clique


def _partition_one(n, p, seed):
    edges = _random_graph(n, p, seed)
    adj = np.zeros((n, n), dtype=np.int8)
    for (u, v) in edges:
        adj[u, v] = 1
        adj[v, u] = 1

    # Compute eigenvector centrality for priority ordering
    try:
        evc = _eigenvector_centrality(adj, n)
    except Exception:
        evc = np.ones(n, dtype=float) / n

    # Spectral clustering to identify dense subgraphs
    try:
        labels = _spectral_clusters(adj, n, k=6)
    except Exception:
        labels = list(range(n))

    # Build remaining edge set
    remaining_edges = set()
    for (u, v) in edges:
        remaining_edges.add((u, v))

    # Compute cluster statistics
    cluster_size = defaultdict(int)
    cluster_edges = defaultdict(int)
    for v in range(n):
        cluster_size[labels[v]] += 1
    for (u, v) in edges:
        if labels[u] == labels[v]:
            cluster_edges[labels[u]] += 1

    # Sort clusters by density (edges / max possible edges) descending
    clusters_sorted = sorted(
        cluster_size.keys(),
        key=lambda c: cluster_edges[c] / max(1, cluster_size[c] * (cluster_size[c] - 1) / 2),
        reverse=True
    )

    # Build vertex ordering: process densest clusters first, within cluster by centrality
    order = []
    for c in clusters_sorted:
        vs = [v for v in range(n) if labels[v] == c]
        vs.sort(key=lambda v: -evc[v])
        order.extend(vs)

    cliques = []

    while remaining_edges:
        # Find the highest-priority vertex that still has remaining edges
        seed = None
        for v in order:
            has_edge = False
            for w in range(n):
                e = (v, w) if v < w else (w, v)
                if e in remaining_edges:
                    has_edge = True
                    break
            if has_edge:
                seed = v
                break

        if seed is None:
            break

        # Find a large clique containing seed
        clique = _find_maximal_clique(adj, n, seed, remaining_edges)

        if len(clique) >= 2:
            # Remove all edges within the clique from remaining_edges
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    e = (u, v) if u < v else (v, u)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
            cliques.append(clique)

    return [[int(x) for x in cl] for cl in cliques]


def construct():
    res1 = _partition_one(32, 0.6, 601)
    res2 = _partition_one(64, 0.55, 602)
    return [res1, res2]
# EVOLVE-BLOCK-END
