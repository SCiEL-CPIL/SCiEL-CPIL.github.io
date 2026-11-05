"""Program for clique partition of two dense random graphs.

Strategy: spectral clustering on the Laplacian to identify dense regions,
then greedy maximum-clique extraction prioritizing eigenvector centrality.
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


def _eigenvector_centrality(adj, n, tol=1e-10, max_iter=200):
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


def _spectral_clusters(adj, n, k=4):
    """Cluster vertices using the Fiedler vector(s) of the normalized Laplacian."""
    deg = adj.sum(axis=1)
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except Exception:
        return list(range(n))
    # Use the smallest eigenvectors (excluding the trivial constant one)
    n_use = min(k, n - 1)
    if n_use < 1:
        return list(range(n))
    vecs = eigvecs[:, 1:1 + n_use]
    # Simple k-means on the embedding
    # initialize with k far-apart points
    pts = vecs
    idx = np.arange(n)
    # pick initial centroids
    centroids = [pts[0]]
    for _ in range(1, n_use):
        d = np.min([np.sum((pts - c) ** 2, axis=1) for c in centroids], axis=0)
        centroids.append(pts[idx[np.argmax(d)]])
    labels = np.zeros(n, dtype=int)
    for _ in range(50):
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


def _greedy_clique_partition(adj, n, order):
    """
    Greedily partition edges into cliques using bitset operations for speed.
    `order` is a permutation of vertices to try as seeds (high priority first).
    Returns a list of cliques (each a list of vertex indices, size >= 2).
    """
    # remaining edges as a boolean matrix
    rem_adj = adj.copy().astype(bool)
    
    # remaining degree per vertex
    rem_deg = np.sum(rem_adj, axis=1)
    
    cliques = []
    
    # Precompute a priority rank: lower rank = higher priority
    rank = np.zeros(n, dtype=int)
    for i, v in enumerate(order):
        rank[v] = i
    
    # Active vertices: those with rem_deg > 0
    active = np.where(rem_deg > 0)[0]
    
    while len(active) > 0:
        # choose seed: vertex with remaining edges and best (lowest) rank
        # Filter active vertices to find the one with min rank
        seed = active[np.argmin(rank[active])]
        
        # Build a clique starting from seed.
        # Current clique members
        clique = [seed]
        # Common neighbors of all clique members (bitset)
        # Start with neighbors of seed
        common = rem_adj[seed].copy()
        common[seed] = False # Exclude self
        
        # We want to grow the clique. 
        # Strategy: repeatedly pick a candidate from `common` that is "best".
        # "Best" could be highest remaining degree, or lowest rank.
        # To maximize clique size, we should pick vertices that keep the intersection large.
        # However, for minimizing number of cliques, we want *large* cliques.
        # Greedy: pick candidate with max (rem_deg) or max (common count)?
        # Let's pick the candidate in `common` that has the highest remaining degree.
        
        while True:
            # Find candidates in common
            cands = np.where(common)[0]
            if len(cands) == 0:
                break
            
            # Score candidates: prioritize those with high remaining degree
            # Tie-break by rank (lower is better)
            # We want to maximize rem_deg, so we sort descending by rem_deg, then ascending by rank
            # To do this efficiently:
            # Create a score: rem_deg * large_number - rank
            # Or just argsort.
            
            # Get rem_deg and rank for candidates
            cand_deg = rem_deg[cands]
            cand_rank = rank[cands]
            
            # Sort indices: primary key -cand_deg (so higher deg is earlier), secondary key cand_rank
            # Since we want max deg, we can just find the max deg, then among those min rank.
            max_deg = np.max(cand_deg)
            max_deg_cands = cands[cand_deg == max_deg]
            
            if len(max_deg_cands) > 1:
                # Pick the one with lowest rank
                best_cand = max_deg_cands[np.argmin(cand_rank[cand_deg == max_deg])]
            else:
                best_cand = max_deg_cands[0]
            
            clique.append(best_cand)
            
            # Update common neighbors
            common = common & rem_adj[best_cand]
            common[best_cand] = False # Exclude self from future candidates (though it's already in clique)
            
            # Optimization: If common is empty, break
            if not np.any(common):
                break
        
        # Now remove all edges among clique members that still exist
        # Only add clique if size >= 2
        if len(clique) >= 2:
            cliques.append(clique)
            
            # Update rem_adj and rem_deg for all pairs in clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    if rem_adj[u, v]:
                        rem_adj[u, v] = False
                        rem_adj[v, u] = False
                        rem_deg[u] -= 1
                        rem_deg[v] -= 1
            
            # Update active list
            # A vertex is active if rem_deg > 0
            active = np.where(rem_deg > 0)[0]
            
    return cliques


def _partition_one(n, p, seed):
    edges = _random_graph(n, p, seed)
    adj = np.zeros((n, n), dtype=np.int8)
    for (u, v) in edges:
        adj[u, v] = 1
        adj[v, u] = 1

    # Eigenvector centrality for priority ordering
    try:
        evc = _eigenvector_centrality(adj, n)
    except Exception:
        evc = np.arange(n, dtype=float)

    # Spectral clusters to process dense regions first
    try:
        labels = _spectral_clusters(adj, n, k=4)
    except Exception:
        labels = list(range(n))

    # Build an order: vertices sorted by (cluster density desc, centrality desc)
    # Compute per-cluster remaining-edge density as a proxy
    cluster_size = defaultdict(int)
    cluster_edges = defaultdict(int)
    for v in range(n):
        cluster_size[labels[v]] += 1
    for (u, v) in edges:
        if labels[u] == labels[v]:
            cluster_edges[labels[u]] += 1

    cluster_score = {}
    for c in cluster_size:
        sz = cluster_size[c]
        cluster_score[c] = (cluster_edges[c] / (sz * sz) if sz > 0 else 0.0, -evc.mean() if False else 0.0)

    # Order vertices: group by cluster (densest first), within cluster by centrality desc
    clusters_sorted = sorted(cluster_size.keys(), key=lambda c: (cluster_edges[c] / max(1, cluster_size[c] ** 2)), reverse=True)
    order = []
    for c in clusters_sorted:
        vs = [v for v in range(n) if labels[v] == c]
        vs.sort(key=lambda v: -evc[v])
        order.extend(vs)

    cliques = _greedy_clique_partition(adj, n, order)

    # Convert to lists of vertex lists
    return [[int(x) for x in cl] for cl in cliques]


def construct():
    res1 = _partition_one(32, 0.6, 601)
    res2 = _partition_one(64, 0.55, 602)
    return [res1, res2]
# EVOLVE-BLOCK-END
