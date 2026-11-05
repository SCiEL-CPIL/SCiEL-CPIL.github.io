def construct():
    import time
    from random import Random
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigs
    
    deadline = time.time() + 140  # 10s safety margin
    
    def generate_graph(n, p, seed):
        rng = Random(seed)
        edges = []
        adj = [[False]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u][v] = True
                    adj[v][u] = True
        return edges, adj
    
    def solve_instance(edges, adj, n):
        if not edges:
            return []
        
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.float64)
        for u, v in edges:
            A[u, v] = 1.0
            A[v, u] = 1.0
        
        # Compute Laplacian L = D - A
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Get eigenvectors of the Laplacian
        # We want the smallest non-zero eigenvalues to find dense subgraphs
        try:
            if n <= 100:
                eigvals, eigvecs = np.linalg.eigh(L)
                # Sort by eigenvalue
                idx = np.argsort(eigvals)
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
            else:
                # Use sparse eigenvalue solver for larger graphs
                L_sp = csr_matrix(L)
                eigvals, eigvecs = eigs(L_sp, k=min(10, n-1), which='SM')
                # eigs returns complex in some cases, take real part
                eigvals = np.real(eigvals)
                eigvecs = np.real(eigvecs)
                # Sort
                idx = np.argsort(eigvals)
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
        except Exception:
            eigvecs = np.eye(n)
        
        # Use the second eigenvector (Fiedler vector) to identify clusters
        # Vertices with similar values in the Fiedler vector are in the same dense region
        fiedler = eigvecs[:, 1] if eigvecs.shape[1] > 1 else np.ones(n)
        
        # Normalize Fiedler vector
        fiedler_mean = fiedler.mean()
        fiedler_std = fiedler.std()
        if fiedler_std > 1e-10:
            fiedler_norm = (fiedler - fiedler_mean) / fiedler_std
        else:
            fiedler_norm = np.zeros(n)
        
        # Eigenvector centrality: use the dominant eigenvector of A
        try:
            eigvals_A, eigvecs_A = np.linalg.eig(A)
            # Get the eigenvector with largest magnitude eigenvalue
            max_idx = np.argmax(np.abs(eigvals_A))
            centrality = np.abs(eigvecs_A[:, max_idx])
            # Normalize
            centrality_max = centrality.max()
            if centrality_max > 1e-10:
                centrality = centrality / centrality_max
            else:
                centrality = np.ones(n) / n
        except Exception:
            centrality = np.ones(n) / n
        
        # Combine: score = centrality * (1 + |fiedler_norm|) to prefer high-centrality vertices
        # in dense regions (high |fiedler| means strong cluster membership)
        vertex_score = centrality * (1.0 + np.abs(fiedler_norm))
        
        # Remaining edges
        remaining = set(edges)
        cliques = []
        
        # Precompute edge set for O(1) lookup
        edge_set = set(edges)
        
        while remaining and time.time() < deadline:
            # Find a good clique using spectral-guided heuristic
            # Pick the vertex with highest combined score among vertices with remaining edges
            active_vertices = set()
            for u, v in remaining:
                active_vertices.add(u)
                active_vertices.add(v)
            
            if not active_vertices:
                break
            
            # Pick start vertex: highest score among active vertices
            start = max(active_vertices, key=lambda v: vertex_score[v])
            
            # Build clique greedily, prioritizing high-score vertices
            clique = [start]
            
            # Candidates: neighbors of start in remaining graph
            candidates = set()
            for (u, v) in remaining:
                if u == start:
                    candidates.add(v)
                elif v == start:
                    candidates.add(u)
            
            while candidates and time.time() < deadline:
                # Find candidate connected to all in clique, with highest score
                best = None
                best_score = -1
                for c in candidates:
                    # Check if c connects to all in clique
                    ok = True
                    for m in clique:
                        if m == c:
                            continue
                        pair = (min(m, c), max(m, c))
                        if pair not in remaining:
                            ok = False
                            break
                    if ok:
                        # Use combined score: vertex_score + degree bonus
                        deg_bonus = sum(1 for (a, b) in remaining if a == c or b == c)
                        score = vertex_score[c] + 0.1 * deg_bonus
                        if score > best_score:
                            best_score = score
                            best = c
                if best is None:
                    break
                clique.append(best)
                # Update candidates: must be connected to best
                new_candidates = set()
                for c in candidates:
                    if c == best:
                        continue
                    pair = (min(c, best), max(c, best))
                    if pair in remaining:
                        new_candidates.add(c)
                candidates = new_candidates
            
            # If clique has at least 2 vertices, add it
            if len(clique) >= 2:
                cliques.append(clique)
                # Remove all edges within clique from remaining
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        pair = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        remaining.discard(pair)
            else:
                # Shouldn't happen, but safety
                break
        
        # Leftover edges as 2-cliques
        for (u, v) in remaining:
            cliques.append([u, v])
        
        return cliques
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges, adj = generate_graph(n, p, seed)
        cliques = solve_instance(edges, adj, n)
        results.append(cliques)
    
    return results
