"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 140.0
    
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency matrix
        adj_mat = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj_mat[u, v] = 1
            adj_mat[v, u] = 1
        
        # Build edge set for tracking
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))
        
        remaining_edges = set(edge_set)
        cliques = []
        
        # Spectral clustering to identify dense regions
        # Compute Laplacian
        degree = adj_mat.sum(axis=1)
        L = np.diag(degree) - adj_mat.astype(np.float64)
        
        # Get eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
        except:
            # Fallback to simple greedy if spectral fails
            return _simple_greedy_solve(n, edges, remaining_edges, deadline)
        
        # Use the smallest non-zero eigenvalue's eigenvector for clustering
        # For dense graphs, the Fiedler vector (2nd smallest eigenvalue) is useful
        # But for finding cliques, we might want to look at the principal components
        # Let's use the first few eigenvectors to get a low-dimensional embedding
        
        # Take the first min(n, 4) eigenvectors
        k = min(n, 4)
        embedding = eigenvectors[:, :k]
        
        # Cluster vertices based on their embedding
        # Simple approach: group vertices that are close in embedding space
        # and also densely connected in the original graph
        
        # For now, let's use a simpler spectral-inspired heuristic:
        # Rank vertices by their centrality (eigenvector centrality)
        # and use that to guide clique extraction
        
        # Compute eigenvector centrality (dominant eigenvector of adjacency)
        try:
            from scipy.sparse.linalg import eigsh
            # Use power iteration or scipy
            # For small n, just use numpy
            w, v = np.linalg.eig(adj_mat.astype(np.float64))
            # Take the eigenvector corresponding to the largest eigenvalue
            max_idx = np.argmax(np.abs(w))
            centrality = np.abs(v[:, max_idx])
        except:
            centrality = degree.astype(np.float64)
        
        # Normalize centrality
        centrality = centrality / (centrality.max() + 1e-10)
        
        # Greedy clique extraction guided by centrality
        # Maintain remaining adjacency
        rem_adj = adj_mat.copy()
        
        while remaining_edges:
            if time.time() > deadline - 10:
                # Fallback: cover remaining edges individually
                for e in remaining_edges:
                    cliques.append(list(e))
                break
            
            # Find vertex with highest centrality among those with remaining edges
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            
            if not active_vertices:
                break
            
            # Pick vertex with highest centrality
            start_vertex = max(active_vertices, key=lambda x: centrality[x])
            
            # Find a clique in the remaining graph starting from start_vertex
            # Use greedy expansion with centrality as tie-breaker
            clique = [start_vertex]
            candidates = set(np.where(rem_adj[start_vertex] == 1)[0].tolist())
            candidates.discard(start_vertex)
            
            # Sort candidates by centrality (descending)
            candidates = sorted(candidates, key=lambda x: centrality[x], reverse=True)
            
            for v in candidates:
                # Check if v is connected to all current clique members in remaining graph
                if all(rem_adj[v, u] == 1 for u in clique):
                    clique.append(v)
            
            if len(clique) < 2:
                # Pick an arbitrary edge
                e = next(iter(remaining_edges))
                clique = [e[0], e[1]]
            
            cliques.append(clique)
            
            # Remove edges in this clique from remaining graph
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    e_key = (min(u, v), max(u, v))
                    if e_key in remaining_edges:
                        remaining_edges.remove(e_key)
                        rem_adj[u, v] = 0
                        rem_adj[v, u] = 0
        
        return cliques
    
    def _simple_greedy_solve(n, edges, remaining_edges, deadline):
        """Fallback simple greedy solver"""
        adj_mat = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj_mat[u, v] = 1
            adj_mat[v, u] = 1
        
        degree = adj_mat.sum(axis=1)
        cliques = []
        
        while remaining_edges:
            if time.time() > deadline - 10:
                for e in remaining_edges:
                    cliques.append(list(e))
                break
            
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            
            if not active_vertices:
                break
            
            start_vertex = max(active_vertices, key=lambda x: degree[x])
            
            clique = [start_vertex]
            candidates = set(np.where(adj_mat[start_vertex] == 1)[0].tolist())
            candidates.discard(start_vertex)
            candidates = sorted(candidates, key=lambda x: degree[x], reverse=True)
            
            for v in candidates:
                if all(adj_mat[v, u] == 1 for u in clique):
                    clique.append(v)
            
            if len(clique) < 2:
                e = next(iter(remaining_edges))
                clique = [e[0], e[1]]
            
            cliques.append(clique)
            
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    e_key = (min(u, v), max(u, v))
                    if e_key in remaining_edges:
                        remaining_edges.remove(e_key)
                        adj_mat[u, v] = 0
                        adj_mat[v, u] = 0
        
        return cliques
    
    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
# EVOLVE-BLOCK-END
