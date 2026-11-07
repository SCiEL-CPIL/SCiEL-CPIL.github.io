"""Optimized spectral graph lift solver using structured search and local refinement."""
import time
import numpy as np
from typing import List, Tuple

def sign_edges(graph, budget):
    """
    Assign signs (-1 or +1) to edges of a prism graph to minimize the spectral radius
    of the signed adjacency matrix, subject to the lift being connected and 3-regular.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin before 90s kill
    
    m = graph['n'] // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build edge index for quick lookup
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    # Identify edge types: cycle edges and matching edges
    # Cycle edges: (i, i+1) mod m for i in 0..m-1 (top) and (m+i, m+i+1) mod m (bottom)
    # Matching edges: (i, m+i) for i in 0..m-1
    
    def get_cycle_edges():
        """Return indices of cycle edges."""
        cycle_idx = []
        for i in range(m):
            # Top cycle
            u, v = i, (i + 1) % m
            if (u, v) in edge_idx:
                cycle_idx.append(edge_idx[(u, v)])
            else:
                cycle_idx.append(edge_idx[(v, u)])
            # Bottom cycle
            u, v = m + i, m + (i + 1) % m
            if (u, v) in edge_idx:
                cycle_idx.append(edge_idx[(u, v)])
            else:
                cycle_idx.append(edge_idx[(v, u)])
        return cycle_idx
    
    def get_matching_edges():
        """Return indices of matching edges."""
        match_idx = []
        for i in range(m):
            u, v = i, m + i
            if (u, v) in edge_idx:
                match_idx.append(edge_idx[(u, v)])
            else:
                match_idx.append(edge_idx[(v, u)])
        return match_idx
    
    def build_signed_adjacency(signs):
        """Build the 2x2 block signed adjacency matrix for the lift."""
        n = graph['n']
        # The signed adjacency matrix A_s is n x n
        # A_s[u][v] = sign if (u,v) is an edge, else 0
        # But for spectral radius of the lift, we need the Schur complement or 
        # equivalently the eigenvalues of the signed adjacency matrix of the base graph
        # with signs. The lift has 2n vertices. The eigenvalues of the lift are the
        # eigenvalues of the signed adjacency matrix A_s of the base graph, but each
        # eigenvalue lambda of A_s gives two eigenvalues of the lift: lambda and -lambda?
        # No, that's not right.
        
        # Actually, for a two-lift defined by signs, the adjacency matrix of the lift
        # is a 2n x 2n matrix. The eigenvalues of the lift are related to the eigenvalues
        # of the signed adjacency matrix A_s (n x n) of the base graph.
        # Specifically, if A_s has eigenvalues lambda_i, then the lift has eigenvalues
        # that are the eigenvalues of the block matrix [[A_s, 0], [0, A_s]]? No.
        
        # Let me think more carefully. The lift has vertices (v, 0) and (v, 1).
        # An edge (u, v) with sign s connects:
        #   (u, 0) to (v, 0) if s = +1, or (u, 0) to (v, 1) if s = -1
        #   (u, 1) to (v, 1) if s = +1, or (u, 1) to (v, 0) if s = -1
        # So the adjacency matrix of the lift is a 2n x 2n matrix.
        # The signed adjacency matrix A_s of the base graph is n x n with A_s[u][v] = s.
        # The eigenvalues of the lift are NOT simply the eigenvalues of A_s.
        
        # However, the problem statement says: "spectral radius of the signed adjacency 
        # matrix of the base graph, max |eigenvalue| from np.linalg.eigvalsh. These are 
        # the new eigenvalues of the lift."
        # So the score is just the spectral radius of the n x n signed adjacency matrix A_s.
        
        A_s = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            A_s[u][v] = s
            A_s[v][u] = s
        return A_s
    
    def spectral_radius(signs):
        """Compute the spectral radius of the signed adjacency matrix."""
        A_s = build_signed_adjacency(signs)
        eigvals = np.linalg.eigvalsh(A_s)
        return np.max(np.abs(eigvals))
    
    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular."""
        n = graph['n']
        # Build the lift graph
        # Vertices: (v, 0) and (v, 1) for v in 0..n-1
        # Total 2n vertices
        # Edge (u, v) with sign s:
        #   s = +1: (u,0)-(v,0) and (u,1)-(v,1)
        #   s = -1: (u,0)-(v,1) and (u,1)-(v,0)
        
        # Build adjacency list for the lift
        adj = [[] for _ in range(2 * n)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                adj[u][v] = adj[v][u] = 1
                adj[n + u][n + v] = adj[n + v][n + u] = 1
            else:
                adj[u][n + v] = adj[n + v][u] = 1
                adj[n + u][v] = adj[v][n + u] = 1
        
        # Check 3-regularity
        for v in range(2 * n):
            if len(adj[v]) != 3:
                return False
        
        # Check connectivity using BFS
        visited = [False] * (2 * n)
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        
        return count == 2 * n
    
    def check_and_score(signs):
        """Check validity and compute spectral radius. Returns (valid, radius)."""
        if not is_valid_lift(signs):
            return False, float('inf')
        return True, spectral_radius(signs)
    
    # Initial solution: use the seed's approach
    rng = np.random.default_rng(3)
    best_signs = None
    best_radius = float('inf')
    
    # Try the seed's random sampling
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=num_edges)
        valid, radius = check_and_score(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(num_edges, dtype=int)
        best_radius = spectral_radius(best_signs)
    
    # Local search: flip one edge at a time
    def local_search(signs, radius, time_limit):
        """Perform local search by flipping individual edges."""
        current_signs = signs.copy()
        current_radius = radius
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            for i in range(num_edges):
                # Try flipping edge i
                new_signs = current_signs.copy()
                new_signs[i] = -new_signs[i]
                valid, new_radius = check_and_score(new_signs)
                if valid and new_radius < current_radius:
                    current_signs = new_signs
                    current_radius = new_radius
                    improved = True
                    if time.time() > time_limit:
                        break
        
        return current_signs, current_radius
    
    # Perform local search with time limit
    time_limit = min(start_time + 50, deadline - 10)
    best_signs, best_radius = local_search(best_signs, best_radius, time_limit)
    
    # If we still have time, try more sophisticated approaches
    if time.time() < deadline - 20:
        # Try pattern-based initialization
        # For prism graphs, certain sign patterns may be optimal
        # Try alternating signs on cycles
        
        patterns = []
        
        # Pattern 1: All matching edges +1, cycle edges alternating
        for phase in range(2):
            signs = np.ones(num_edges, dtype=int)
            for i in range(m):
                # Top cycle edge i
                idx = edge_idx.get((i, (i+1)%m), edge_idx.get(((i+1)%m, i)))
                signs[idx] = 1 if (i + phase) % 2 == 0 else -1
                # Bottom cycle edge i
                idx = edge_idx.get((m+i, m+(i+1)%m), edge_idx.get((m+(i+1)%m, m+i)))
                signs[idx] = 1 if (i + phase) % 2 == 0 else -1
            patterns.append(signs)
        
        # Pattern 2: All cycle edges +1, matching edges alternating
        for phase in range(2):
            signs = np.ones(num_edges, dtype=int)
            for i in range(m):
                idx = edge_idx.get((i, m+i), edge_idx.get((m+i, i)))
                signs[idx] = 1 if (i + phase) % 2 == 0 else -1
            patterns.append(signs)
        
        # Try each pattern
        for signs in patterns:
            valid, radius = check_and_score(signs)
            if valid and radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
                # Refine with local search
                if time.time() < deadline - 15:
                    time_limit = min(start_time + 60, deadline - 10)
                    best_signs, best_radius = local_search(best_signs, best_radius, time_limit)
        
        # If still time, try more random restarts
        if time.time() < deadline - 15:
            for _ in range(100):
                if time.time() > deadline - 15:
                    break
                signs = rng.choice([-1, 1], size=num_edges)
                valid, radius = check_and_score(signs)
                if valid and radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
                    # Quick local search
                    if time.time() < deadline - 10:
                        time_limit = min(start_time + 65, deadline - 5)
                        best_signs, best_radius = local_search(best_signs, best_radius, time_limit)
    
    return best_signs

# Keep the original function name and interface
def sign_edges(graph, budget):
    """
    Assign signs (-1 or +1) to edges of a prism graph to minimize the spectral radius
    of the signed adjacency matrix, subject to the lift being connected and 3-regular.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin before 90s kill
    
    m = graph['n'] // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build edge index for quick lookup
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    def build_signed_adjacency(signs):
        """Build the signed adjacency matrix."""
        n = graph['n']
        A_s = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            A_s[u][v] = s
            A_s[v][u] = s
        return A_s
    
    def spectral_radius(signs):
        """Compute the spectral radius of the signed adjacency matrix."""
        A_s = build_signed_adjacency(signs)
        eigvals = np.linalg.eigvalsh(A_s)
        return np.max(np.abs(eigvals))
    
    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular."""
        n = graph['n']
        adj = [[] for _ in range(2 * n)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                adj[u].append(v)
                adj[v].append(u)
                adj[n + u].append(n + v)
                adj[n + v].append(n + u)
            else:
                adj[u].append(n + v)
                adj[n + v].append(u)
                adj[n + u].append(v)
                adj[v].append(n + u)
        
        # Check 3-regularity
        for v in range(2 * n):
            if len(adj[v]) != 3:
                return False
        
        # Check connectivity using BFS
        visited = [False] * (2 * n)
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        
        return count == 2 * n
    
    def check_and_score(signs):
        """Check validity and compute spectral radius. Returns (valid, radius)."""
        if not is_valid_lift(signs):
            return False, float('inf')
        return True, spectral_radius(signs)
    
    # Initial solution: use the seed's approach
    rng = np.random.default_rng(3)
    best_signs = None
    best_radius = float('inf')
    
    # Try the seed's random sampling
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=num_edges)
        valid, radius = check_and_score(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(num_edges, dtype=int)
        best_radius = spectral_radius(best_signs)
    
    # Local search: flip one edge at a time
    def local_search(signs, radius, time_limit):
        """Perform local search by flipping individual edges."""
        current_signs = signs.copy()
        current_radius = radius
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            for i in range(num_edges):
                new_signs = current_signs.copy()
                new_signs[i] = -new_signs[i]
                valid, new_radius = check_and_score(new_signs)
                if valid and new_radius < current_radius:
                    current_signs = new_signs
                    current_radius = new_radius
                    improved = True
                    if time.time() > time_limit:
                        break
        
        return current_signs, current_radius
    
    # Perform local search with time limit
    time_limit = min(start_time + 50, deadline - 10)
    best_signs, best_radius = local_search(best_signs, best_radius, time_limit)
    
    # If we still have time, try pattern-based initialization
    if time.time() < deadline - 20:
        patterns = []
        
        # Pattern 1: All matching edges +1, cycle edges alternating
        for phase in range(2):
            signs = np.ones(num_edges, dtype=int)
            for i in range(m):
                idx = edge_idx.get((i, (i+1)%m), edge_idx.get(((i+1)%m, i)))
                signs[idx] = 1 if (i + phase) % 2 == 0 else -1
                idx = edge_idx.get((m+i, m+(i+1)%m), edge_idx.get((m+(i+1)%m, m+i)))
                signs[idx] = 1 if (i + phase) % 2 == 0 else -1
            patterns.append(signs)
        
        # Pattern 2: All cycle edges +1, matching edges alternating
        for phase in range(2):
            signs = np.ones(num_edges, dtype=int)
            for i in range(m):
                idx = edge_idx.get((i, m+i), edge_idx.get((m+i, i)))
                signs[idx] = 1 if (i + phase) % 2 == 0 else -1
            patterns.append(signs)
        
        # Try each pattern
        for signs in patterns:
            valid, radius = check_and_score(signs)
            if valid and radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
                if time.time() < deadline - 15:
                    time_limit = min(start_time + 60, deadline - 10)
                    best_signs, best_radius = local_search(best_signs, best_radius, time_limit)
        
        # If still time, try more random restarts
        if time.time() < deadline - 15:
            for _ in range(100):
                if time.time() > deadline - 15:
                    break
                signs = rng.choice([-1, 1], size=num_edges)
                valid, radius = check_and_score(signs)
                if valid and radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
                    if time.time() < deadline - 10:
                        time_limit = min(start_time + 65, deadline - 5)
                        best_signs, best_radius = local_search(best_signs, best_radius, time_limit)
    
    return best_signs
