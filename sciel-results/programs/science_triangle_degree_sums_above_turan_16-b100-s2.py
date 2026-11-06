"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _compute_score(n, edges):
    """Compute the maximum triangle degree sum."""
    if not edges:
        return 0
    # Build adjacency sets and degree counts
    adj = [set() for _ in range(n)]
    deg = [0] * n
    edge_set = set()
    for u, v in edges:
        if u == v:
            continue
        if (v, u) in edge_set:
            continue
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    # Find all triangles
    max_sum = 0
    # For each edge (u, v), find common neighbors w
    # To avoid duplicates, only consider w > v (or some ordering)
    for u, v in edge_set:
        if u > v:
            continue
        common = adj[u] & adj[v]
        for w in common:
            s = deg[u] + deg[v] + deg[w]
            if s > max_sum:
                max_sum = s
    return max_sum

class _GraphState:
    """Maintain graph state with incremental triangle tracking for fast score updates."""
    
    def __init__(self, n, edge_set):
        self.n = n
        self.edge_set = set()
        self.adj = [set() for _ in range(n)]
        self.deg = [0] * n
        # triangle_sums: dict mapping frozenset({u,v,w}) -> sum of degrees
        # But degrees change, so we need to recompute affected triangle sums
        # Instead, track triangles as list of (u,v,w) and recompute max when needed
        self.triangles = []  # list of (u, v, w)
        self._rebuild()
        for u, v in edge_set:
            self.add_edge(u, v)
    
    def _rebuild(self):
        self.edge_set = set()
        self.adj = [set() for _ in range(self.n)]
        self.deg = [0] * self.n
        self.triangles = []
    
    def add_edge(self, u, v):
        if u == v:
            return
        if (u, v) in self.edge_set or (v, u) in self.edge_set:
            return
        self.edge_set.add((u, v))
        self.adj[u].add(v)
        self.adj[v].add(u)
        self.deg[u] += 1
        self.deg[v] += 1
        # Find new triangles: common neighbors of u and v
        common = self.adj[u] & self.adj[v]
        for w in common:
            self.triangles.append((u, v, w))
    
    def remove_edge(self, u, v):
        if (u, v) not in self.edge_set and (v, u) not in self.edge_set:
            return
        self.edge_set.discard((u, v))
        self.edge_set.discard((v, u))
        self.adj[u].discard(v)
        self.adj[v].discard(u)
        self.deg[u] -= 1
        self.deg[v] -= 1
        # Remove triangles that used this edge
        self.triangles = [t for t in self.triangles if not ((t[0] == u and t[1] == v) or 
                                                              (t[0] == v and t[1] == u) or 
                                                              (t[0] == u and t[2] == v) or 
                                                              (t[0] == v and t[2] == u) or 
                                                              (t[1] == u and t[2] == v) or 
                                                              (t[1] == v and t[2] == u))]
    
    def get_score(self):
        if not self.triangles:
            return 0
        max_sum = 0
        for u, v, w in self.triangles:
            s = self.deg[u] + self.deg[v] + self.deg[w]
            if s > max_sum:
                max_sum = s
        return max_sum
    
    def get_edges(self):
        return list(self.edge_set)

def _construct_baseline():
    """K_{8,8} plus one edge inside a part."""
    edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    return edges

def _random_local_search(n, target_edges, max_iter, deadline):
    """Local search using incremental graph state for fast score evaluation."""
    start_time = time.time()
    
    # Start from baseline
    edges = _construct_baseline()
    edges_set = set()
    for u, v in edges:
        edges_set.add((min(u, v), max(u, v)))
    
    state = _GraphState(n, edges_set)
    current_score = state.get_score()
    best_score = current_score
    best_edges = set(state.edge_set)
    
    # Precompute all possible edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        existing_edges = list(state.edge_set)
        if not existing_edges:
            break
        
        # Try several swaps per iteration
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Pick random edge to remove
            u1, v1 = random.choice(existing_edges)
            # Pick random edge to add
            u2, v2 = random.choice(all_possible)
            
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                continue
            if (u2, v2) in state.edge_set or (v2, u2) in state.edge_set:
                continue
            
            # Perform swap
            state.remove_edge(u1, v1)
            state.add_edge(u2, v2)
            
            new_score = state.get_score()
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_edges = set(state.edge_set)
                # Keep the swap
            else:
                # Revert
                state.remove_edge(u2, v2)
                state.add_edge(u1, v1)
    
    return list(best_edges)

def _simulated_annealing(n, target_edges, max_iter, deadline):
    """Simulated annealing using incremental graph state."""
    start_time = time.time()
    
    # Start from baseline
    edges = _construct_baseline()
    edges_set = set()
    for u, v in edges:
        edges_set.add((min(u, v), max(u, v)))
    
    state = _GraphState(n, edges_set)
    current_score = state.get_score()
    best_score = current_score
    best_edges = set(state.edge_set)
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        existing_edges = list(state.edge_set)
        if not existing_edges:
            break
        
        u1, v1 = random.choice(existing_edges)
        u2, v2 = random.choice(all_possible)
        
        if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
            continue
        if (u2, v2) in state.edge_set or (v2, u2) in state.edge_set:
            continue
        
        # Perform swap
        state.remove_edge(u1, v1)
        state.add_edge(u2, v2)
        
        new_score = state.get_score()
        delta = new_score - current_score
        
        import math
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = set(state.edge_set)
        else:
            # Revert
            state.remove_edge(u2, v2)
            state.add_edge(u1, v1)
        
        T *= cooling
    
    return list(best_edges)

def construct():
    """Construct a graph on 16 vertices with 65 edges minimizing max triangle degree sum."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit (150s kill)
    
    n = 16
    target_edges = 65
    
    # Start with baseline
    best_edges = _construct_baseline()
    best_score = _compute_score(n, best_edges)
    
    # Run local search
    if time.time() < deadline:
        ls_edges = _random_local_search(n, target_edges, max_iter=8000, deadline=deadline)
        ls_score = _compute_score(n, ls_edges)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
    
    # Run simulated annealing
    if time.time() < deadline:
        sa_edges = _simulated_annealing(n, target_edges, max_iter=15000, deadline=deadline)
        sa_score = _compute_score(n, sa_edges)
        if sa_score < best_score:
            best_score = sa_score
            best_edges = sa_edges
    
    # Second round: local search from best so far
    if time.time() < deadline - 5:
        ls_edges2 = _random_local_search(n, target_edges, max_iter=5000, deadline=deadline)
        ls_score2 = _compute_score(n, ls_edges2)
        if ls_score2 < best_score:
            best_score = ls_score2
            best_edges = ls_edges2
    
    # Final verification: ensure exactly 65 edges
    edge_set = set()
    for u, v in best_edges:
        edge_set.add((min(u, v), max(u, v)))
    
    if len(edge_set) != target_edges:
        # Fall back to baseline
        best_edges = _construct_baseline()
        edge_set = set()
        for u, v in best_edges:
            edge_set.add((min(u, v), max(u, v)))
    
    return {'n': n, 'edges': [list(e) for e in sorted(edge_set)]}
# EVOLVE-BLOCK-END
