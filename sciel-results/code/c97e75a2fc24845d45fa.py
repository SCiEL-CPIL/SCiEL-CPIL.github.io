"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import math

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
        # triangles: list of (u, v, w) where u < v < w to ensure uniqueness
        self.triangles = []
        
        # Add edges to build initial state
        # We need to add edges in a way that we can detect triangles
        # Since we are building from scratch, we can just add all and then find triangles
        for u, v in edge_set:
            self._add_edge_internal(u, v)
        
        # Rebuild triangles from scratch to be safe and sorted
        self._rebuild_triangles()

    def _add_edge_internal(self, u, v):
        if u == v:
            return
        if (u, v) in self.edge_set or (v, u) in self.edge_set:
            return
        self.edge_set.add((min(u, v), max(u, v)))
        self.adj[u].add(v)
        self.adj[v].add(u)
        self.deg[u] += 1
        self.deg[v] += 1

    def _rebuild_triangles(self):
        """Rebuild the list of triangles from current adjacency."""
        self.triangles = []
        for u in range(self.n):
            for v in self.adj[u]:
                if v > u:
                    common = self.adj[u] & self.adj[v]
                    for w in common:
                        if w > v:
                            self.triangles.append((u, v, w))
        # Sort triangles by degree sum descending to help find max quickly?
        # Or just keep them as is.
        
    def add_edge(self, u, v):
        if u == v:
            return False
        if (u, v) in self.edge_set or (v, u) in self.edge_set:
            return False
            
        self.edge_set.add((min(u, v), max(u, v)))
        self.adj[u].add(v)
        self.adj[v].add(u)
        self.deg[u] += 1
        self.deg[v] += 1
        
        # Find new triangles involving (u, v)
        # A new triangle is formed if there is a w connected to both u and v
        common = self.adj[u] & self.adj[v]
        new_tris = []
        for w in common:
            # Ensure unique representation u < v < w
            a, b, c = sorted((u, v, w))
            new_tris.append((a, b, c))
        
        self.triangles.extend(new_tris)
        return True

    def remove_edge(self, u, v):
        if (u, v) not in self.edge_set and (v, u) not in self.edge_set:
            return False
            
        self.edge_set.discard((u, v))
        self.edge_set.discard((v, u))
        self.adj[u].discard(v)
        self.adj[v].discard(u)
        self.deg[u] -= 1
        self.deg[v] -= 1
        
        # Remove triangles that use edge (u, v)
        # A triangle (a, b, c) uses (u, v) if {u, v} is a subset of {a, b, c}
        # Since u, v are specific, we check if u and v are in the triangle
        # Optimization: We can filter the list
        self.triangles = [t for t in self.triangles if not ((u in t and v in t))]
        return True

    def get_score_and_witness(self):
        """Returns (max_score, list_of_witness_triangles)."""
        if not self.triangles:
            return 0, []
        
        max_sum = 0
        witnesses = []
        for u, v, w in self.triangles:
            s = self.deg[u] + self.deg[v] + self.deg[w]
            if s > max_sum:
                max_sum = s
                witnesses = [(u, v, w)]
            elif s == max_sum:
                witnesses.append((u, v, w))
        return max_sum, witnesses

    def get_edges(self):
        return list(self.edge_set)

def _construct_baseline():
    """K_{8,8} plus one edge inside a part."""
    edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    return edges

def _targeted_local_search(n, target_edges, max_iter, deadline):
    """
    Local search that prioritizes removing edges from the current worst triangle.
    """
    start_time = time.time()
    
    # Start from baseline
    edges = _construct_baseline()
    edges_set = set()
    for u, v in edges:
        edges_set.add((min(u, v), max(u, v)))
    
    state = _GraphState(n, edges_set)
    current_score, current_witnesses = state.get_score_and_witness()
    best_score = current_score
    best_edges = set(state.edge_set)
    
    # Precompute all possible edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Track recent moves to avoid immediate oscillation
    recent_moves = []
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # 1. Identify candidate edges to remove
        # Strategy: Pick an edge from one of the worst triangles
        # If no triangles, pick random edge
        candidate_edges_to_remove = []
        if current_witnesses:
            # Pick a random witness triangle
            tri = random.choice(current_witnesses)
            # The edges in this triangle are (tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])
            candidate_edges_to_remove = [
                (min(tri[0], tri[1]), max(tri[0], tri[1])),
                (min(tri[1], tri[2]), max(tri[1], tri[2])),
                (min(tri[0], tri[2]), max(tri[0], tri[2]))
            ]
        else:
            candidate_edges_to_remove = list(state.edge_set)
            
        if not candidate_edges_to_remove:
            break
            
        # 2. Pick an edge to remove (randomly from candidates, or cycle through)
        # To diversify, we can pick randomly from the candidate list
        u1, v1 = random.choice(candidate_edges_to_remove)
        
        # Ensure we are not undoing a very recent move (optional, but helps)
        # For now, just proceed
        
        # 3. Pick an edge to add
        # Strategy: Pick a random edge not in the graph.
        # We can bias towards edges that connect low-degree vertices to balance degrees?
        # Or just random.
        u2, v2 = random.choice(all_possible)
        
        if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
            continue
        if (u2, v2) in state.edge_set or (v2, u2) in state.edge_set:
            continue
        
        # 4. Perform swap
        state.remove_edge(u1, v1)
        state.add_edge(u2, v2)
        
        new_score, new_witnesses = state.get_score_and_witness()
        
        # 5. Acceptance
        # Greedy: Accept if better.
        # To escape local minima, we can accept worse moves with small probability.
        # But since we want to be fast and effective, let's stick to strict improvement 
        # or a very small random walk if stuck.
        
        if new_score < current_score:
            current_score = new_score
            current_witnesses = new_witnesses
            if new_score < best_score:
                best_score = new_score
                best_edges = set(state.edge_set)
            # Reset recent moves history on success?
            # recent_moves = []
        else:
            # Revert
            state.remove_edge(u2, v2)
            state.add_edge(u1, v1)
            
            # If we are stuck (score didn't improve), we might want to force a random move
            # to escape. However, pure random walk is slow.
            # Let's just continue.
            
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
    current_score, _ = state.get_score_and_witness()
    best_score = current_score
    best_edges = set(state.edge_set)
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    T = 5.0
    T_min = 0.1
    cooling = 0.99
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick random edge to remove
        existing_edges = list(state.edge_set)
        if not existing_edges:
            break
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
        
        new_score, _ = state.get_score_and_witness()
        delta = new_score - current_score
        
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
        if T < T_min:
            T = T_min
    
    return list(best_edges)

def construct():
    """Construct a graph on 16 vertices with 65 edges minimizing max triangle degree sum."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n = 16
    target_edges = 65
    
    # Start with baseline
    best_edges = _construct_baseline()
    best_score = _compute_score(n, best_edges)
    
    # 1. Run Targeted Local Search (Greedy)
    # This should quickly fix the "bad" triangles in the baseline
    if time.time() < deadline:
        ls_edges = _targeted_local_search(n, target_edges, max_iter=2000, deadline=deadline)
        ls_score = _compute_score(n, ls_edges)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
    
    # 2. Run Simulated Annealing
    # Start from the best result of the local search
    # We need to re-initialize state from best_edges for SA
    if time.time() < deadline:
        # Create a state from best_edges to start SA
        # We can just call SA which starts from baseline, but we want to start from best_edges
        # Let's modify SA to accept initial edges, or just run it and compare.
        # For simplicity and speed, let's run SA from the best found so far.
        
        # Re-initialize SA with best_edges
        edges_set = set()
        for u, v in best_edges:
            edges_set.add((min(u, v), max(u, v)))
            
        state = _GraphState(n, edges_set)
        current_score, _ = state.get_score_and_witness()
        sa_best_score = current_score
        sa_best_edges = set(state.edge_set)
        
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        T = 3.0
        T_min = 0.1
        cooling = 0.995
        
        # Run SA for remaining time
        while time.time() < deadline:
            existing_edges = list(state.edge_set)
            if not existing_edges:
                break
            
            u1, v1 = random.choice(existing_edges)
            u2, v2 = random.choice(all_possible)
            
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                continue
            if (u2, v2) in state.edge_set or (v2, u2) in state.edge_set:
                continue
            
            state.remove_edge(u1, v1)
            state.add_edge(u2, v2)
            
            new_score, _ = state.get_score_and_witness()
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_score = new_score
                if new_score < sa_best_score:
                    sa_best_score = new_score
                    sa_best_edges = set(state.edge_set)
            else:
                state.remove_edge(u2, v2)
                state.add_edge(u1, v1)
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        if sa_best_score < best_score:
            best_score = sa_best_score
            best_edges = sa_best_edges

    # Final verification
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
