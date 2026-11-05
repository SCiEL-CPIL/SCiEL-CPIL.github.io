"""K2,3-free graph on 20 vertices: maximize edges via greedy + local search."""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # Strategy 1: Multiple greedy restarts with fast common-neighbor tracking
    for seed in range(3000):
        if time.time() > deadline - 15.0:
            break
        
        edges = _greedy_construct_fast(n, seed)
        score = len(edges)
        
        if score > best_score:
            best_score = score
            best_edges = edges[:]
    
    # Strategy 2: Local search from best greedy result
    if time.time() < deadline - 10.0 and best_edges:
        improved = _local_search_fast(n, best_edges, deadline)
        if len(improved) > best_score:
            best_score = len(improved)
            best_edges = improved
    
    # Strategy 3: More greedy restarts with heuristic, then local search
    for seed in range(3000, 8000):
        if time.time() > deadline - 5.0:
            break
        
        edges = _greedy_construct_heuristic_fast(n, seed)
        score = len(edges)
        
        if score > best_score:
            best_score = score
            best_edges = edges[:]
        
        # Quick local search on promising results
        if score >= best_score - 3 and time.time() < deadline - 5.0:
            improved = _local_search_fast(n, edges, deadline)
            if len(improved) > best_score:
                best_score = len(improved)
                best_edges = improved[:]
    
    return best_edges


class K23Tracker:
    """Fast K2,3-freeness tracker using common-neighbor counts."""
    
    def __init__(self, n):
        self.n = n
        self.adj = [set() for _ in range(n)]
        # cn[u][v] = number of common neighbors of u and v
        self.cn = [[0] * n for _ in range(n)]
        self.edge_count = 0
    
    def can_add(self, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # After adding (u,v):
        # - cn[u][v] stays the same (they already have cn[u][v] common neighbors)
        #   Actually, u and v are not yet adjacent, so their common neighbors
        #   are exactly cn[u][v]. Adding the edge doesn't change common neighbors.
        #   But we need cn[u][v] < 3.
        if self.cn[u][v] >= 3:
            return False
        
        # For each w in adj[u]: cn[v][w] increases by 1 (because u becomes a common neighbor of v and w)
        #   We need cn[v][w] + 1 < 3, i.e., cn[v][w] < 2
        for w in self.adj[u]:
            if w == v:
                continue
            if self.cn[v][w] >= 2:
                return False
        
        # For each w in adj[v]: cn[u][w] increases by 1
        #   We need cn[u][w] + 1 < 3, i.e., cn[u][w] < 2
        for w in self.adj[v]:
            if w == u:
                continue
            if self.cn[u][w] >= 2:
                return False
        
        return True
    
    def add(self, u, v):
        """Add edge (u,v) and update common-neighbor counts."""
        self.adj[u].add(v)
        self.adj[v].add(u)
        self.edge_count += 1
        
        # Update cn: for each w in adj[u], cn[v][w] += 1 and cn[w][v] += 1
        #   (u is now a common neighbor of v and w)
        for w in self.adj[u]:
            if w == v:
                continue
            self.cn[v][w] += 1
            self.cn[w][v] += 1
        
        # For each w in adj[v], cn[u][w] += 1 and cn[w][u] += 1
        #   (v is now a common neighbor of u and w)
        for w in self.adj[v]:
            if w == u:
                continue
            self.cn[u][w] += 1
            self.cn[w][u] += 1
    
    def remove(self, u, v):
        """Remove edge (u,v) and update common-neighbor counts."""
        self.adj[u].discard(v)
        self.adj[v].discard(u)
        self.edge_count -= 1
        
        # For each w in adj[u] (after removal), cn[v][w] -= 1
        for w in self.adj[u]:
            if w == v:
                continue
            self.cn[v][w] -= 1
            self.cn[w][v] -= 1
        
        # For each w in adj[v] (after removal), cn[u][w] -= 1
        for w in self.adj[v]:
            if w == u:
                continue
            self.cn[u][w] -= 1
            self.cn[w][u] -= 1
    
    def get_edges(self):
        edges = []
        for u in range(self.n):
            for v in self.adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges


def _greedy_construct_fast(n, seed):
    """
    Greedy construction with fast K2,3 checking via common-neighbor counts.
    """
    random.seed(seed)
    
    tracker = K23Tracker(n)
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if tracker.can_add(u, v):
            tracker.add(u, v)
    
    return tracker.get_edges()


def _greedy_construct_heuristic_fast(n, seed):
    """
    Greedy with heuristic ordering using fast K2,3 checking.
    Prefer edges that keep more flexibility (lower degree vertices).
    """
    random.seed(seed)
    
    tracker = K23Tracker(n)
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    remaining = list(all_edges)
    random.shuffle(remaining)
    
    # Process in batches; within each batch, pick the "safest" edge
    while remaining and tracker.edge_count < 190:
        # Sample a subset for speed
        if len(remaining) > 80:
            candidates = random.sample(remaining, 80)
        else:
            candidates = remaining[:]
        
        best_edge = None
        best_score = -1
        
        for u, v in candidates:
            if not tracker.can_add(u, v):
                continue
            
            # Heuristic: prefer edges between lower-degree vertices
            score = -(len(tracker.adj[u]) + len(tracker.adj[v]))
            
            if score > best_score:
                best_score = score
                best_edge = (u, v)
        
        if best_edge is None:
            # No candidate can be added; try next batch
            remaining = [e for e in remaining if e not in set(candidates)]
            if not remaining:
                break
            continue
        
        u, v = best_edge
        tracker.add(u, v)
        remaining.remove((u, v))
    
    return tracker.get_edges()


def _local_search_fast(n, edges, deadline):
    """
    Local search with fast K2,3 checking: try to add missing edges,
    and if stuck, try removing some edges to unblock others.
    """
    tracker = K23Tracker(n)
    for u, v in edges:
        tracker.add(u, v)
    
    current_edges = tracker.get_edges()
    
    # Phase 1: Try to add all possible missing edges
    missing = [(u, v) for u in range(n) for v in range(u + 1, n) if v not in tracker.adj[u]]
    random.shuffle(missing)
    
    added = True
    while added and time.time() < deadline - 5.0:
        added = False
        new_missing = []
        for u, v in missing:
            if v in tracker.adj[u]:  # already added
                continue
            if tracker.can_add(u, v):
                tracker.add(u, v)
                added = True
            else:
                new_missing.append((u, v))
        missing = new_missing
    
    current_edges = tracker.get_edges()
    
    # Phase 2: Try removing one edge and re-adding to find better configurations
    for iteration in range(200):
        if time.time() > deadline - 5.0:
            break
        
        if not current_edges:
            break
        
        # Pick a random edge to remove
        idx = random.randint(0, len(current_edges) - 1)
        u, v = current_edges[idx]
        
        # Remove it
        tracker.remove(u, v)
        
        # Try to re-add all missing edges
        missing2 = [(a, b) for a in range(n) for b in range(a + 1, n) if b not in tracker.adj[a]]
        random.shuffle(missing2)
        
        for a, b in missing2:
            if tracker.can_add(a, b):
                tracker.add(a, b)
        
        current_edges = tracker.get_edges()
    
    return current_edges


# EVOLVE-BLOCK-END
