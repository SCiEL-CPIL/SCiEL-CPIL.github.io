"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit

    n = 40
    best_edges = 0
    best_edge_list = None

    def build_neighbors(adj):
        neighbors = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
        return neighbors

    def get_edge_list(adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i][j]]

    def try_add_edge(adj, neighbors, u, v):
        """Try to add edge (u,v). Return True if successful (no C6 created)."""
        if adj[u][v]:
            return False
        # Check if adding (u,v) creates a C6
        # A C6 is created iff there's a path of length 5 from u to v in current graph
        # Path: u-a-b-c-d-v where all 6 vertices are distinct
        nu = neighbors[u]
        nv = neighbors[v]
        for a in nu:
            for d in nv:
                if d == a:
                    continue
                for b in neighbors[a]:
                    if b == u or b == d:
                        continue
                    for c in neighbors[d]:
                        if c == v or c == a or c == b:
                            continue
                        if adj[b][c]:
                            # Check all distinct
                            if len({u, a, b, c, d, v}) == 6:
                                return False
        # Safe to add
        adj[u][v] = adj[v][u] = True
        neighbors[u].add(v)
        neighbors[v].add(u)
        return True

    def try_remove_edge(adj, neighbors, u, v):
        """Remove edge (u,v)."""
        if not adj[u][v]:
            return
        adj[u][v] = adj[v][u] = False
        neighbors[u].discard(v)
        neighbors[v].discard(u)

    def run_greedy_add(adj, neighbors, deadline):
        """Repeatedly try to add edges until no more can be added or deadline."""
        current_edges = int(np.sum(adj)) // 2
        # Build list of non-edges
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not adj[i][j]]
        random.shuffle(non_edges)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(non_edges)
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                if not adj[u][v]:
                    if try_add_edge(adj, neighbors, u, v):
                        current_edges += 1
                        improved = True
        return current_edges

    def run_swap_search(adj, neighbors, deadline, max_iters=5000):
        """Try swapping: remove one edge, try to add two edges (net +1)."""
        current_edges = int(np.sum(adj)) // 2
        best_local = current_edges
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Get current edges
            edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]]
            if not edge_list:
                break
            
            # Pick random edge to remove
            ru, rv = random.choice(edge_list)
            try_remove_edge(adj, neighbors, ru, rv)
            current_edges -= 1
            
            # Try to add as many edges as possible
            non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not adj[i][j]]
            random.shuffle(non_edges)
            added = 0
            for u, v in non_edges:
                if try_add_edge(adj, neighbors, u, v):
                    current_edges += 1
                    added += 1
                    if added >= 2:
                        break
            
            if current_edges > best_local:
                best_local = current_edges
            
            # If no net gain, revert
            if current_edges <= best_local:
                # Restore the removed edge
                adj[ru][rv] = adj[rv][ru] = True
                neighbors[ru].add(rv)
                neighbors[rv].add(ru)
                current_edges += 1
                # Remove any edges we added
                # We need to track which edges we added... let's just rebuild
                # Actually, let's be more careful
                # We added up to 2 edges. Let's just restore state properly.
                # This is getting complex. Let's use a simpler approach:
                # Save state before, restore if no improvement.
        
        return best_local

    # Better swap search with proper state management
    def run_swap_search_v2(adj, neighbors, deadline, max_iters=3000):
        """Try swapping with proper state save/restore."""
        current_edges = int(np.sum(adj)) // 2
        best_local = current_edges
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Save state
            adj_save = adj.copy()
            neighbors_save = [s.copy() for s in neighbors]
            
            # Get current edges
            edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]]
            if not edge_list:
                break
            
            # Pick random edge to remove
            ru, rv = random.choice(edge_list)
            try_remove_edge(adj, neighbors, ru, rv)
            
            # Try to add edges greedily
            non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not adj[i][j]]
            random.shuffle(non_edges)
            for u, v in non_edges:
                try_add_edge(adj, neighbors, u, v)
            
            new_edges = int(np.sum(adj)) // 2
            
            if new_edges > best_local:
                best_local = new_edges
                current_edges = new_edges
                # Keep this state
            else:
                # Restore state
                adj[:] = adj_save
                neighbors = [s.copy() for s in neighbors_save]
        
        return best_local

    # Try multiple initializations
    initializations = []
    
    # Init 1: 8 disjoint K5's
    adj1 = np.zeros((n, n), dtype=bool)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                adj1[5 * k + a][5 * k + b] = adj1[5 * k + b][5 * k + a] = True
    initializations.append(adj1.copy())
    
    # Init 2: Random graph with ~100 edges
    for trial in range(3):
        adj_rand = np.zeros((n, n), dtype=bool)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        for i, j in all_pairs[:100]:
            adj_rand[i][j] = adj_rand[j][i] = True
        initializations.append(adj_rand)
    
    # Init 3: Perturb the K5 baseline
    adj3 = adj1.copy()
    for _ in range(10):
        i, j = random.sample(range(n), 2)
        if adj3[i][j]:
            adj3[i][j] = adj3[j][i] = False
        else:
            adj3[i][j] = adj3[j][i] = True
    initializations.append(adj3)

    # Try each initialization
    for init_adj in initializations:
        if time.time() > DEADLINE - 5:
            break
        
        adj = init_adj.copy()
        neighbors = build_neighbors(adj)
        
        # Phase 1: Greedy add
        current_edges = run_greedy_add(adj, neighbors, DEADLINE - 5)
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_edge_list = get_edge_list(adj)
        
        # Phase 2: Swap search
        if time.time() < DEADLINE - 3:
            neighbors = build_neighbors(adj)
            current_edges = run_swap_search_v2(adj, neighbors, DEADLINE - 3)
            if current_edges > best_edges:
                best_edges = current_edges
                best_edge_list = get_edge_list(adj)
        
        # Phase 3: Final greedy add
        if time.time() < DEADLINE - 1:
            neighbors = build_neighbors(adj)
            current_edges = run_greedy_add(adj, neighbors, DEADLINE - 1)
            if current_edges > best_edges:
                best_edges = current_edges
                best_edge_list = get_edge_list(adj)

    # Fallback
    if best_edge_list is None:
        best_edge_list = []
        for k in range(8):
            best_edge_list += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]

    return best_edge_list
# EVOLVE-BLOCK-END
