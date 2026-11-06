import sys
import time
import random
import numpy as np
from collections import deque

# Set recursion limit for deep DFS
sys.setrecursionlimit(10000)

class HamiltonianSolver:
    def __init__(self, n, edge_order):
        self.n = n
        self.edge_order = edge_order
        self.edge_index = {}
        for i, (u, v) in enumerate(edge_order):
            self.edge_index[(u, v)] = i + 1
            self.edge_index[(v, u)] = i + 1

    def _build_adjacency(self, k):
        """Build adjacency list for prefix k, sorted by edge arrival index."""
        adj = [[] for _ in range(self.n)]
        for i in range(k):
            u, v = self.edge_order[i]
            adj[u].append((v, i + 1))
            adj[v].append((u, i + 1))
        # Sort neighbors by arrival index (earlier edges first)
        for i in range(self.n):
            adj[i].sort(key=lambda x: x[1])
        return adj

    def _check_min_degree(self, adj):
        """Check if all vertices have degree >= 2."""
        for i in range(self.n):
            if len(adj[i]) < 2:
                return False
        return True

    def _check_connectivity(self, adj):
        """Check if graph is connected using BFS."""
        visited = [False] * self.n
        queue = deque([0])
        visited[0] = True
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v, _ in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return count == self.n

    def _dfs_find_cycle(self, adj, timeout=5.0):
        """Find Hamiltonian cycle using DFS with pruning."""
        start_time = time.time()
        n = self.n
        
        # Precompute adjacency as sets for fast lookup
        adj_sets = [set(v for v, _ in adj[i]) for i in range(n)]
        # Also keep sorted list version
        adj_lists = [sorted(v for v, _ in adj[i]) for i in range(n)]
        
        # Try multiple random orderings
        for attempt in range(20):
            if time.time() - start_time > timeout:
                break
                
            # Create shuffled adjacency lists
            shuffled_adj = [list(adj_lists[i]) for i in range(n)]
            for i in range(n):
                random.shuffle(shuffled_adj[i])
            
            path = [0]
            visited = [False] * n
            visited[0] = True
            
            def dfs(u, depth):
                if time.time() - start_time > timeout:
                    return False
                
                if depth == n:
                    return 0 in adj_sets[u]
                
                # Warnsdorff heuristic: prioritize vertices with fewer unvisited neighbors
                candidates = []
                for v in shuffled_adj[u]:
                    if not visited[v]:
                        # Count unvisited neighbors of v
                        unvisited_count = sum(1 for w in adj_sets[v] if not visited[w])
                        # If v is not the last node, it must have at least 1 unvisited neighbor
                        # OR it must connect to 0 if it's the last step
                        if depth == n - 1:
                            # Last node: must connect to 0
                            if 0 in adj_sets[v]:
                                candidates.append((0, v))  # Priority 0: must take this
                        else:
                            if unvisited_count >= 1:
                                candidates.append((unvisited_count, v))
                
                # Sort by priority (fewer unvisited neighbors first - Warnsdorff)
                candidates.sort(key=lambda x: x[0])
                
                for _, v in candidates:
                    if not visited[v]:
                        path.append(v)
                        visited[v] = True
                        if dfs(v, depth + 1):
                            return True
                        path.pop()
                        visited[v] = False
                        
                return False
            
            if dfs(0, 1):
                return path
        
        return None

    def _two_opt_improve(self, cycle, adj, max_iter=100):
        """Try to improve cycle using 2-opt moves to reduce max edge index."""
        n = self.n
        if cycle is None:
            return None
            
        # Build edge index lookup
        edge_idx = {}
        for i, (u, v) in enumerate(self.edge_order):
            edge_idx[(u, v)] = i + 1
            edge_idx[(v, u)] = i + 1
        
        # Get current max edge index
        def get_max_edge(cyc):
            max_idx = 0
            for i in range(n):
                u = cyc[i]
                v = cyc[(i + 1) % n]
                idx = edge_idx.get((u, v), float('inf'))
                if idx > max_idx:
                    max_idx = idx
            return max_idx
        
        current_max = get_max_edge(cycle)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            
            # Try all 2-opt moves
            for i in range(n):
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue  # Skip reversing entire cycle (same cycle)
                    
                    # Create new cycle by reversing segment [i+1..j]
                    new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                    
                    # Check if all edges exist in graph
                    valid = True
                    new_max = 0
                    for k in range(n):
                        u = new_cycle[k]
                        v = new_cycle[(k + 1) % n]
                        if (u, v) not in edge_idx:
                            valid = False
                            break
                        idx = edge_idx[(u, v)]
                        if idx > new_max:
                            new_max = idx
                        if new_max >= current_max:
                            break
                    
                    if valid and new_max < current_max:
                        cycle = new_cycle
                        current_max = new_max
                        improved = True
        
        return cycle

    def get_cycle(self, k, timeout=5.0):
        # Build adjacency list
        adj = self._build_adjacency(k)
        
        # Quick checks
        if not self._check_min_degree(adj):
            return None
        
        if not self._check_connectivity(adj):
            return None
        
        # Find cycle using DFS
        cycle = self._dfs_find_cycle(adj, timeout=timeout)
        
        if cycle is not None:
            # Try to improve with 2-opt
            cycle = self._two_opt_improve(cycle, adj)
        
        return cycle

# Global variables for the task
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

solver = None

def construct():
    global solver
    start_time = time.time()
    deadline = start_time + 80.0  # 80s margin (evaluator kills at 90s)
    
    if solver is None:
        solver = HamiltonianSolver(P['n'], P['edge_order'])
    
    n = P['n']
    min_k = n
    max_k = len(P['edge_order'])
    
    # Binary search for minimal K
    low = min_k
    high = max_k
    best_cycle = None
    best_k = max_k
    
    # First, do a quick linear scan from low to find a feasible k
    # This helps us get an initial solution quickly
    for k in range(min_k, min(min_k + 100, max_k + 1)):
        if time.time() > deadline:
            break
        cycle = solver.get_cycle(k, timeout=2.0)
        if cycle is not None:
            best_cycle = cycle
            best_k = k
            break
    
    # If we found a solution, use binary search to refine
    if best_cycle is not None:
        high = best_k - 1
        while low <= high:
            if time.time() > deadline:
                break
                
            mid = (low + high) // 2
            cycle = solver.get_cycle(mid, timeout=3.0)
            
            if cycle is not None:
                best_cycle = cycle
                best_k = mid
                high = mid - 1
            else:
                low = mid + 1
    else:
        # No solution found in quick scan, try binary search on full range
        while low <= high:
            if time.time() > deadline:
                break
                
            mid = (low + high) // 2
            cycle = solver.get_cycle(mid, timeout=3.0)
            
            if cycle is not None:
                best_cycle = cycle
                best_k = mid
                high = mid - 1
            else:
                low = mid + 1
    
    if best_cycle is None:
        # Fallback: return identity cycle if we couldn't find a better one
        return list(range(n))
        
    return best_cycle
