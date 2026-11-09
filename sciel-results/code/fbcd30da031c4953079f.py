"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict
from random import Random
import numpy as np

def construct():
    _t0 = time.time()
    _DEADLINE = 138.0

    def _instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def _build_conflict_matrix(edges, adj):
        """Build numpy boolean conflict matrix."""
        m = len(edges)
        C = np.zeros((m, m), dtype=bool)
        # vertex -> list of edge indices
        v_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            v_edges[u].append(i)
            v_edges[v].append(i)
        # adjacency via vertex
        for i, (u1, v1) in enumerate(edges):
            for j in v_edges[u1]:
                if j != i:
                    C[i, j] = True
                    C[j, i] = True
            for j in v_edges[v1]:
                if j != i:
                    C[i, j] = True
                    C[j, i] = True
            # joined by an edge: endpoints of i adjacent to endpoints of j
            for u2 in adj.get(u1, ()):
                for j in v_edges[u2]:
                    if j != i:
                        C[i, j] = True
                        C[j, i] = True
            for v2 in adj.get(v1, ()):
                for j in v_edges[v2]:
                    if j != i:
                        C[i, j] = True
                        C[j, i] = True
        return C

    def _greedy_with_cp(edges, C, order):
        """Greedy coloring with constraint propagation (forward checking)."""
        m = len(edges)
        color = [-1] * m
        # For each color, track which edges are "blocked" from using it
        # blocked[c][i] = True if edge i cannot use color c
        # We'll maintain: for each color c, the set of edges that conflict with some edge in c
        color_members = [set() for _ in range(m + 1)]  # up to m colors
        # Conflict count per edge per color: how many edges in color c conflict with edge i
        conflict_count = [np.zeros(m, dtype=np.int32) for _ in range(m + 1)]
        
        for i in order:
            # Find smallest color c where no member conflicts with i
            best_c = None
            for c in range(len(color_members)):
                # Check if any edge in color c conflicts with i
                if not np.any(C[i, list(color_members[c])]):
                    best_c = c
                    break
            if best_c is None:
                best_c = len(color_members)
                color_members.append(set())
                conflict_count.append(np.zeros(m, dtype=np.int32))
            
            color[i] = best_c
            color_members[best_c].add(i)
            # Update conflict counts: for all edges, mark that best_c is now blocked if they conflict with i
            conflict_count[best_c][C[i]] += 1
            # Actually we just need to check membership, the np.any above handles it
            # But for efficiency, let's track: for edge j, color c is unavailable if C[j,c_member] for some member
            # The np.any check is O(|color_members[c]|) which is fine for small graphs
        
        return color

    def _local_search(colour, C, time_limit=15.0):
        """
        Local search: repeatedly try to move an edge to a lower color.
        Uses numpy for fast conflict checking.
        """
        t_start = time.time()
        m = len(colour)
        col = colour[:]
        n_colors = max(col) + 1 if m > 0 else 0
        
        # Build color member lists
        color_members = defaultdict(list)
        for i, c in enumerate(col):
            color_members[c].append(i)
        
        # Precompute: for each edge, which colors are "full" (can't be used)
        # We'll recompute on the fly using numpy
        
        improved = True
        while improved:
            improved = False
            if time.time() - t_start > time_limit:
                break
            
            for i in range(m):
                cur_c = col[i]
                if cur_c == 0:
                    continue
                # Try to move to a lower color
                for c in range(cur_c):
                    members = color_members[c]
                    if not members:
                        # Empty color, always valid
                        color_members[cur_c].remove(i)
                        color_members[c].append(i)
                        col[i] = c
                        improved = True
                        break
                    # Check conflicts using numpy
                    member_arr = np.array(members, dtype=np.int32)
                    if not np.any(C[i, member_arr]):
                        color_members[cur_c].remove(i)
                        color_members[c].append(i)
                        col[i] = c
                        improved = True
                        break
            
            if improved:
                # Compact colors
                present = sorted(set(col))
                if present != list(range(len(present))):
                    mapping = {old: new for new, old in enumerate(present)}
                    for i in range(m):
                        col[i] = mapping[col[i]]
                    color_members = defaultdict(list)
                    for i, c in enumerate(col):
                        color_members[c].append(i)
        
        return col

    def _simulated_annealing(colour, C, time_limit=20.0):
        """
        Simulated annealing: allow moves to higher colors temporarily to escape local optima,
        always accept moves to lower colors.
        """
        t_start = time.time()
        m = len(colour)
        col = colour[:]
        n_colors = max(col) + 1 if m > 0 else 0
        
        color_members = defaultdict(list)
        for i, c in enumerate(col):
            color_members[c].append(i)
        
        best_col = col[:]
        best_n = n_colors
        rng = Random(123)
        
        # Temperature schedule
        t_max = 5.0
        t_min = 0.01
        n_iter = 0
        max_iter = 200000  # cap iterations
        
        while time.time() - t_start < time_limit and n_iter < max_iter:
            n_iter += 1
            # Temperature
            progress = n_iter / max_iter
            temp = t_max * (t_min / t_max) ** progress
            
            # Pick a random edge
            i = rng.randrange(m)
            cur_c = col[i]
            
            # Try a random different color
            new_c = rng.randrange(n_colors)
            if new_c == cur_c:
                continue
            
            # Check if new color is valid for edge i
            members = color_members[new_c]
            if members:
                member_arr = np.array(members, dtype=np.int32)
                if np.any(C[i, member_arr]):
                    continue  # Invalid move, skip
            
            # Perform the move
            color_members[cur_c].remove(i)
            color_members[new_c].append(i)
            old_n_colors = n_colors
            
            # Update n_colors
            if not color_members[cur_c]:
                # Color cur_c is now empty
                if cur_c == n_colors - 1:
                    n_colors -= 1
            elif new_c >= n_colors:
                n_colors = new_c + 1
            
            col[i] = new_c
            
            # Accept or reject
            if new_c < cur_c:
                # Improvement, always accept
                pass
            elif new_c > cur_c:
                # Worsening, accept with probability
                delta = 1  # cost increase
                if rng.random() > np.exp(-delta / temp):
                    # Reject: undo
                    color_members[new_c].remove(i)
                    color_members[cur_c].append(i)
                    col[i] = cur_c
                    n_colors = old_n_colors
            else:
                # Same cost, accept
                pass
            
            # Track best
            if n_colors < best_n:
                best_n = n_colors
                best_col = col[:]
        
        # Compact best
        if best_n < n_colors:
            present = sorted(set(best_col))
            if present != list(range(len(present))):
                mapping = {old: new for new, old in enumerate(present)}
                for i in range(m):
                    best_col[i] = mapping[best_col[i]]
        
        return best_col

    def _iterative_deepening(edges, C, current_best, time_limit=10.0):
        """
        Try to reduce color count using iterative deepening with constraint propagation.
        Start from current_best - 1 and go down.
        """
        m = len(edges)
        t_start = time.time()
        current_n = max(current_best) + 1 if m > 0 else 0
        
        # Order by constraint (most conflicts first)
        conflict_counts = np.sum(C, axis=1)
        order = np.argsort(-conflict_counts).tolist()
        
        for target in range(current_n - 1, 0, -1):
            if time.time() - t_start > time_limit:
                break
            
            # Backtracking with forward checking
            color = [-1] * m
            color_members = [set() for _ in range(target)]
            # For each edge, track which colors are still available
            # available[i] = set of colors that edge i can use
            available = [set(range(target)) for _ in range(m)]
            
            def backtrack(pos):
                if time.time() - t_start > time_limit:
                    return False
                if pos == len(order):
                    return True
                
                i = order[pos]
                # Try colors in order of least constraining value
                # Sort available colors by how many options they leave for future edges
                avail = list(available[i])
                if not avail:
                    return False
                
                # MRV heuristic: try color that leaves most options for neighbors
                # Simplified: just try in order
                for c in avail:
                    # Check if c is actually valid (no conflict with existing members)
                    members = color_members[c]
                    if members:
                        member_list = list(members)
                        if np.any(C[i, member_list]):
                            continue
                    
                    # Assign
                    color[i] = c
                    color_members[c].add(i)
                    
                    # Forward checking: update available for unassigned edges
                    affected = []
                    for j in range(pos + 1, len(order)):
                        ej = order[j]
                        if C[i, ej] and c in available[ej]:
                            available[ej].discard(c)
                            affected.append(ej)
                    
                    if backtrack(pos + 1):
                        return True
                    
                    # Undo
                    color[i] = -1
                    color_members[c].discard(i)
                    for ej in affected:
                        available[ej].add(c)
                
                return False
            
            if backtrack(0):
                return color
        
        return None

    def _solve_instance(n, d, seed):
        edges = _instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        C = _build_conflict_matrix(edges, adj)
        
        # Multiple greedy initializations with different orderings
        orderings = [
            list(range(m)),
            list(range(m - 1, -1, -1)),
            np.argsort(-np.sum(C, axis=1)).tolist(),  # most conflicts first
            np.argsort(np.sum(C, axis=1)).tolist(),   # least conflicts first
        ]
        # Add a few random orderings
        rng = Random(99)
        for _ in range(3):
            ro = list(range(m))
            rng.shuffle(ro)
            orderings.append(ro)
        
        best = None
        best_n = float('inf')
        for order in orderings:
            col = _greedy_with_cp(edges, C, order)
            n_col = max(col) + 1
            if n_col < best_n:
                best_n = n_col
                best = col[:]
        
        # Local search
        best = _local_search(best, C, time_limit=12.0)
        best_n = max(best) + 1
        
        # Simulated annealing
        best_sa = _simulated_annealing(best, C, time_limit=25.0)
        best_sa_n = max(best_sa) + 1 if best_sa else 0
        if best_sa_n < best_n:
            best = best_sa
            best_n = best_sa_n
        
        # Iterative deepening to try to reduce further
        remaining = _DEADLINE - time.time()
        if remaining > 15:
            id_limit = min(remaining - 5, 15.0)
            result = _iterative_deepening(edges, C, best, time_limit=id_limit)
            if result is not None:
                best = result
                best_n = max(best) + 1
        
        return best

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        if time.time() > _DEADLINE - 20:
            # Not enough time, use a quick greedy
            edges = _instance(n, d, seed)
            m = len(edges)
            if m == 0:
                out.append([])
                continue
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            C = _build_conflict_matrix(edges, adj)
            col = _greedy_with_cp(edges, C, list(range(m)))
            out.append(col)
        else:
            out.append(_solve_instance(n, d, seed))
    
    return out
# EVOLVE-BLOCK-END
