"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
ADJ = [[] for _ in range(N)]
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)

def _has_bicolored_cycle(coloring, edges):
    """Check if any pair of colors induces a cycle using union-find."""
    n = len(coloring)
    color_map = defaultdict(list)
    for i, c in enumerate(coloring):
        color_map[c].append(i)
    
    colors = list(color_map.keys())
    for i in range(len(colors)):
        for j in range(i+1, len(colors)):
            ci, cj = colors[i], colors[j]
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True
                parent[rx] = ry
                return False
            
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == ci and cv == cj) or (cu == cj and cv == ci):
                    if union(u, v):
                        return True
    return False

def _is_valid(coloring, edges):
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    return not _has_bicolored_cycle(coloring, edges)

def _count_colors(coloring):
    return len(set(coloring))

def _solve_k_coloring(k, deadline):
    """
    Backtracking solver for k-coloring with bicolored-forest constraint.
    Uses incremental DSU per color pair with proper undo.
    """
    sys.setrecursionlimit(10000)
    
    color = [-1] * N
    
    # For each pair of colors (a, b) with a < b, maintain:
    # - DSU parent array (only for vertices colored a or b)
    # - We track which vertices are "active" in this pair
    # - For undo, we save the state before each assignment
    
    # pair_dsu[(a,b)] = {'parent': dict, 'active': set, 'edge_count': int}
    pair_dsu = {}
    
    def make_key(a, b):
        if a > b:
            a, b = b, a
        return (a, b)
    
    def init_pair(pk):
        if pk not in pair_dsu:
            pair_dsu[pk] = {'parent': {}, 'active': set(), 'edge_count': 0}
    
    def find(pk, v):
        d = pair_dsu[pk]
        parent = d['parent']
        if v not in parent:
            parent[v] = v
            d['active'].add(v)
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    
    def union(pk, u, v):
        d = pair_dsu[pk]
        ru = find(pk, u)
        rv = find(pk, v)
        if ru == rv:
            return False  # cycle
        d['parent'][rv] = ru
        d['edge_count'] += 1
        return True
    
    # For undo, we need to save the full state of affected pairs
    # This is expensive but correct
    
    def get_state(pk):
        """Save state of a pair for undo."""
        if pk not in pair_dsu:
            return None
        d = pair_dsu[pk]
        return {
            'parent': dict(d['parent']),
            'active': set(d['active']),
            'edge_count': d['edge_count']
        }
    
    def restore_state(pk, state):
        """Restore state of a pair."""
        if state is None:
            if pk in pair_dsu:
                del pair_dsu[pk]
        else:
            pair_dsu[pk] = {
                'parent': state['parent'],
                'active': state['active'],
                'edge_count': state['edge_count']
            }
    
    def assign(v, c):
        """
        Assign color c to vertex v.
        Returns (success, saved_states) where saved_states is a list of (pk, state) for undo.
        """
        color[v] = c
        saved_states = []
        
        for u in ADJ[v]:
            d = color[u]
            if d == -1 or d == c:
                continue
            pk = make_key(c, d)
            # Save state before modification
            state = get_state(pk)
            init_pair(pk)
            
            # Add edge (v, u) to this pair's DSU
            if not union(pk, v, u):
                # Cycle detected - undo everything
                color[v] = -1
                # Restore all saved states
                for pk2, state2 in saved_states:
                    restore_state(pk2, state2)
                return (False, [])
            
            saved_states.append((pk, state))
        
        return (True, saved_states)
    
    def unassign(v, c, saved_states):
        """Undo assignment of color c to vertex v."""
        color[v] = -1
        for pk, state in saved_states:
            restore_state(pk, state)
    
    # Vertex ordering: process in natural order for simplicity
    # Could try MRV heuristic but let's start simple
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == N:
            return True
        
        v = idx
        # Try each color
        for c in range(k):
            # Check proper coloring: no neighbor has same color
            valid = True
            for u in ADJ[v]:
                if color[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            success, saved_states = assign(v, c)
            if success:
                if backtrack(idx + 1):
                    return True
                unassign(v, c, saved_states)
        
        return False
    
    if backtrack(0):
        return color[:]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, 10s safety margin
    
    n = N
    edges = EDGES
    
    # Try k from 3 upward
    for k in range(3, 15):
        if time.time() > deadline:
            break
        result = _solve_k_coloring(k, deadline)
        if result is not None:
            # Verify
            if _is_valid(result, edges):
                return [int(x) for x in result]
    
    # Fallback: try modular patterns
    for k in range(3, 20):
        if time.time() > deadline:
            break
        for a in range(1, k):
            for b in range(k):
                coloring = [(i * a + b) % k for i in range(n)]
                if _is_valid(coloring, edges):
                    return [int(x) for x in coloring]
    
    # Last resort: 50 colors (always valid)
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
