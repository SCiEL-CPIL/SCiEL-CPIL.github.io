"""
R_4(C6): maximize n with a 4-colouring of K_n with no monochromatic C6.
Uses bit-parallel constraint-propagation backtracking to build K_18 edge-by-edge,
maintaining incremental C6-detection state for aggressive pruning.
"""
import time
import random
import numpy as np

DEADLINE_MARGIN = 5.0  # seconds before the 150s wall-clock kill to stop


def _build_adj(colors, n):
    """Build adjacency bitmasks for each color class from a flat color list."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj


def _has_c6_bitset(adj, n):
    """Check if the graph defined by adjacency masks 'adj' contains a 6-cycle."""
    for v0 in range(n):
        mask_v1 = adj[v0]
        while mask_v1:
            v1 = (mask_v1 & -mask_v1).bit_length() - 1
            mask_v1 &= mask_v1 - 1

            mask_v2 = adj[v1] & ~(1 << v0)
            while mask_v2:
                v2 = (mask_v2 & -mask_v2).bit_length() - 1
                mask_v2 &= mask_v2 - 1

                mask_v3 = adj[v2] & ~(1 << v0) & ~(1 << v1)
                while mask_v3:
                    v3 = (mask_v3 & -mask_v3).bit_length() - 1
                    mask_v3 &= mask_v3 - 1

                    mask_v4 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                    while mask_v4:
                        v4 = (mask_v4 & -mask_v4).bit_length() - 1
                        mask_v4 &= mask_v4 - 1

                        mask_v5 = adj[v4] & adj[v0]
                        used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                        mask_v5 &= ~used_mask
                        if mask_v5:
                            return True
    return False


def _is_valid(colors, n):
    """Check if the coloring is valid (no mono C6)."""
    adj = _build_adj(colors, n)
    for c in range(4):
        if _has_c6_bitset(adj[c], n):
            return False
    return True


def _find_c6_in_color(adj, n):
    """Find one C6 in a specific color class. Returns (v0..v5) or None."""
    for v0 in range(n):
        mask_v1 = adj[v0]
        while mask_v1:
            v1 = (mask_v1 & -mask_v1).bit_length() - 1
            mask_v1 &= mask_v1 - 1

            mask_v2 = adj[v1] & ~(1 << v0)
            while mask_v2:
                v2 = (mask_v2 & -mask_v2).bit_length() - 1
                mask_v2 &= mask_v2 - 1

                mask_v3 = adj[v2] & ~(1 << v0) & ~(1 << v1)
                while mask_v3:
                    v3 = (mask_v3 & -mask_v3).bit_length() - 1
                    mask_v3 &= mask_v3 - 1

                    mask_v4 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                    while mask_v4:
                        v4 = (mask_v4 & -mask_v4).bit_length() - 1
                        mask_v4 &= mask_v4 - 1

                        mask_v5 = adj[v4] & adj[v0]
                        used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                        mask_v5 &= ~used_mask
                        if mask_v5:
                            v5 = (mask_v5 & -mask_v5).bit_length() - 1
                            return (v0, v1, v2, v3, v4, v5)
    return None


def _edge_index(i, j, n):
    """Convert (i,j) with i<j to linear index in lexicographic order."""
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)


def _count_c6s_fast(adj, n):
    """Count C6s in all color classes using bit-parallel method."""
    total = 0
    for c in range(4):
        for v0 in range(n):
            mask_v1 = adj[c][v0]
            while mask_v1:
                v1 = (mask_v1 & -mask_v1).bit_length() - 1
                mask_v1 &= mask_v1 - 1
                mask_v2 = adj[c][v1] & ~(1 << v0)
                while mask_v2:
                    v2 = (mask_v2 & -mask_v2).bit_length() - 1
                    mask_v2 &= mask_v2 - 1
                    mask_v3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                    while mask_v3:
                        v3 = (mask_v3 & -mask_v3).bit_length() - 1
                        mask_v3 &= mask_v3 - 1
                        mask_v4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                        while mask_v4:
                            v4 = (mask_v4 & -mask_v4).bit_length() - 1
                            mask_v4 &= mask_v4 - 1
                            mask_v5 = adj[c][v4] & adj[c][v0]
                            used = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                            mask_v5 &= ~used
                            total += bin(mask_v5).count('1')
    return total


class _C6ConstraintSolver:
    """
    Bit-parallel constraint-propagation solver for 4-coloring K_n with no mono C6.
    Builds the coloring edge-by-edge, maintaining adjacency masks and checking
    C6 constraints incrementally with aggressive pruning.
    """
    
    def __init__(self, n, deadline):
        self.n = n
        self.deadline = deadline
        self.num_edges = n * (n - 1) // 2
        # Edge ordering: process edges in a specific order for better pruning
        self.edge_order = self._generate_edge_order()
        self.edge_to_pair = {}
        for idx, (i, j) in enumerate(self.edge_order):
            self.edge_to_pair[idx] = (i, j)
        self.adj = [[0] * n for _ in range(4)]
        self.colors = [None] * self.num_edges
        self._edge_color = [None] * self.num_edges  # color assigned to edge in edge_order
        self.nodes_visited = 0
        self._deadline_hit = False
    
    def _generate_edge_order(self):
        """Generate edge ordering that promotes early constraint satisfaction."""
        n = self.n
        edges = []
        # Process edges from high-index vertices first to build dense subgraphs early
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                edges.append((i, j))
        return edges
    
    def _check_c6_for_color(self, c, i, j):
        """
        Check if adding edge (i,j) to color c creates a C6.
        Only checks cycles that include the new edge for efficiency.
        """
        n = self.n
        # A C6 using edge (i,j) in color c: i - a - b - d - e - j - i
        # We need to find paths of length 4 from i to j using only color-c edges
        # (excluding the direct edge i-j which we're about to add)
        
        # Save current state
        self.adj[c][i] |= (1 << j)
        self.adj[c][j] |= (1 << i)
        
        # Check for C6: find 4 distinct vertices a,b,d,e such that
        # i-a, a-b, b-d, d-e, e-j are all in color c, and all 6 vertices distinct
        found = False
        # Iterate over neighbors of i in color c (excluding j)
        mask_a = self.adj[c][i] & ~(1 << j)
        while mask_a and not found:
            a = (mask_a & -mask_a).bit_length() - 1
            mask_a &= mask_a - 1
            if a == j:
                continue
            # From a, find neighbors b (not i, not j)
            mask_b = self.adj[c][a] & ~(1 << i) & ~(1 << j)
            while mask_b and not found:
                b = (mask_b & -mask_b).bit_length() - 1
                mask_b &= mask_b - 1
                if b == i or b == j or b == a:
                    continue
                # From b, find neighbors d (not i, j, a)
                mask_d = self.adj[c][b] & ~(1 << i) & ~(1 << j) & ~(1 << a)
                while mask_d and not found:
                    d = (mask_d & -mask_d).bit_length() - 1
                    mask_d &= mask_d - 1
                    if d in (i, j, a, b):
                        continue
                    # From d, find neighbors e (not i, j, a, b)
                    mask_e = self.adj[c][d] & ~(1 << i) & ~(1 << j) & ~(1 << a) & ~(1 << b)
                    while mask_e and not found:
                        e = (mask_e & -mask_e).bit_length() - 1
                        mask_e &= mask_e - 1
                        if e in (i, j, a, b, d):
                            continue
                        # Check if e connects to j
                        if self.adj[c][e] & (1 << j):
                            found = True
                            break
        
        # Remove the edge we added
        self.adj[c][i] &= ~(1 << j)
        self.adj[c][j] &= ~(1 << i)
        
        return found
    
    def _check_c6_full(self, c):
        """Full C6 check for color c (used for final verification)."""
        return _has_c6_bitset(self.adj[c], self.n)
    
    def _check_all_c6(self):
        """Check if any color has a C6."""
        for c in range(4):
            if _has_c6_bitset(self.adj[c], self.n):
                return False
        return True
    
    def solve(self, initial_colors=None):
        """
        Solve the constraint satisfaction problem.
        If initial_colors is provided, it's a list of colors for edges in standard order.
        Returns the solution colors in standard lexicographic order, or None.
        """
        if initial_colors is not None:
            # Load initial colors
            for idx, (i, j) in enumerate(self.edge_order):
                std_idx = _edge_index(i, j, self.n)
                self._edge_color[idx] = initial_colors[std_idx]
                c = initial_colors[std_idx]
                self.adj[c][i] |= (1 << j)
                self.adj[c][j] |= (1 << i)
                self.colors[std_idx] = c
            # Verify initial state is valid
            if not self._check_all_c6():
                return None
            # Now try to find if we can modify to get a valid solution
            # For now, just verify and return
            return self.colors[:]
        
        return self._backtrack(0)
    
    def _backtrack(self, pos):
        """Recursive backtracking with constraint propagation."""
        self.nodes_visited += 1
        if self._deadline_hit:
            return None
        if time.time() > self.deadline:
            self._deadline_hit = True
            return None
        
        if pos == self.num_edges:
            # All edges assigned - verify no C6
            if self._check_all_c6():
                return self.colors[:]
            return None
        
        i, j = self.edge_to_pair[pos]
        
        # Try each color
        for c in range(4):
            if self._deadline_hit:
                return None
            
            # Check if adding edge (i,j) with color c creates a C6
            if self._check_c6_for_color(c, i, j):
                continue
            
            # Assign the edge
            self._edge_color[pos] = c
            self.adj[c][i] |= (1 << j)
            self.adj[c][j] |= (1 << i)
            std_idx = _edge_index(i, j, self.n)
            self.colors[std_idx] = c
            
            # Recurse
            result = self._backtrack(pos + 1)
            if result is not None:
                return result
            
            # Undo
            self._edge_color[pos] = None
            self.adj[c][i] &= ~(1 << j)
            self.adj[c][j] &= ~(1 << i)
            self.colors[std_idx] = None
        
        return None
    
    def solve_extend(self, base_colors, new_vertex):
        """
        Extend a valid coloring of K_n to K_{n+1} by adding new_vertex.
        base_colors: colors for K_n in standard order
        new_vertex: the new vertex index (should be n)
        Returns extended colors or None.
        """
        n_old = self.n - 1
        n_new = self.n
        self.n = n_new
        self.num_edges = n_new * (n_new - 1) // 2
        self.edge_order = self._generate_edge_order()
        self.edge_to_pair = {}
        for idx, (i, j) in enumerate(self.edge_order):
            self.edge_to_pair[idx] = (i, j)
        self.adj = [[0] * n_new for _ in range(4)]
        self.colors = [None] * self.num_edges
        self._edge_color = [None] * self.num_edges
        
        # Load base colors
        for idx, (i, j) in enumerate(self.edge_order):
            if i < n_old and j < n_old:
                std_idx = _edge_index(i, j, n_old)
                c = base_colors[std_idx]
                self._edge_color[idx] = c
                self.adj[c][i] |= (1 << j)
                self.adj[c][j] |= (1 << i)
                self.colors[_edge_index(i, j, n_new)] = c
            else:
                # This edge involves the new vertex - leave unassigned
                pass
        
        # Find which edges in edge_order involve the new vertex
        new_edges = []
        for idx, (i, j) in enumerate(self.edge_order):
            if i == new_vertex or j == new_vertex:
                new_edges.append(idx)
        
        # Backtrack only on the new edges
        return self._backtrack_new_edges(new_edges, 0)
    
    def _backtrack_new_edges(self, new_edges, pos):
        """Backtrack only on edges involving the new vertex."""
        self.nodes_visited += 1
        if self._deadline_hit:
            return None
        if time.time() > self.deadline:
            self._deadline_hit = True
            return None
        
        if pos == len(new_edges):
            if self._check_all_c6():
                return self.colors[:]
            return None
        
        edge_idx = new_edges[pos]
        i, j = self.edge_to_pair[edge_idx]
        
        for c in range(4):
            if self._deadline_hit:
                return None
            
            if self._check_c6_for_color(c, i, j):
                continue
            
            self._edge_color[edge_idx] = c
            self.adj[c][i] |= (1 << j)
            self.adj[c][j] |= (1 << i)
            std_idx = _edge_index(i, j, self.n)
            self.colors[std_idx] = c
            
            result = self._backtrack_new_edges(new_edges, pos + 1)
            if result is not None:
                return result
            
            self._edge_color[edge_idx] = None
            self.adj[c][i] &= ~(1 << j)
            self.adj[c][j] &= ~(1 << i)
            self.colors[std_idx] = None
        
        return None


def _find_valid_k17_fast(deadline):
    """
    Find a valid 4-coloring of K_17 using bit-parallel backtracking.
    """
    solver = _C6ConstraintSolver(17, deadline)
    result = solver.solve()
    if result is not None:
        return result
    return None


def _extend_to_k18(base_colors_17, deadline):
    """
    Extend K_17 coloring to K_18 using bit-parallel backtracking.
    """
    solver = _C6ConstraintSolver(18, deadline)
    result = solver.solve_extend(base_colors_17, 17)
    if result is not None:
        return result
    return None


def _local_search_minimize_c6(colors, n, time_limit, seed=42):
    """
    Local search minimizing C6 count using incremental updates.
    Returns (colors, c6_count) or (None, None).
    """
    rng = random.Random(seed)
    deadline = time.time() + time_limit
    num_edges = n * (n - 1) // 2
    
    colors = colors[:]
    adj = _build_adj(colors, n)
    current_c6s = _count_c6s_fast(adj, n)
    
    if current_c6s == 0:
        return colors, 0
    
    best_colors = colors[:]
    best_c6s = current_c6s
    
    # Precompute edge pairs for fast lookup
    edge_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_pairs.append((i, j))
    
    for iteration in range(100000):
        if time.time() > deadline:
            break
        if best_c6s == 0:
            return best_colors, 0
        
        edge_idx = rng.randint(0, num_edges - 1)
        i, j = edge_pairs[edge_idx]
        old_color = colors[edge_idx]
        new_color = (old_color + 1 + rng.randint(0, 2)) % 4
        
        # Incremental update
        adj[old_color][i] &= ~(1 << j)
        adj[old_color][j] &= ~(1 << i)
        adj[new_color][i] |= (1 << j)
        adj[new_color][j] |= (1 << i)
        colors[edge_idx] = new_color
        
        new_c6s = _count_c6s_fast(adj, n)
        
        if new_c6s <= best_c6s:
            best_c6s = new_c6s
            best_colors = colors[:]
            current_c6s = new_c6s
        else:
            # Undo
            adj[new_color][i] &= ~(1 << j)
            adj[new_color][j] &= ~(1 << i)
            adj[old_color][i] |= (1 << j)
            adj[old_color][j] |= (1 << i)
            colors[edge_idx] = old_color
    
    if best_c6s == 0:
        return best_colors, 0
    return None, best_c6s


def _construct_known_17():
    """
    Find a valid 4-coloring of K_17 using bit-parallel backtracking.
    Falls back to local search if backtracking times out.
    """
    start = time.time()
    deadline = start + 20.0
    
    # Try bit-parallel backtracking
    colors_17 = _find_valid_k17_fast(deadline)
    
    if colors_17 is not None:
        # Verify
        adj = _build_adj(colors_17, 17)
        valid = True
        for c in range(4):
            if _has_c6_bitset(adj[c], 17):
                valid = False
                break
        if valid:
            return {'n': 17, 'colors': colors_17}
    
    # Fallback: local search
    rng = random.Random(42)
    num_edges = 17 * 16 // 2
    for attempt in range(5):
        if time.time() > deadline:
            break
        colors = [rng.randint(0, 3) for _ in range(num_edges)]
        result_colors, c6_count = _local_search_minimize_c6(colors, 17, 3.0, seed=42 + attempt)
        if result_colors is not None and c6_count == 0:
            return {'n': 17, 'colors': result_colors}
    
    # Try smaller n
    for n in range(16, 4, -1):
        if time.time() > deadline:
            break
        num_edges = n * (n - 1) // 2
        for attempt in range(3):
            colors = [rng.randint(0, 3) for _ in range(num_edges)]
            result_colors, c6_count = _local_search_minimize_c6(colors, n, 2.0, seed=42 + attempt)
            if result_colors is not None and c6_count == 0:
                return {'n': n, 'colors': result_colors}
    
    # Last resort: n=4
    rng = random.Random(42)
    colors = [rng.randint(0, 3) for _ in range(6)]
    return {'n': 4, 'colors': colors}


def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)
    
    # Get a valid baseline
    result = _construct_known_17()
    best_n = result['n']
    best_colors = result['colors']
    
    # If we have n=17, try to extend to n=18 using bit-parallel backtracking
    if best_n == 17 and time.time() < deadline - 10:
        extension_deadline = min(deadline, time.time() + 40)
        extension_result = _extend_to_k18(best_colors, extension_deadline)
        if extension_result is not None:
            # Verify the extension
            adj = _build_adj(extension_result, 18)
            valid = True
            for c in range(4):
                if _has_c6_bitset(adj[c], 18):
                    valid = False
                    break
            if valid:
                return {'n': 18, 'colors': extension_result}
    
    # If we still have time, try a full backtracking search for K_18 from scratch
    if time.time() < deadline - 20:
        solver_deadline = min(deadline, time.time() + 30)
        solver = _C6ConstraintSolver(18, solver_deadline)
        result_18 = solver.solve()
        if result_18 is not None:
            adj = _build_adj(result_18, 18)
            valid = True
            for c in range(4):
                if _has_c6_bitset(adj[c], 18):
                    valid = False
                    break
            if valid:
                return {'n': 18, 'colors': result_18}
    
    return {'n': best_n, 'colors': best_colors}
