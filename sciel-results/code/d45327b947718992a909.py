"""5-colourings of K_n without monochromatic C4.
Combines algebraic constructions (Z_2 x Z_13, GF(25)) with efficient
delta-energy local search to find valid colourings up to n=26+.
"""
import time
import random
import math

_START = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _time_left():
    return _LIMIT - _MARGIN - (time.time() - _START)

def _validate(n, colors):
    """Check that no colour class contains a 4-cycle (C4)."""
    if len(colors) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
    nbrs = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            nbrs[c][i].add(j)
            nbrs[c][j].add(i)
            idx += 1
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                if len(nbrs[c][i] & nbrs[c][j]) >= 2:
                    return False
    return True

def _seed_construction():
    """Original K11 seed: split into five Hamiltonian cycles by circular distance 1..5."""
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def _cyclic_construction(n):
    """Cyclic 5-colouring using circular distance mapped to colour (d-1)%5."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            d = min(d, n - d)
            c = (d - 1) % 5
            colors.append(c)
    return {'n': n, 'colors': colors}

def _gf25_construction():
    """
    Construction using GF(25) = GF(5^2).
    Vertices: 25 elements of GF(25).
    """
    def add(p, q):
        return ((p[0]+q[0])%5, (p[1]+q[1])%5)
    
    def sub(p, q):
        return ((p[0]-q[0])%5, (p[1]-q[1])%5)
    
    def mul(p, q):
        a, b = p
        c, d = q
        real = (a*c + 3*b*d) % 5
        imag = (a*d + b*c) % 5
        return (real, imag)
    
    def pow_elem(e, n):
        result = (1, 0)
        base = e
        while n > 0:
            if n % 2 == 1:
                result = mul(result, base)
            base = mul(base, base)
            n //= 2
        return result
    
    alpha = (0, 1)
    powers = []
    cur = (1, 0)
    for i in range(24):
        powers.append(cur)
        cur = mul(cur, alpha)
    
    if len(set(powers)) != 24:
        return None
    
    elem_to_exp = {}
    for i, p in enumerate(powers):
        elem_to_exp[p] = i
    
    all_elems = [(a, b) for a in range(5) for b in range(5)]
    
    n = 25
    colors = []
    for i in range(n):
        ei = all_elems[i]
        for j in range(i+1, n):
            ej = all_elems[j]
            diff = sub(ej, ei)
            if diff == (0, 0):
                continue
            exp = elem_to_exp.get(diff, None)
            if exp is None:
                c = 0
            else:
                c = exp // 5
            colors.append(c)
    
    return {'n': n, 'colors': colors}

def _z2_z13_construction():
    """
    Construction using Z_2 x Z_13.
    Vertices: (x, y) where x in {0,1}, y in Z_13. Total 26 vertices.
    """
    n = 26
    qr = {1, 3, 4, 9, 10, 12}
    qnr = {2, 5, 6, 7, 8, 11}
    
    color_map_same = {}
    for d in [1, 3]:
        color_map_same[d] = 0
    for d in [4, 9]:
        color_map_same[d] = 1
    for d in [10, 12]:
        color_map_same[d] = 2
    for d in [2, 5, 6]:
        color_map_same[d] = 3
    for d in [7, 8, 11]:
        color_map_same[d] = 4
    
    color_map_diff = {}
    for d in [1, 2]:
        color_map_diff[d] = 0
    for d in [3, 5]:
        color_map_diff[d] = 1
    for d in [4, 6]:
        color_map_diff[d] = 2
    for d in [7, 10]:
        color_map_diff[d] = 3
    for d in [8, 9, 11, 12]:
        color_map_diff[d] = 4
    
    colors = []
    for i in range(n):
        x1 = i % 2
        y1 = i // 2
        for j in range(i + 1, n):
            x2 = j % 2
            y2 = j // 2
            dx = (x2 - x1) % 2
            dy = (y2 - y1) % 13
            if dx == 0:
                if dy == 0:
                    c = 0
                else:
                    c = color_map_same.get(dy, 0)
            else:
                if dy == 0:
                    c = 4
                else:
                    c = color_map_diff.get(dy, 0)
            colors.append(c)
    
    return {'n': n, 'colors': colors}

class C4EnergyTracker:
    """Tracks C4 count per color with efficient delta updates."""
    
    def __init__(self, n, colors):
        self.n = n
        self.colors = colors[:]
        self.num_edges = n * (n - 1) // 2
        # adj[c][i] = set of neighbors of i in color c
        self.adj = [[set() for _ in range(n)] for _ in range(5)]
        # pair_count[c][i][j] = number of common neighbors of i and j in color c
        # We'll use a dict for sparse storage, or a 2D array
        # For n up to 28, n^2 = 784, so a 2D array is fine
        self.pair_count = [[0] * n for _ in range(n)]  # will be per color
        self.pair_count = [[ [0]*n for _ in range(n) ] for _ in range(5)]
        self.c4_count = [0] * 5  # total C4s per color
        self._build()
    
    def _build(self):
        """Build adjacency and pair counts from scratch."""
        for c in range(5):
            for i in range(self.n):
                self.adj[c][i].clear()
                for j in range(self.n):
                    self.pair_count[c][i][j] = 0
            self.c4_count[c] = 0
        
        idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                c = self.colors[idx]
                self.adj[c][i].add(j)
                self.adj[c][j].add(i)
                idx += 1
        
        # Compute pair counts and C4 counts
        for c in range(5):
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    common = len(self.adj[c][i] & self.adj[c][j])
                    self.pair_count[c][i][j] = common
                    self.pair_count[c][j][i] = common
                    if common >= 2:
                        self.c4_count[c] += common - 1
    
    def get_total_energy(self):
        return sum(self.c4_count)
    
    def _edge_idx(self, a, b):
        if a > b:
            a, b = b, a
        return a * (self.n - 1) - a * (a + 1) // 2 + (b - a - 1)
    
    def _remove_edge(self, i, j, c):
        """Remove edge (i,j) from color c, updating counts."""
        # i and j are no longer adjacent in color c
        # For each k in adj[c][i] \ {j}, pair_count[c][k][j] decreases by 1
        # For each k in adj[c][j] \ {i}, pair_count[c][i][k] decreases by 1
        
        old_c4 = self.c4_count[c]
        
        # Update pair counts for pairs involving j and neighbors of i
        for k in list(self.adj[c][i]):
            if k == j:
                continue
            old_pc = self.pair_count[c][k][j]
            self.pair_count[c][k][j] = old_pc - 1
            self.pair_count[c][j][k] = old_pc - 1
            if old_pc >= 2 and self.pair_count[c][k][j] < 2:
                self.c4_count[c] -= 1
            elif old_pc < 2 and self.pair_count[c][k][j] >= 2:
                self.c4_count[c] += 1
        
        # Update pair counts for pairs involving i and neighbors of j
        for k in list(self.adj[c][j]):
            if k == i:
                continue
            old_pc = self.pair_count[c][i][k]
            self.pair_count[c][i][k] = old_pc - 1
            self.pair_count[c][k][i] = old_pc - 1
            if old_pc >= 2 and self.pair_count[c][i][k] < 2:
                self.c4_count[c] -= 1
            elif old_pc < 2 and self.pair_count[c][i][k] >= 2:
                self.c4_count[c] += 1
        
        # Remove the edge itself
        self.adj[c][i].remove(j)
        self.adj[c][j].remove(i)
    
    def _add_edge(self, i, j, c):
        """Add edge (i,j) to color c, updating counts."""
        # For each k in adj[c][i], pair_count[c][k][j] increases by 1
        for k in list(self.adj[c][i]):
            old_pc = self.pair_count[c][k][j]
            self.pair_count[c][k][j] = old_pc + 1
            self.pair_count[c][j][k] = old_pc + 1
            if old_pc >= 2 and self.pair_count[c][k][j] < 2:
                self.c4_count[c] -= 1
            elif old_pc < 2 and self.pair_count[c][k][j] >= 2:
                self.c4_count[c] += 1
        
        # For each k in adj[c][j], pair_count[c][i][k] increases by 1
        for k in list(self.adj[c][j]):
            old_pc = self.pair_count[c][i][k]
            self.pair_count[c][i][k] = old_pc + 1
            self.pair_count[c][k][i] = old_pc + 1
            if old_pc >= 2 and self.pair_count[c][i][k] < 2:
                self.c4_count[c] -= 1
            elif old_pc < 2 and self.pair_count[c][i][k] >= 2:
                self.c4_count[c] += 1
        
        # Add the edge itself
        self.adj[c][i].add(j)
        self.adj[c][j].add(i)
    
    def recolor_edge(self, i, j, new_color):
        """Recolor edge (i,j) to new_color, returning energy delta."""
        old_color = self.colors[self._edge_idx(i, j)]
        if old_color == new_color:
            return 0
        
        old_energy = self.get_total_energy()
        
        self._remove_edge(i, j, old_color)
        self._add_edge(i, j, new_color)
        self.colors[self._edge_idx(i, j)] = new_color
        
        new_energy = self.get_total_energy()
        return new_energy - old_energy
    
    def random_recolor(self, rng):
        """Pick a random edge and recolor it, returning energy delta."""
        i = rng.randint(0, self.n - 2)
        j = rng.randint(i + 1, self.n - 1)
        new_color = rng.randint(0, 4)
        delta = self.recolor_edge(i, j, new_color)
        return delta

def _local_search(n, initial_colors, max_steps=200000, seed=42):
    """
    Local search with delta-energy updates to minimize C4 count.
    Returns (colors, energy) of best solution found.
    """
    rng = random.Random(seed)
    tracker = C4EnergyTracker(n, initial_colors)
    
    best_colors = initial_colors[:]
    best_energy = tracker.get_total_energy()
    
    if best_energy == 0:
        return best_colors, 0
    
    # Simulated annealing
    temperature = 1.0
    cooling_rate = 0.9999
    min_temperature = 0.01
    
    for step in range(max_steps):
        if _time_left() < 10:
            break
        
        delta = tracker.random_recolor(rng)
        
        if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 0.001)):
            new_energy = tracker.get_total_energy()
            if new_energy < best_energy:
                best_energy = new_energy
                best_colors = tracker.colors[:]
                if best_energy == 0:
                    break
        else:
            # Roll back
            # We need to undo the recolor
            # The last edge colored is the one we just changed
            # But random_recolor doesn't tell us which edge
            # Let's modify random_recolor to return more info
            pass
        
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature
    
    return best_colors, best_energy

def _local_search_v2(n, initial_colors, max_steps=200000, seed=42):
    """
    Improved local search that properly handles rollbacks.
    """
    rng = random.Random(seed)
    tracker = C4EnergyTracker(n, initial_colors)
    
    best_colors = initial_colors[:]
    best_energy = tracker.get_total_energy()
    
    if best_energy == 0:
        return best_colors, 0
    
    temperature = 1.5
    cooling_rate = 0.99995
    min_temperature = 0.005
    
    for step in range(max_steps):
        if _time_left() < 8:
            break
        
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        new_color = rng.randint(0, 4)
        
        old_color = tracker.colors[tracker._edge_idx(i, j)]
        if new_color == old_color:
            continue
        
        delta = tracker.recolor_edge(i, j, new_color)
        
        if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 0.001)):
            new_energy = tracker.get_total_energy()
            if new_energy < best_energy:
                best_energy = new_energy
                best_colors = tracker.colors[:]
                if best_energy == 0:
                    break
        else:
            # Roll back
            tracker.recolor_edge(i, j, old_color)
        
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature
    
    return best_colors, best_energy

def _greedy_with_repair(n, max_iter=50000, seed=42):
    """
    Greedy construction with local search to find a valid 5-colouring of K_n.
    """
    random.seed(seed)
    
    m = n * (n - 1) // 2
    # Start with cyclic-like initialisation
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = min(j - i, n - j + i)
            colors.append((d - 1) % 5)
    
    # Try local search
    best_colors, best_energy = _local_search_v2(n, colors, max_steps=100000, seed=seed)
    
    if best_energy == 0:
        return {'n': n, 'colors': best_colors}
    
    return None

def construct():
    """Entry point: return the best 5-colouring found."""
    if _time_left() <= 0:
        return _seed_construction()
    
    best = _seed_construction()
    
    # Try cyclic constructions for various n
    for n in [25, 20, 15, 10, 5]:
        if _time_left() < 5:
            break
        res = _cyclic_construction(n)
        if _validate(n, res['colors']):
            if n > best['n']:
                best = res
    
    # Try GF(25) construction
    if _time_left() > 10:
        res = _gf25_construction()
        if res and _validate(res['n'], res['colors']):
            if res['n'] > best['n']:
                best = res
    
    # Try Z_2 x Z_13 construction
    if _time_left() > 10:
        res = _z2_z13_construction()
        if _validate(res['n'], res['colors']):
            if res['n'] > best['n']:
                best = res
    
    # If we have a valid coloring for n >= 25, try to extend to n+1, n+2, etc.
    # Use local search starting from the existing coloring
    current_n = best['n']
    
    # Try to grow from current_n up to 28
    for target_n in range(current_n + 1, 29):
        if _time_left() < 15:
            break
        
        # Initialize with random colors for the new vertices
        # Start from the existing coloring and add random edges for new vertices
        base_colors = best['colors']
        
        # Build initial coloring for target_n
        # First target_n - current_n vertices are new
        # We'll use a cyclic-like initialization for the full graph
        init_colors = []
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if i < current_n and j < current_n:
                    # Use existing coloring
                    idx = i * (current_n - 1) - i * (i + 1) // 2 + (j - i - 1)
                    init_colors.append(base_colors[idx])
                else:
                    # New edge: use cyclic-like
                    d = min(j - i, target_n - j + i)
                    init_colors.append((d - 1) % 5)
        
        # Run local search
        for attempt in range(3):
            if _time_left() < 10:
                break
            
            seed = 42 + attempt * 1000
            best_colors, best_energy = _local_search_v2(
                target_n, init_colors, max_steps=50000, seed=seed
            )
            
            if best_energy == 0:
                if _validate(target_n, best_colors):
                    best = {'n': target_n, 'colors': best_colors}
                    break
            elif best_energy < 5:
                # Close to valid, try more iterations
                if _time_left() > 20:
                    best_colors2, best_energy2 = _local_search_v2(
                        target_n, best_colors, max_steps=100000, seed=seed + 7
                    )
                    if best_energy2 == 0 and _validate(target_n, best_colors2):
                        best = {'n': target_n, 'colors': best_colors2}
                        break
    
    # If we still haven't found n >= 26, try direct local search from random init
    if best['n'] < 26 and _time_left() > 30:
        for target_n in [26, 25, 24]:
            if _time_left() < 15:
                break
            
            for attempt in range(5):
                if _time_left() < 10:
                    break
                
                rng = random.Random(1000 + attempt)
                num_edges = target_n * (target_n - 1) // 2
                init_colors = [rng.randint(0, 4) for _ in range(num_edges)]
                
                best_colors, best_energy = _local_search_v2(
                    target_n, init_colors, max_steps=80000, seed=1000 + attempt
                )
                
                if best_energy == 0 and _validate(target_n, best_colors):
                    if target_n > best['n']:
                        best = {'n': target_n, 'colors': best_colors}
                    break
    
    return best

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, valid={_validate(res['n'], res['colors'])}")
    print(res['colors'][:20])
