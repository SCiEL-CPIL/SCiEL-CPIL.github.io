import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    # Try n=26 first (known algebraic construction exists)
    # Then try n=27, 28 with search
    
    target_ns = [26, 27, 28]
    
    # First, try the algebraic construction for n=26
    if time.time() < deadline - 30:
        result = try_algebraic_26()
        if result is not None:
            return result
    
    # Try cyclic search for n=27, 28
    for n in [28, 27]:
        if time.time() > deadline - 20:
            break
        result = search_cyclic(n, time_limit=15)
        if result is not None:
            return result
    
    # Try general local search for n=27, 28
    for n in [28, 27]:
        if time.time() > deadline - 10:
            break
        result = local_search_general(n, time_limit=15)
        if result is not None:
            return result
    
    # Fallback: return a valid small coloring
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def try_algebraic_26():
    """
    Try the Lazebnik-Woldar type algebraic construction for n=26.
    This uses a finite field / affine geometry approach.
    
    For R_5(C4) > 26, we need a 5-coloring of K_26 with no monochromatic C4.
    
    One known approach: use vertices as pairs (i, j) where i in {0,1} and j in GF(25)
    or similar. But let's try a simpler cyclic/difference set approach.
    
    Actually, let's try a direct construction based on finite geometry.
    The known result R_5(C4) >= 26 comes from specific constructions.
    
    Let me try a systematic approach: use 26 vertices, and for the coloring,
    use a known pattern or try to find one via structured search.
    """
    n = 26
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Try a structured initialization based on modular arithmetic
    # Color based on (j-i) mod 26, mapped to 5 colors
    # This is a cyclic coloring - each difference gets a fixed color
    
    # For cyclic group Z_26, differences 1..13 (since diff and 26-diff are same edge)
    # We need to assign colors to differences 1..13 such that no color class has C4
    
    # A C4 in the color class for difference d means:
    # vertices a, a+d, a+2d, a+d... wait, let me think more carefully.
    # In a cyclic coloring where edge (i,j) has color c((j-i) mod n),
    # a monochromatic C4 would be four vertices a, b, c, d such that all four
    # edges form a cycle and all have the same difference class.
    
    # Actually for a circulant graph, a C4 exists if there are a, b, c, d with
    # b-a = d-c and d-b = c-a (mod n), i.e., the four points form a parallelogram
    # in the cyclic group.
    
    # Let's just try random restarts with the cyclic structure
    k = n // 2  # 13
    
    for attempt in range(200):
        params = [random.randint(0, 4) for _ in range(k)]
        colors = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            diff = (j - i) % n
            if diff <= k:
                d = diff
            else:
                d = n - diff
            colors[idx] = params[d - 1]
        
        if is_valid_coloring(n, colors):
            return {'n': n, 'colors': colors}
    
    return None

def is_valid_coloring(n, colors):
    """Check if a coloring has no monochromatic C4."""
    neighbors = [[0] * n for _ in range(5)]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
    
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = neighbors[c][u] & neighbors[c][v]
                if common.bit_count() > 1:
                    return False
    return True

def search_cyclic(n, time_limit=15):
    k = n // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    def gen_colors(params):
        colors = [0] * num_edges
        for idx, (i, j) in enumerate(edges):
            diff = (j - i) % n
            if diff <= k:
                d = diff
            else:
                d = n - diff
            colors[idx] = params[d - 1]
        return colors
    
    def calc_violations(colors):
        neighbors = [[0] * n for _ in range(5)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    cnt = common.bit_count()
                    if cnt > 1:
                        violations += cnt * (cnt - 1) // 2
        return violations
    
    best_violations = float('inf')
    best_params = None
    
    start_time = time.time()
    temperature = 2.0
    cooling_rate = 0.995
    
    for restart in range(50):
        params = [random.randint(0, 4) for _ in range(k)]
        current_colors = gen_colors(params)
        current_violations = calc_violations(current_colors)
        
        if current_violations == 0:
            return {'n': n, 'colors': current_colors}
        
        if current_violations < best_violations:
            best_violations = current_violations
            best_params = params[:]
        
        temperature = 2.0
        local_start = time.time()
        
        while time.time() < local_start + time_limit / 50 and time.time() < start_time + time_limit:
            idx = random.randint(0, k - 1)
            old_val = params[idx]
            new_val = random.randint(0, 4)
            if new_val == old_val:
                continue
                
            params[idx] = new_val
            new_colors = gen_colors(params)
            new_violations = calc_violations(new_colors)
            
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_violations = new_violations
                current_colors = new_colors
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_params = params[:]
                    if best_violations == 0:
                        return {'n': n, 'colors': current_colors}
            else:
                params[idx] = old_val
                
            temperature *= cooling_rate
            if temperature < 0.01:
                temperature = 0.01
    
    return None

def local_search_general(n, time_limit=15):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    def calc_violations(colors):
        neighbors = [[0] * n for _ in range(5)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
        violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    cnt = common.bit_count()
                    if cnt > 1:
                        violations += cnt * (cnt - 1) // 2
        return violations
    
    best_violations = float('inf')
    best_colors = None
    
    start_time = time.time()
    
    for restart in range(30):
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        current_violations = calc_violations(colors)
        
        if current_violations == 0:
            return {'n': n, 'colors': colors}
        
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
        
        temperature = 2.0
        cooling_rate = 0.999
        local_start = time.time()
        steps = 0
        
        while time.time() < local_start + time_limit / 30 and time.time() < start_time + time_limit:
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
                
            colors[e_idx] = new_color
            new_violations = calc_violations(colors)
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return {'n': n, 'colors': colors}
            else:
                colors[e_idx] = old_color
                
            temperature *= cooling_rate
            if temperature < 0.01:
                temperature = 0.01
            steps += 1
    
    return None
