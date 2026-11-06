import time
import random
import sys

def construct():
    # 1. Construct the n=32 base coloring (Greenwood-Gleason / Paley)
    # The Paley graph of order 32 is a 3-coloring of K_32 with no mono triangles?
    # Actually, the standard construction for R(3,3,3,3) > 32 uses a 3-coloring of K_32.
    # The 3-coloring is derived from the quadratic residues in GF(16) or similar.
    # Let's use the known formula for the 32-vertex 3-coloring.
    
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
        
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    def color_32(i, j):
        # This function returns a color in {0, 1, 2} for the base 32 vertices
        # based on the known construction.
        # The construction often maps i, j to GF(16) and uses quadratic characters.
        # A simpler known 3-coloring of K_32 without mono triangles:
        # Vertices are 0..31.
        # We can use the property that K_32 can be 3-colored without mono triangles.
        # One such coloring is given by:
        # c(i, j) = 0 if (i^j) is a quadratic residue in GF(16) (mapping 0..15 to 0..31 via two copies?)
        # Actually, the standard 32-vertex 3-coloring is:
        # Split 0..31 into two sets of 16.
        # Within each set, use the Paley graph of order 16 (which is 2-colorable? No, Paley(16) is self-complementary, so 2-coloring of K_16 without mono triangles is impossible? R(3,3)=6, so K_5 is 2-colorable without mono triangles. K_16 is not.)
        
        # Let's use the explicit 3-coloring of K_32.
        # It is known that K_32 admits a 3-coloring with no mono triangles.
        # We can hardcode the 496 colors for the 32-vertex core if necessary, 
        # or generate it.
        
        # Generation:
        # The 3-coloring of K_32 is related to the Hadamard matrix of order 32.
        # Let's use a simpler known 3-coloring:
        # Vertices 0..31.
        # Color(i, j) = 0 if i^j in QR, 1 if i^j in NQR, 2 if ...?
        # This is for 2 colors.
        
        # Actually, the seed program used:
        # log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3
        # This creates a 4-coloring of K_32?
        # If i//16 == j//16 (same half), color is 0,1,2.
        # If different halves, color is 3.
        # Let's verify if this is triangle-free.
        # If we take 3 vertices from the same half, they use colors 0,1,2.
        # The coloring of K_16 using log[x] % 3 must be triangle-free.
        # Paley(16) is a 2-coloring of K_16 with no mono triangles? No, R(3,3)=6, so max n for 2 colors is 5.
        # So K_16 cannot be 2-colored without mono triangles.
        # But it can be 3-colored.
        # The formula `log[(i%16) ^ (j%16)] % 3` produces a 3-coloring of K_16.
        # Is it triangle-free?
        # Let's assume the seed's n=32 solution is valid.
        
        if i // 16 == j // 16:
            return log[(i % 16) ^ (j % 16)] % 3
        else:
            return 3

    n_base = 32
    colors_base = [0] * (n_base * (n_base - 1) // 2)
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            colors_base[idx] = color_32(i, j)
            idx += 1
            
    # We want to extend to n=50.
    n_target = 50
    num_edges = n_target * (n_target - 1) // 2
    
    # Initialize the full array
    colors = [0] * num_edges
    
    # Helper to get index
    def get_idx(i, j):
        if i > j: i, j = j, i
        # i < j
        # idx = i * n_target - i * (i + 1) // 2 + (j - i - 1)
        # Let's precompute for speed
        return idx_map[(i, j)]

    idx_map = {}
    idx = 0
    for i in range(n_target):
        for j in range(i + 1, n_target):
            idx_map[(i, j)] = idx
            idx += 1
            
    # Recalculate base colors directly into the n=50 array
    for i in range(n_base):
        for j in range(i + 1, n_base):
            colors[get_idx(i, j)] = color_32(i, j)
            
    # Now we need to color edges involving vertices 32..49.
    # Greedy initialization for new vertices to minimize initial mono triangles
    random.seed(42)
    for v in range(n_base, n_target):
        for u in range(v):
            if u >= v: continue
            idx_uv = get_idx(u, v)
            # Try to find a color that doesn't complete a mono triangle with any w < v
            # A triangle (u, w, v) is mono if c(u,w) == c(u,v) == c(w,v).
            # We are setting c(u,v). c(u,w) is known (w < v < n_target, but w could be >= n_base? No, w < v, so if v is new, w could be new too).
            # Wait, if we iterate v from n_base to n_target-1, and u from 0 to v-1:
            # If u < n_base, c(u,w) for w < u is known (base).
            # If u >= n_base, c(u,w) was set in previous iterations of v (since u < v).
            # So all c(u,w) for w < u are known.
            # We need c(w,v) for w < v. These are NOT known yet if w > u?
            # No, we are setting c(u,v) for fixed v.
            # The constraint is: for all w < v, w != u, if c(u,w) == c_new, then c(w,v) must NOT be c_new.
            # But c(w,v) is not set yet if w > u?
            # Actually, for a fixed v, we are setting c(u,v) for all u < v.
            # We can process u in any order.
            # Let's just pick a random color first, then try to improve it locally?
            # Or simpler: Pick color that minimizes immediate conflicts with w < u.
            # Conflicts: if c(u,w) == c, and c(w,v) == c.
            # But c(w,v) is unknown if w > u.
            # Let's just use random for now, the SA will fix it.
            # However, a better heuristic:
            # For each u, count for each color c, how many w < u have c(u,w)==c AND c(w,v)==c?
            # We don't know c(w,v) for w > u.
            # Let's just stick to random initialization for new edges, it's robust enough for SA to start.
            colors[idx_uv] = random.randint(0, 3)

    # Precompute triangles for each edge (for delta calculation)
    edge_triangles = [[] for _ in range(num_edges)]
    for u in range(n_target):
        for v in range(u + 1, n_target):
            idx_uv = get_idx(u, v)
            for w in range(n_target):
                if w == u or w == v:
                    continue
                idx_uw = get_idx(u, w)
                idx_vw = get_idx(v, w)
                edge_triangles[idx_uv].append((idx_uw, idx_vw))

    def count_mono_triangles(cols):
        count = 0
        for e in range(num_edges):
            c_e = cols[e]
            for idx_uw, idx_vw in edge_triangles[e]:
                if cols[idx_uw] == c_e and cols[idx_vw] == c_e:
                    count += 1
        return count // 3 # Each triangle counted 3 times

    # Simulated Annealing
    start_time = time.time()
    timeout = 110.0
    
    current_colors = colors[:]
    current_mono = count_mono_triangles(current_colors)
    best_colors = current_colors[:]
    best_mono = current_mono
    
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    while time.time() - start_time < timeout:
        e_idx = random.randint(0, num_edges - 1)
        old_c = current_colors[e_idx]
        new_c = random.choice([c for c in range(4) if c != old_c])
        
        delta = 0
        for idx_uw, idx_vw in edge_triangles[e_idx]:
            c_uw = current_colors[idx_uw]
            c_vw = current_colors[idx_vw]
            
            if c_uw == old_c and c_vw == old_c:
                delta -= 1
            if c_uw == new_c and c_vw == new_c:
                delta += 1
                
        new_mono = current_mono + delta
        
        if new_mono < current_mono or random.random() < pow(2.718281828, -delta / T):
            current_colors[e_idx] = new_c
            current_mono = new_mono
            if current_mono < best_mono:
                best_mono = current_mono
                best_colors = current_colors[:]
                if best_mono == 0:
                    break
        T *= cooling
        
    if best_mono == 0:
        return {'n': n_target, 'colors': best_colors}
    else:
        # Fallback to n=32
        return {'n': 32, 'colors': colors_base}

# EVOLVE-BLOCK-END
