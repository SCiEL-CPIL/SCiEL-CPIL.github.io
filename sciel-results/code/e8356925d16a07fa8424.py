import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    def gf2_construct():
        """Construct a 4-coloring of K_49 using GF(7^2) and cubic residues."""
        # GF(7^2) = GF(49)
        # Elements are pairs (a, b) representing a + b*alpha where alpha^2 = 3 (since 3 is not a square in GF(7))
        # Actually, let's use a simpler representation: GF(49) as polynomials mod an irreducible polynomial
        
        # Let's use GF(7)[x] / (x^2 - 3) since 3 is not a quadratic residue mod 7
        # Squares mod 7: 0, 1, 4, 2 (1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1)
        # So 3 is not a square, x^2 - 3 is irreducible over GF(7)
        
        # Represent elements as (a, b) = a + b*alpha where alpha^2 = 3
        def gf_add(a, b):
            return ((a[0] + b[0]) % 7, (a[1] + b[1]) % 7)
        
        def gf_sub(a, b):
            return ((a[0] - b[0]) % 7, (a[1] - b[1]) % 7)
        
        def gf_mul(a, b):
            # (a0 + a1*alpha)(b0 + b1*alpha) = a0*b0 + (a0*b1 + a1*b0)*alpha + a1*b1*alpha^2
            # = (a0*b0 + a1*b1*3) + (a0*b1 + a1*b0)*alpha
            a0, a1 = a
            b0, b1 = b
            return ((a0 * b0 + a1 * b1 * 3) % 7, (a0 * b1 + a1 * b0) % 7)
        
        def gf_pow(a, n):
            result = (1, 0)  # multiplicative identity
            base = a
            while n > 0:
                if n & 1:
                    result = gf_mul(result, base)
                base = gf_mul(base, base)
                n >>= 1
            return result
        
        # Generate all 49 elements
        elements = [(a, b) for a in range(7) for b in range(7)]
        
        # Find a primitive element (generator of multiplicative group of order 48)
        # The multiplicative group has order 48 = 2^4 * 3
        # An element is primitive if its order is 48
        
        def gf_order(a):
            if a == (0, 0):
                return 0
            for k in range(1, 49):
                if gf_pow(a, k) == (1, 0):
                    return k
            return -1
        
        # Find primitive element
        primitive = None
        for elem in elements:
            if elem == (0, 0):
                continue
            if gf_order(elem) == 48:
                primitive = elem
                break
        
        if primitive is None:
            # Fallback: try alpha = (0, 1)
            primitive = (0, 1)
        
        # Generate powers of primitive element
        powers = [primitive]
        for i in range(1, 48):
            powers.append(gf_mul(powers[-1], primitive))
        
        # The multiplicative group has order 48
        # We need to partition into 4 color classes
        # For 4-coloring, we can use: color based on which 1/4 of the cyclic group
        # 48 / 4 = 12, so each color class has 12 elements from the multiplicative group
        # Plus the zero element gets assigned to one color
        
        # Color assignment:
        # Color 0: powers[0:12]
        # Color 1: powers[12:24]
        # Color 2: powers[24:36]
        # Color 3: powers[36:48]
        # Zero element: assign to color 0
        
        color_of_elem = {}
        color_of_elem[(0, 0)] = 0
        for i, p in enumerate(powers):
            color_of_elem[p] = i // 12
        
        # Now create the coloring
        n = 49
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = gf_sub(elements[i], elements[j])
                c = color_of_elem[diff]
                colors.append(c)
        
        return {'n': n, 'colors': colors}
    
    def count_triangles(mat):
        n = mat.shape[0]
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                c = mat[i][j]
                for k in range(j+1, n):
                    if mat[i][k] == c and mat[j][k] == c:
                        count += 1
        return count
    
    def find_mono_triangle(mat):
        n = mat.shape[0]
        for _ in range(200):
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            k = random.randint(0, n-1)
            if i == j or j == k or i == k:
                continue
            c1 = mat[i][j]
            c2 = mat[i][k]
            c3 = mat[j][k]
            if c1 == c2 == c3:
                return (i, j, k)
        for i in range(n):
            for j in range(i+1, n):
                c = mat[i][j]
                for k in range(j+1, n):
                    if mat[i][k] == c and mat[j][k] == c:
                        return (i, j, k)
        return None
    
    def local_search_sa(mat, n, time_budget):
        current_cost = count_triangles(mat)
        if current_cost == 0:
            return mat, True
        
        best_mat = mat.copy()
        best_cost = current_cost
        
        T = 1.0
        cooling = 0.9999
        start = time.time()
        
        while time.time() - start < time_budget:
            if current_cost == 0:
                return mat, True
            
            tri = find_mono_triangle(mat)
            if tri:
                i, j, k = tri
                edge_idx = random.randint(0, 2)
                if edge_idx == 0:
                    u, v = i, j
                elif edge_idx == 1:
                    u, v = i, k
                else:
                    u, v = j, k
            else:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
            
            old_color = mat[u][v]
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue
            
            delta = 0
            for w in range(n):
                if w == u or w == v:
                    continue
                if mat[u][w] == old_color and mat[v][w] == old_color:
                    delta -= 1
                if mat[u][w] == new_color and mat[v][w] == new_color:
                    delta += 1
            
            new_cost = current_cost + delta
            
            if new_cost < current_cost:
                mat[u][v] = new_color
                mat[v][u] = new_color
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_mat = mat.copy()
            elif delta == 0:
                mat[u][v] = new_color
                mat[v][u] = new_color
                current_cost = new_cost
            else:
                if random.random() < np.exp(-delta / max(T, 1e-10)):
                    mat[u][v] = new_color
                    mat[v][u] = new_color
                    current_cost = new_cost
            
            T *= cooling
            T = max(T, 0.01)
            
            if time.time() > deadline:
                break
        
        return best_mat, best_cost == 0
    
    # Try the GF(49) construction first
    gf_res = gf2_construct()
    n = gf_res['n']
    colors = gf_res['colors']
    
    # Convert to matrix and verify
    mat = np.zeros((n, n), dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            mat[i][j] = colors[idx]
            mat[j][i] = colors[idx]
            idx += 1
    
    # Check if valid
    tri_count = count_triangles(mat)
    
    if tri_count == 0:
        # Valid 49-vertex coloring!
        best_n = n
        best_colors = colors
    else:
        # Try to fix with local search
        time_budget = min(30.0, deadline - 5 - (time.time() - start_time))
        res_mat, success = local_search_sa(mat, n, time_budget)
        
        if success:
            best_n = n
            best_colors = []
            for i in range(n):
                for j in range(i+1, n):
                    best_colors.append(int(res_mat[i][j]))
        else:
            # Fallback to 32-vertex construction
            def get_seed_solution():
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
                colors = []
                for i in range(32):
                    for j in range(i + 1, 32):
                        colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
                return {'n': 32, 'colors': colors}
            
            seed_res = get_seed_solution()
            best_n = seed_res['n']
            best_colors = seed_res['colors']
    
    return {'n': best_n, 'colors': best_colors}
