import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115.0
    
    # Fallback: The known n=32 construction
    def get_seed_32():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        
        log = {}
        x = 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
            
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return 32, colors

    def get_idx(i, j, n):
        if i > j: i, j = j, i
        return i*n - i*(i+1)//2 + (j-i-1)

    def try_grow(n_target, time_limit):
        # Start with n=32
        n, colors_list = get_seed_32()
        colors = np.array(colors_list, dtype=np.int8)
        
        # Precompute triangle neighbors for the current graph?
        # We will update this as we grow.
        # For efficiency, we will just check triangles involving the new vertex.
        
        # We need a mapping from (i,j) to index in colors array.
        # Since n changes, the indices change.
        # It is easier to keep a 2D array or a dictionary.
        # But the output requires a flat list in lexicographic order.
        # We can maintain a 2D matrix `M` of size n_target x n_target.
        
        M = np.zeros((n_target, n_target), dtype=np.int8)
        
        # Fill M with the seed 32 coloring
        idx = 0
        for i in range(32):
            for j in range(i+1, 32):
                M[i,j] = colors[idx]
                M[j,i] = colors[idx]
                idx += 1
                
        current_n = 32
        
        # Helper to count mono-triangles involving a specific vertex k
        def count_mono_triangles_with_k(k, M, current_n):
            count = 0
            for i in range(current_n):
                if i == k: continue
                for j in range(i+1, current_n):
                    if j == k: continue
                    c_ki = M[k, i]
                    c_kj = M[k, j]
                    c_ij = M[i, j]
                    if c_ki == c_kj and c_kj == c_ij:
                        count += 1
            return count

        # Helper to get list of mono-triangles involving k
        def get_mono_triangles_with_k(k, M, current_n):
            tris = []
            for i in range(current_n):
                if i == k: continue
                for j in range(i+1, current_n):
                    if j == k: continue
                    c_ki = M[k, i]
                    c_kj = M[k, j]
                    c_ij = M[i, j]
                    if c_ki == c_kj and c_kj == c_ij:
                        tris.append((i, j))
            return tris

        start_time = time.time()
        
        for k in range(current_n, n_target):
            # Initialize colors for edges (k, 0..k-1)
            # Try random initialization
            found = False
            attempts = 0
            max_attempts = 20
            
            while time.time() < start_time + time_limit and not found and attempts < max_attempts:
                attempts += 1
                # Random init for new edges
                for i in range(k):
                    M[k, i] = np.random.randint(0, 4)
                    M[i, k] = M[k, i]
                
                # Local search to fix triangles
                curr_violations = get_mono_triangles_with_k(k, M, k)
                
                # We only care about triangles involving k.
                # Triangles not involving k are already valid.
                
                T = 1.0
                cooling = 0.99
                no_improve = 0
                max_no_improve = 5000
                
                while time.time() < start_time + time_limit and no_improve < max_no_improve:
                    if len(curr_violations) == 0:
                        found = True
                        break
                    
                    # Pick a random violating triangle (i, j)
                    i, j = random.choice(curr_violations)
                    
                    # We can change M[k,i], M[k,j], or M[i,j].
                    # Changing M[i,j] affects other triangles not involving k?
                    # Yes, but we want to keep the existing graph valid.
                    # So we should prefer changing M[k,i] or M[k,j].
                    
                    # Let's try changing M[k,i]
                    old_ci = M[k, i]
                    best_new_ci = -1
                    best_delta = 0
                    
                    # Calculate delta for changing M[k,i]
                    # Triangles involving (k,i) are (k,i,x) for x != k,i.
                    # We need to check which of these are currently mono and which become mono.
                    
                    # This is complex to vectorize easily in Python loops.
                    # Given the small size (k <= 50), a simple loop is fine.
                    
                    # Current count for triangles involving (k,i)
                    # A triangle (k,i,x) is mono if M[k,i]==M[k,x]==M[i,x].
                    
                    # Let's just re-evaluate the violation count for the specific edge change?
                    # Or simpler:
                    # Try all 4 colors for M[k,i].
                    best_violations = len(curr_violations)
                    best_color = old_ci
                    
                    for new_c in range(4):
                        if new_c == old_ci: continue
                        M[k, i] = new_c
                        M[i, k] = new_c
                        # Recalculate violations for triangles involving k and i?
                        # Actually, we can just recalculate the full violation count for k.
                        # It's O(k^2) which is 2500 ops. Very fast.
                        new_violations = get_mono_triangles_with_k(k, M, k)
                        if len(new_violations) < best_violations:
                            best_violations = len(new_violations)
                            best_color = new_c
                            
                    if best_color != old_ci:
                        M[k, i] = best_color
                        M[i, k] = best_color
                        curr_violations = get_mono_triangles_with_k(k, M, k)
                        no_improve = 0
                    else:
                        no_improve += 1
                        # If stuck, maybe try changing M[k,j] or M[i,j]?
                        # For now, just restart or change strategy.
                        if no_improve > 100:
                            # Try changing a random edge connected to k
                            i2 = random.randint(0, k-1)
                            old_c = M[k, i2]
                            new_c = (old_c + 1) % 4
                            M[k, i2] = new_c
                            M[i2, k] = new_c
                            curr_violations = get_mono_triangles_with_k(k, M, k)
                            no_improve = 0
                            
                    T *= cooling
                    if T < 0.01: T = 0.01

            if not found:
                # Failed to add vertex k
                # We can try to continue with a smaller n_target?
                # Or just return what we have.
                # Since we are growing, if we fail at 33, we return 32.
                # If we fail at 40, we return 39?
                # We should return the largest valid n found.
                # So we break and return the current state.
                break
            
            current_n = k + 1
            
        # Extract colors in lexicographic order
        final_colors = []
        for i in range(current_n):
            for j in range(i+1, current_n):
                final_colors.append(int(M[i,j]))
                
        return current_n, final_colors

    # Try to grow to 50
    # If it fails, it returns the largest n it managed to reach (at least 32)
    # We should also try to find a better seed or different strategy if growth fails quickly?
    # But growth is very promising.
    
    n, colors = try_grow(50, 110.0)
    
    return {'n': n, 'colors': colors}
