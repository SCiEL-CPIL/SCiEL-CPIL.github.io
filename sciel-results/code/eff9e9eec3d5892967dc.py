import time
import random

def construct():
    # Seed result
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
    seed_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            seed_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    best_n = 32
    best_colors = seed_colors
    deadline = time.time() + 145.0
    
    # Helper to check validity and count triangles
    def count_triangles(colors, n):
        # colors is list of length n(n-1)/2
        # Map (i,j) to index
        # We can just iterate all triples
        # For speed, precompute edge indices
        # But for n=33, N=528. Triples ~ 5456.
        # 5456 checks per evaluation.
        # Local search moves: 528 edges * 3 colors = 1584 possible moves.
        # This is fast enough.
        
        # Efficient check:
        # For each triangle (i,j,k), check if colors are same.
        # Index for (i,j) where i<j: 
        # idx = i*n - i*(i+1)//2 + (j-i-1)
        
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                idx_ij = i*n - i*(i+1)//2 + (j-i-1)
                c_ij = colors[idx_ij]
                for k in range(j+1, n):
                    idx_ik = i*n - i*(i+1)//2 + (k-i-1)
                    idx_jk = j*n - j*(j+1)//2 + (k-j-1)
                    if colors[idx_ik] == c_ij and colors[idx_jk] == c_ij:
                        count += 1
        return count

    def get_idx(i, j, n):
        if i > j: i, j = j, i
        return i*n - i*(i+1)//2 + (j-i-1)

    # Local Search for a specific n
    def local_search(n, time_limit):
        N = n*(n-1)//2
        colors = [random.randint(0, 3) for _ in range(N)]
        
        # Initial count
        curr_count = count_triangles(colors, n)
        
        if curr_count == 0:
            return colors
        
        start_time = time.time()
        last_improve = start_time
        
        while time.time() < start_time + time_limit:
            # Pick random edge
            edge_idx = random.randint(0, N-1)
            # Find i, j for edge_idx
            # Inverse of get_idx
            # i is approx sqrt(2*edge_idx)
            # i*n - i(i+1)/2 <= edge_idx
            # Let's just precompute a map or use a simple loop since N is small
            # Or better, precompute edge_list
            # For efficiency in the loop, let's precompute edges
            # But we are inside a function called multiple times.
            # Let's precompute edges for this n once outside the while loop.
            
            # Optimization: Precompute edges for this n
            # This is done outside the while loop in the actual implementation below.
            # For now, assume we have edges list.
            
            # Let's restructure to precompute edges.
            pass 
        return None # Failed

    # Better implementation of local search with precomputation
    def try_solve(n, time_limit):
        N = n*(n-1)//2
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                edges.append((i, j))
        
        # Precompute for each edge, the list of other vertices w
        # And the indices of the other two edges in the triangle
        edge_neighbors = []
        for idx, (i, j) in enumerate(edges):
            neighbors = []
            for w in range(n):
                if w == i or w == j: continue
                # Triangle (i, j, w)
                # Need indices for (i,w) and (j,w)
                # Ensure i<w and j<w for index calc? No, get_idx handles it.
                # But for speed, we can store the indices directly.
                idx_iw = get_idx(i, w, n)
                idx_jw = get_idx(j, w, n)
                neighbors.append((idx_iw, idx_jw))
            edge_neighbors.append(neighbors)
            
        # Random init
        colors = [random.randint(0, 3) for _ in range(N)]
        curr_count = count_triangles(colors, n)
        
        if curr_count == 0:
            return colors
            
        start_time = time.time()
        
        while time.time() < start_time + time_limit:
            # Pick random edge
            e_idx = random.randint(0, N-1)
            i, j = edges[e_idx]
            old_c = colors[e_idx]
            
            # Try other 3 colors
            best_new_c = -1
            best_new_count = curr_count
            
            for new_c in range(4):
                if new_c == old_c: continue
                
                # Calculate delta
                delta = 0
                for idx_iw, idx_jw in edge_neighbors[e_idx]:
                    c_iw = colors[idx_iw]
                    c_jw = colors[idx_jw]
                    
                    # Before: mono if old_c == c_iw == c_jw
                    if old_c == c_iw and old_c == c_jw:
                        delta -= 1
                    # After: mono if new_c == c_iw == c_jw
                    if new_c == c_iw and new_c == c_jw:
                        delta += 1
                        
                new_count = curr_count + delta
                if new_count < best_new_count:
                    best_new_count = new_count
                    best_new_c = new_c
                    
            if best_new_c != -1 and best_new_count < curr_count:
                colors[e_idx] = best_new_c
                curr_count = best_new_count
                
                if curr_count == 0:
                    return colors
            
            # If no improvement, maybe random walk?
            # For now, just accept any move that doesn't increase count?
            # Or just keep trying.
            
        return None

    # Main loop
    for n in range(33, 63):
        if time.time() > deadline - 5:
            break
        # Try to solve for n with a time slice
        # Give 2 seconds per n?
        # If n=33 is hard, it might fail.
        # Let's give 5 seconds.
        t0 = time.time()
        result = try_solve(n, 5.0)
        if result is not None:
            best_n = n
            best_colors = result
            # Continue to next n? Or keep searching for higher n?
            # The task says "Maximize n". So we should keep going.
            # But if we found 33, we should try 34.
            # If we don't find 34, we return 33.
            # So we don't break, we just update best.
        else:
            # If we fail to improve n, maybe we should stop?
            # It's unlikely that if 33 is not found in 5s, 34 will be.
            # But random restarts might work.
            # Let's just break to be safe and return best found.
            break
            
    return {'n': best_n, 'colors': best_colors}
