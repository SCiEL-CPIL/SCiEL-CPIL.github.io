import time
import random

def construct():
    start_time = time.time()
    deadline = 120.0

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

    def solve(n):
        # Initialize
        colors = [random.randint(0, 3) for _ in range(n*(n-1)//2)]
        # Build adj matrix
        adj = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
                
        # Precompute triangle count for each edge?
        # Let's just do a full count first
        def full_count():
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    c = adj[i][j]
                    for k in range(j+1, n):
                        if adj[i][k] == c and adj[j][k] == c:
                            count += 1
            return count

        curr_count = full_count()
        if curr_count == 0:
            return colors

        best_count = curr_count
        best_colors = colors[:]

        # Helper to get index
        def get_idx(i, j):
            if i > j: i, j = j, i
            return i*n - i*(i+1)//2 + (j-i-1)

        # Local search with O(n) updates
        # For each edge (i, j), the triangles are (i, j, k).
        # If we change color of (i, j) from c_old to c_new:
        # For each k:
        #   old contribution: 1 if adj[i][k]==c_old and adj[j][k]==c_old else 0
        #   new contribution: 1 if adj[i][k]==c_new and adj[j][k]==c_new else 0
        #   delta = new - old
        # total_delta = sum(delta for all k)
        
        iterations = 20000
        for _ in range(iterations):
            if time.time() > deadline:
                break
            if curr_count == 0:
                return colors

            # Pick random edge
            idx = random.randint(0, len(colors)-1)
            # Decode idx to i, j
            # This is slow. Let's pick i, j directly.
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            c_old = adj[i][j]
            c_new = (c_old + random.randint(1, 3)) % 4
            if c_new == c_old:
                continue

            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                # Check old
                if adj[i][k] == c_old and adj[j][k] == c_old:
                    delta -= 1
                # Check new
                if adj[i][k] == c_new and adj[j][k] == c_new:
                    delta += 1
            
            new_count = curr_count + delta
            
            if new_count < curr_count:
                # Accept
                adj[i][j] = c_new
                adj[j][i] = c_new
                idx = get_idx(i, j)
                colors[idx] = c_new
                curr_count = new_count
                if new_count < best_count:
                    best_count = new_count
                    best_colors = colors[:]
            else:
                # Reject (or accept with prob if SA, but let's stick to HC)
                pass

        return best_colors if best_count == 0 else None

    # Try n=34, 33
    for n in [34, 33]:
        result = solve(n)
        if result is not None:
            return {'n': n, 'colors': result}
            
    return get_seed_solution()
