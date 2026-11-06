import time
import random
import math

def construct():
    deadline = time.time() + 110.0
    
    # Helper to check validity and count violations using bitsets
    # This is much faster than matrix multiplication or nested list checks in pure Python
    
    def count_violations_bitset(n, colors):
        """
        Returns the number of C4 violations.
        colors: list of length n*(n-1)//2, values 0-4.
        """
        if len(colors) != n * (n - 1) // 2:
            return float('inf')
            
        # neighbors[c][i] is a bitmask of neighbors of i in color c
        # Using list of lists of ints
        neighbors = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return float('inf')
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        total_violations = 0
        for c in range(5):
            # For each color, check all pairs
            # Optimization: skip if degree is low? No, just iterate.
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    if nv == 0:
                        continue
                    common = nu & nv
                    # Check if popcount >= 2
                    # common & (common - 1) is non-zero if there are at least 2 bits set
                    if common & (common - 1):
                        # Count exactly how many for the cost function
                        # int.bit_count() is available in Python 3.8+
                        cnt = common.bit_count()
                        total_violations += (cnt - 1)
                        
        return total_violations

    def is_valid_bitset(n, colors):
        return count_violations_bitset(n, colors) == 0

    def generate_cyclic_colors(n):
        """
        Generates a cyclic coloring.
        Distance d gets color (d-1) % 5? 
        For n=26, max distance is 13.
        We can map distances 1..13 to colors.
        """
        colors = []
        # Map distance to color. 
        # A simple heuristic: color = (distance - 1) % 5
        # This might have C4s, but it's a good structured start.
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                if d > n - d:
                    d = n - d
                c = (d - 1) % 5
                colors.append(c)
        return colors

    def generate_random_colors(n):
        m = n * (n - 1) // 2
        return [random.randint(0, 4) for _ in range(m)]

    def solve_sa(n, time_limit, init_type="random"):
        m = n * (n - 1) // 2
        if init_type == "cyclic":
            colors = generate_cyclic_colors(n)
        else:
            colors = generate_random_colors(n)
            
        current_cost = count_violations_bitset(n, colors)
        best_cost = current_cost
        best_colors = colors[:]
        
        if best_cost == 0:
            return best_colors

        T = 1.0
        alpha = 0.9995
        min_T = 0.01
        start_time = time.time()
        
        # Precompute edge indices for faster access? 
        # colors is a flat list. 
        # To pick a random edge, we pick a random index in [0, m-1].
        # But we need to know which (i,j) it corresponds to for potential optimizations?
        # Actually, for the bitset cost function, we just rebuild the neighbors from scratch.
        # This is O(n^2) per evaluation.
        # n=26 -> 676 edges. 5 colors.
        # Rebuilding neighbors: 325 iterations.
        # Checking violations: 5 * (26*25/2) = 1625 pair checks.
        # Total ~2000 simple ops. Very fast.
        
        while time.time() < start_time + time_limit:
            k = random.randint(0, m - 1)
            old_color = colors[k]
            new_color = random.randint(0, 4)
            
            if new_color == old_color:
                continue
            
            colors[k] = new_color
            new_cost = count_violations_bitset(n, colors)
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = colors[:]
                    if best_cost == 0:
                        return best_colors
            else:
                colors[k] = old_color
                
            T *= alpha
            if T < min_T:
                T = min_T
                
        return best_colors if best_cost == 0 else None

    # Strategy:
    # The known bound is 26. We should try hard to find 26.
    # If we fail, try 25, 24, etc.
    
    best_n = 11
    # Fallback seed solution (from current program logic, n=11)
    # The seed program likely produces n=11.
    # We will generate a valid n=11 coloring to be safe.
    # Cyclic coloring for n=11: distances 1..5. Colors 0..4.
    best_colors = []
    for i in range(11):
        for j in range(i + 1, 11):
            d = min(j - i, 11 - (j - i))
            best_colors.append(d - 1)
            
    # Verify fallback
    if is_valid_bitset(11, best_colors):
        best_n = 11
    else:
        # Random search for 11 just in case
        for _ in range(1000):
            c = generate_random_colors(11)
            if is_valid_bitset(11, c):
                best_colors = c
                break

    # Try to improve
    # Allocate time to higher n first
    # n=26 is the target.
    
    # Time allocation:
    # 60s for n=26
    # 20s for n=25
    # 10s for n=24
    # 5s for n=23
    # 5s for n=22
    # 10s remaining for buffer/verification
    
    attempts = [
        (26, 60),
        (25, 20),
        (24, 10),
        (23, 5),
        (22, 5),
    ]
    
    for n, t_alloc in attempts:
        if time.time() > deadline - 2:
            break
            
        t_alloc = min(t_alloc, deadline - time.time() - 1)
        if t_alloc < 1:
            continue
            
        # Try multiple restarts if time permits?
        # For n=26, one long run with reheat is probably better.
        # The solve_sa function handles cooling and reheating implicitly via min_T?
        # No, it just cools. Let's add a restart mechanism inside or just rely on the long time.
        # With 60s, we can do ~100,000-500,000 iterations.
        
        # Try cyclic first, then random?
        # Cyclic is more structured.
        res = solve_sa(n, t_alloc, init_type="cyclic")
        
        if res:
            if is_valid_bitset(n, res):
                best_n = n
                best_colors = res
                # If we found 26, we can return immediately as it's the known max (or close to it).
                # The task says "27 or 28 would be new", implying 26 is the current record.
                # So finding 26 is the best we can likely do.
                if n >= 26:
                    return {'n': best_n, 'colors': best_colors}
            else:
                # Should not happen if solve_sa returns None on fail
                pass
        
        # If cyclic failed, try random with remaining time for this n?
        # To keep it simple and fast, we just move to next n if cyclic fails.
        # But for n=26, it's crucial. Let's do a second random attempt if time allows.
        if n == 26:
            time_left = deadline - time.time() - 2
            if time_left > 5:
                res2 = solve_sa(n, time_left, init_type="random")
                if res2:
                    if is_valid_bitset(n, res2):
                        best_n = n
                        best_colors = res2
                        return {'n': best_n, 'colors': best_colors}

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
