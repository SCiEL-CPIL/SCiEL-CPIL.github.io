import time
import random
import sys

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4.
    Maximizes n.
    """
    deadline = time.time() + 110.0  # Safety margin for 120s limit
    
    # Known seed solution for n=11 (cyclic group Z11, colors based on difference)
    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()
    
    # Helper to get edge index in lexicographic order
    def get_edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (i - 1) // 2 + (j - i - 1)

    def is_valid(n, colors):
        """
        Verifies if the coloring has no monochromatic C4.
        Returns True if valid.
        """
        if len(colors) != n * (n - 1) // 2:
            return False
        
        # Build neighbor bitsets for each color
        # neighbors[c][u] is an integer bitmask of vertices connected to u with color c
        neighbors = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return False
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        # Check for C4:
        # A C4 exists if there are two vertices u, v that share at least 2 common neighbors in any color c.
        # Common neighbors of u and v in color c: (neighbors[c][u] & neighbors[c][v])
        # We need to ensure popcount(common) < 2 for all u != v and all c.
        
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    # Check if popcount(common) >= 2
                    # Fast check: if common has at least 2 bits set
                    # common & (common - 1) removes the lowest set bit.
                    # If result is non-zero, there were at least 2 bits.
                    if common & (common - 1):
                        return False
        return True

    def count_violations(n, colors):
        """
        Counts the total number of monochromatic C4s (pairs of vertices with >= 2 common neighbors in a color).
        """
        neighbors = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        cost = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common & (common - 1):
                        cost += 1
        return cost

    def solve_sa(n, time_limit):
        """
        Runs Simulated Annealing to find a valid coloring for K_n.
        """
        total_edges = n * (n - 1) // 2
        
        # Initial random coloring
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        # Precompute edge endpoints for fast access
        # edge_idx -> (u, v)
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[idx] = (i, j)
                idx += 1
        
        # To speed up cost evaluation, we maintain the neighbor bitsets.
        # But updating them on every move and recalculating full cost is O(n^2).
        # For n=26, n^2 = 676. 676 operations per move is very fast in Python if using bit ops.
        # However, we can optimize by only checking the cost change locally?
        # Changing edge (u,v) from c1 to c2 affects:
        # 1. Pairs involving u or v.
        # 2. Specifically, common neighbor counts for (u, w) and (v, w).
        
        # Let's stick to full recalculation first for simplicity and correctness, 
        # as n is small enough (<=29). 
        # 29^2 * 5 = 4205 bit operations. Very fast.
        
        current_cost = count_violations(n, colors)
        
        # If we start with 0 violations, we are done (unlikely for random start of large n)
        if current_cost == 0:
            return colors
            
        best_cost = current_cost
        best_colors = colors[:]
        
        # SA Parameters
        T = 1.0
        cooling_rate = 0.9995
        # We will run until time limit
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Pick a random edge
            e_idx = random.randint(0, total_edges - 1)
            u, v = edge_map[e_idx]
            old_c = colors[e_idx]
            
            # Pick a new color different from old
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
                
            # Apply move
            colors[e_idx] = new_c
            
            # Calculate new cost
            new_cost = count_violations(n, colors)
            
            # Acceptance probability
            delta = new_cost - current_cost
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    if best_cost == 0:
                        return best_colors
            else:
                # Reject
                colors[e_idx] = old_c
                
            T *= cooling_rate
            
            # If T is very small, we might get stuck. 
            # Reheat strategy: if no improvement for many steps, increase T?
            # Or just let it cool. With many restarts, this is fine.
            
        return best_colors if best_cost == 0 else None

    # Strategy:
    # Try n from 29 down to 12.
    # For each n, run SA for a few seconds.
    # If found, return.
    
    # Time allocation
    # n=29, 28, 27: 5s each
    # n=26: 20s (known to be possible, but SA might need more time to find it from random)
    # n=25..12: 5s each
    
    # Note: SA from random start for n=26 might be hard.
    # We can improve by starting from a "structured" random start or using a better initialization.
    # For now, random is the baseline.
    
    # To improve success rate for n=26, we can try multiple restarts within the time limit.
    
    for n in range(29, 11, -1):
        if time.time() > deadline - 5:
            break
            
        if n >= 27:
            t_alloc = 5.0
        elif n == 26:
            t_alloc = 30.0
        else:
            t_alloc = 5.0
            
        # Ensure we don't exceed deadline
        t_alloc = min(t_alloc, deadline - time.time() - 2.0)
        if t_alloc <= 0:
            break
            
        # Run solver with restarts
        start = time.time()
        found = False
        # For n=26, we might want more restarts if it fails quickly?
        # Let's just run one long SA instance per n for now, or multiple short ones?
        # The current solve_sa runs until time_limit.
        # So we can just call it once with the allocated time.
        
        res = solve_sa(n, t_alloc)
        
        if res:
            # Verify solution just in case
            if is_valid(n, res):
                best_n = n
                best_colors = res
                # Since we go high to low, this is the max found so far.
                # We can return immediately if we want to save time?
                # Yes, because we are searching downwards.
                return {'n': best_n, 'colors': best_colors}
                
    # If all fail, return seed
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Found n={res['n']}")
