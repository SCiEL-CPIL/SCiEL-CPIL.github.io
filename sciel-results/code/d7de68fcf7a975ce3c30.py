import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    def _edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (2 * n - i - 1) // 2 + j - i - 1

    def _count_c6_using_edge(u, v, adj):
        """
        Count the number of simple paths of length 4 from u to v in the graph defined by adj.
        The path is u-a-b-c-d-v.
        The edge (u,v) is effectively ignored in the path interior by skipping u and v.
        """
        count = 0
        # a must be neighbor of u, a != v
        for a in adj[u]:
            if a == v:
                continue
            # b must be neighbor of a, b != u, b != v
            for b in adj[a]:
                if b == u or b == v:
                    continue
                # c must be neighbor of b, c != u, c != v, c != a
                for c in adj[b]:
                    if c == u or c == v or c == a:
                        continue
                    # d must be neighbor of c, d != u, d != v, d != b
                    # and d must be neighbor of v
                    for d in adj[c]:
                        if d == u or d == v or d == b:
                            continue
                        # Check if d is connected to v
                        if d in adj[v]:
                            count += 1
        return count

    def _build_adj(n, colors):
        adj = [[[] for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[_edge_index(i, j, n)]
                adj[c][i].append(j)
                adj[c][j].append(i)
        return adj

    def _calc_cost(n, colors):
        adj = _build_adj(n, colors)
        cost = [0] * 4
        for c in range(4):
            sum_counts = 0
            for i in range(n):
                for j in adj[c][i]:
                    if i < j:
                        sum_counts += _count_c6_using_edge(i, j, adj[c])
            cost[c] = sum_counts // 6
        return sum(cost)

    def _verify(colors, n):
        if n < 6:
            return True
        adj = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        # Check for C6 in each color class
        # A C6 exists if there are 6 distinct vertices v0..v5 such that 
        # (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0) all have the same color.
        # We can check all combinations of 6 vertices.
        # For each combination, check if the induced subgraph contains a cycle of length 6.
        # Since n <= 20, C(20,6) = 38760. This is fast.
        
        # Precompute cycles on 0..5
        cycles = []
        seen = set()
        # Generate all permutations of 1..5, prepend 0
        # A cycle is a permutation (0, p1, p2, p3, p4, p5)
        # Edges: (0,p1), (p1,p2), (p2,p3), (p3,p4), (p4,p5), (p5,0)
        # We can just check all 6! = 720 permutations? No, 60 unique cycles.
        # But checking 720 is also fine.
        from itertools import permutations
        for perm in permutations(range(1, 6)):
            cycle = (0,) + perm
            edges = frozenset(frozenset((cycle[k], cycle[(k+1)%6])) for k in range(6))
            if edges not in seen:
                seen.add(edges)
                cycles.append(cycle)
        
        for c in range(4):
            for subset in combinations(range(n), 6):
                # Check if any cycle in 'cycles' is present in subset with color c
                found = False
                for cycle in cycles:
                    ok = True
                    for k in range(6):
                        u = subset[cycle[k]]
                        w = subset[cycle[(k+1)%6]]
                        if adj[u][w] != c:
                            ok = False
                            break
                    if ok:
                        found = True
                        break
                if found:
                    return False
        return True

    def _solve(n_target):
        if time.time() > deadline:
            return None
        
        num_edges = n_target * (n_target - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Initial cost
        total_cost = _calc_cost(n_target, colors)
        
        if total_cost == 0:
            return colors.copy()
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.9995
        steps = 0
        max_steps = 1000000 # High cap, time limited
        
        best_cost = total_cost
        best_colors = colors.copy()
        
        adj = _build_adj(n_target, colors)
        
        while time.time() < deadline - 5: # Stop 5s before deadline to allow return/verify
            steps += 1
            
            # Pick random edge
            i = random.randint(0, n_target - 1)
            j = random.randint(0, n_target - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            idx = _edge_index(i, j, n_target)
            old_c = colors[idx]
            new_c = random.randint(0, 3)
            if old_c == new_c:
                continue
            
            # Calculate delta cost
            # Remove edge (i,j) from old_c, add to new_c
            # Delta cost for old_c: - (count of C6s using (i,j) in old_c) / 6
            # Delta cost for new_c: + (count of C6s using (i,j) in new_c) / 6
            
            cnt_old = _count_c6_using_edge(i, j, adj[old_c])
            cnt_new = _count_c6_using_edge(i, j, adj[new_c])
            
            delta = (cnt_new - cnt_old) / 6.0
            
            # Acceptance
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                colors[idx] = new_c
                # Update adj lists
                adj[old_c][i].remove(j)
                adj[old_c][j].remove(i)
                adj[new_c][i].append(j)
                adj[new_c][j].append(i)
                
                total_cost += delta
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_colors = colors.copy()
                    
                if best_cost == 0:
                    return best_colors
            
            T *= cooling
            
            # Restart if stuck?
            if steps % 10000 == 0 and best_cost > 0:
                # Slight restart: randomize a few edges?
                # Or just continue cooling.
                pass

        return best_colors if best_cost == 0 else None

    # Try to find a valid coloring for n=17, then 16, 15, etc.
    # We want to maximize n.
    # Start with n=17. If we find one, return it.
    # If not, try n=16, etc.
    
    best_result = None
    best_n = 0
    
    # Prioritize higher n.
    # Since n=17 is the best known, we spend most time there.
    # If we fail n=17, we fall back to lower n which is easier.
    
    # Strategy:
    # 1. Try n=17 for 60s.
    # 2. If success, return.
    # 3. If fail, try n=16 for remaining time? Or just return best found?
    # The task says "Maximize n".
    # So if we find a valid n=17, we return it.
    # If we don't, we should try to find the largest n we can.
    # Since n=17 is the target, and n=16 is likely easy, we can just return n=16 if n=17 fails.
    # But wait, the seed returns n=9. We can easily do better.
    
    # Let's try n=17 first.
    # If time allows, we can try multiple restarts for n=17.
    
    # To be safe and fast, let's just run SA for n=17 until deadline-5s.
    # If it finds a solution, great.
    # If not, we try n=16.
    
    # Actually, n=16 is very likely to be found quickly.
    # Let's try n=17 for 80s. If found, return.
    # Else, try n=16 for remaining time.
    
    # But the evaluator kills at 120s.
    # So:
    # t=0 to t=80: try n=17
    # t=80 to t=115: try n=16 (if n=17 failed)
    
    # However, if we find n=17, we stop immediately.
    
    # Let's modify _solve to take a time limit.
    
    def _solve_with_limit(n_target, time_limit):
        start = time.time()
        limit = start + time_limit
        
        num_edges = n_target * (n_target - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        total_cost = _calc_cost(n_target, colors)
        if total_cost == 0:
            return colors.copy()
        
        T = 1.0
        cooling = 0.9995
        steps = 0
        best_cost = total_cost
        best_colors = colors.copy()
        adj = _build_adj(n_target, colors)
        
        while time.time() < limit:
            steps += 1
            i = random.randint(0, n_target - 1)
            j = random.randint(0, n_target - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            idx = _edge_index(i, j, n_target)
            old_c = colors[idx]
            new_c = random.randint(0, 3)
            if old_c == new_c:
                continue
            
            cnt_old = _count_c6_using_edge(i, j, adj[old_c])
            cnt_new = _count_c6_using_edge(i, j, adj[new_c])
            delta = (cnt_new - cnt_old) / 6.0
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                colors[idx] = new_c
                adj[old_c][i].remove(j)
                adj[old_c][j].remove(i)
                adj[new_c][i].append(j)
                adj[new_c][j].append(i)
                
                total_cost += delta
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        return best_colors
            T *= cooling
            
            # Random restarts within the time limit?
            # If no improvement for a while, restart?
            # Simple SA usually works.
        
        return best_colors if best_cost == 0 else None

    # Try n=17
    # Give it 70 seconds.
    colors_17 = _solve_with_limit(17, 70)
    if colors_17:
        return {'n': 17, 'colors': colors_17}
        
    # Try n=16
    # Give it remaining time, up to 40s.
    elapsed = time.time() - start_time
    remaining = max(1, deadline - elapsed - 5)
    colors_16 = _solve_with_limit(16, remaining)
    if colors_16:
        return {'n': 16, 'colors': colors_16}
        
    # Fallback: Try n=15
    elapsed = time.time() - start_time
    remaining = max(1, deadline - elapsed - 5)
    colors_15 = _solve_with_limit(15, remaining)
    if colors_15:
        return {'n': 15, 'colors': colors_15}
        
    # Fallback: n=9 seed
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

if __name__ == "__main__":
    res = construct()
    print(res)
