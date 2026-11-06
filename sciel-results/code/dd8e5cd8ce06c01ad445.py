import time
import random
import math
import sys
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def get_fallback(n):
        # For n <= 5, any coloring is valid (C6 requires 6 vertices)
        # We use a simple deterministic pattern for consistency
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append((i + j) % 4)
        return {'n': n, 'colors': colors}

    best_result = get_fallback(5)
    best_n = 5

    def precompute(n):
        """
        Precompute edge map and C6 cycles for a given n.
        """
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        
        m = idx
        edge_to_cycles = [[] for _ in range(m)]
        cycles = []
        
        if n < 6:
            return edge_map, cycles, edge_to_cycles

        vertices = list(range(n))
        
        # Generate all unique C6 cycles
        # A C6 is a sequence of 6 distinct vertices v0..v5 such that edges (v0,v1)..(v5,v0) exist.
        # In K_n, all such edges exist.
        # Number of distinct cycles on 6 labeled vertices is 5!/2 = 60.
        # We fix v0 to be the smallest vertex in the cycle to avoid duplicates? 
        # No, we need all cycles in K_n.
        # Standard way: iterate over all 6-subsets of vertices. For each subset, 
        # generate all distinct cyclic orderings.
        
        for combo in itertools.combinations(vertices, 6):
            v0 = combo[0]
            others = combo[1:]
            seen = set()
            for perm in itertools.permutations(others):
                # Cycle: v0, p0, p1, p2, p3, p4
                # To avoid reverse duplicates, we can enforce p0 < p4
                # (since the cycle is v0-p0-...-p4-v0, the reverse is v0-p4-...-p0-v0)
                if perm[0] > perm[-1]:
                    continue
                
                path = [v0] + list(perm)
                cycle_edges = []
                valid = True
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    if u > v:
                        u, v = v, u
                    e = edge_map[(u, v)]
                    cycle_edges.append(e)
                
                # Store
                cycle_id = len(cycles)
                cycles.append(cycle_edges)
                for e in cycle_edges:
                    edge_to_cycles[e].append(cycle_id)
        
        return edge_map, cycles, edge_to_cycles

    def search_n(n, time_budget):
        if time.time() > deadline - 5:
            return None
            
        try:
            edge_map, cycles, edge_to_cycles = precompute(n)
        except Exception:
            return None
            
        m = n * (n - 1) // 2
        num_cycles = len(cycles)
        
        if num_cycles == 0:
            # n < 6, trivial
            colors = [0] * m
            return {'n': n, 'colors': colors}
        
        # Initial random coloring
        colors = [random.randint(0, 3) for _ in range(m)]
        
        # Calculate initial bad count
        def calc_bad(colors):
            bad = 0
            for ce in cycles:
                c0 = colors[ce[0]]
                # Optimization: break early if not all match
                for e in ce[1:]:
                    if colors[e] != c0:
                        break
                else:
                    bad += 1
            return bad

        bad_count = calc_bad(colors)
        
        if bad_count == 0:
            return {'n': n, 'colors': colors}
        
        best_colors = colors[:]
        best_bad = bad_count
        
        # Simulated Annealing parameters
        T = 2.0
        cool_rate = 0.999
        max_iter = 100000
        iter_time_limit = time_budget
        
        start_iter_time = time.time()
        
        for i in range(max_iter):
            if time.time() > start_iter_time + iter_time_limit:
                break
            if time.time() > deadline - 5:
                break
                
            # Pick random edge
            e = random.randint(0, m - 1)
            c_old = colors[e]
            
            # Pick new color
            c_new = random.choice([c for c in range(4) if c != c_old])
            
            # Calculate delta
            delta = 0
            for ci in edge_to_cycles[e]:
                ce = cycles[ci]
                # Check if was mono
                was_mono = True
                for ed in ce:
                    if ed != e and colors[ed] != c_old:
                        was_mono = False
                        break
                # Check if will be mono
                will_be_mono = True
                for ed in ce:
                    if ed != e and colors[ed] != c_new:
                        will_be_mono = False
                        break
                
                if was_mono:
                    delta -= 1
                if will_be_mono:
                    delta += 1
            
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            else:
                # exp(-delta/T)
                if delta > 0:
                    # Optimization: if delta is large, prob is very small
                    if delta / T > 100:
                        accept = False
                    else:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            accept = True
            
            if accept:
                colors[e] = c_new
                bad_count += delta
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors[:]
                    if best_bad == 0:
                        return {'n': n, 'colors': best_colors}
            
            T *= cool_rate
            # Reheat occasionally to escape local minima
            if i % 5000 == 0:
                T = max(T, 1.0)

        # If we didn't find a perfect solution, check if best is valid
        if best_bad == 0:
            return {'n': n, 'colors': best_colors}
            
        return None

    # Targets to try
    # We want to maximize n.
    # Best known is 17. We try 17, 16, 15, etc.
    # We allocate time to higher n first.
    # If we find a valid coloring for n, we update best_result.
    # We continue to try smaller n only if we haven't found a valid one for larger n?
    # No, the goal is to MAXIMIZE n. So if we find n=17, we are done (unless we can prove 18 is possible, but 18 is likely impossible or very hard).
    # Actually, if we find n=17, that's the theoretical max (or close).
    # If we find n=16, we should try 17 again with more time?
    # Strategy: Try 17 with a large budget. If fail, try 16. If success, try 17 again with remaining time?
    # Or just return the highest n found.
    
    # Let's try 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6
    # We prioritize high n.
    
    targets = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    
    for n in targets:
        if time.time() > deadline - 10:
            break
        
        # Allocate time
        # n=17: 40s
        # n=16: 30s
        # n=15: 20s
        # n=14: 15s
        # n=13: 10s
        # n=12: 5s
        # n<12: 2s
        
        if n >= 17:
            budget = 40.0
        elif n >= 16:
            budget = 30.0
        elif n >= 15:
            budget = 20.0
        elif n >= 14:
            budget = 15.0
        elif n >= 13:
            budget = 10.0
        elif n >= 12:
            budget = 5.0
        else:
            budget = 2.0
            
        if time.time() + budget > deadline - 5:
            budget = deadline - 5 - time.time()
            if budget < 1.0:
                break

        result = search_n(n, budget)
        if result:
            if n > best_n:
                best_result = result
                best_n = n
                # If we found n=17, we can stop? 
                # It's the known max. Trying 18 is likely futile and risky.
                if n >= 17:
                    break
            # If we found a valid coloring for n, we don't need to try smaller n 
            # because we want to maximize n. 
            # However, if we failed at 17, we try 16. If we succeed at 16, 
            # should we try 17 again? 
            # Given time constraints, probably not. 
            # But if we have lots of time left, maybe.
            # For now, we break if we find a valid coloring for the current n, 
            # UNLESS we have a lot of time and n is small.
            # Actually, the loop structure "for n in targets" implies we try them in order.
            # If we find n=16, we update best. Then we try n=15? No, that's lower.
            # We should break if we find a valid coloring for n, because we are going down in n.
            # Wait, if we find n=16, we have a valid 16. We don't need 15.
            # So we should break.
            break

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
