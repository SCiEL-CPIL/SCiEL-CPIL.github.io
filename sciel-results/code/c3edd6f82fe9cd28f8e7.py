import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    def popcount(x):
        return bin(x).count('1')

    def get_idx(i, j, n):
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def build_adjacency(n, colors):
        N = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j, n)]
                N[c][i] |= (1 << j)
                N[c][j] |= (1 << i)
        return N

    def count_violations(n, colors, N):
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j, n)]
                if popcount(N[c][i] & N[c][j]) >= 2:
                    violations += 1
        return violations

    def is_violating(u, v, c, N):
        return popcount(N[c][u] & N[c][v]) >= 2

    def sa_search(n, max_time=10.0, seed=None):
        if seed is not None:
            random.seed(seed)
        
        n_edges = n * (n - 1) // 2
        if n_edges == 0:
            return [], 0
            
        # Initialize
        # Try a few random restarts or structured
        best_colors = None
        best_violations = float('inf')
        
        # Structured init: random
        colors = [random.randint(0, 4) for _ in range(n_edges)]
        N = build_adjacency(n, colors)
        violations = count_violations(n, colors, N)
        
        if violations == 0:
            return colors, 0
            
        T = 1.0
        cooling = 0.995
        current_time = time.time()
        
        # Precompute edge list for random access? 
        # Or just pick random i, j?
        # Picking random i, j is O(1).
        
        while time.time() - current_time < max_time:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            idx = get_idx(i, j, n)
            c_old = colors[idx]
            
            # Pick new color
            c_new = random.randint(0, 4)
            if c_new == c_old:
                continue
            
            # Calculate Delta
            delta = 0
            
            # 1. Edge (i, j) itself
            # Old status
            old_viol_ij = 1 if is_violating(i, j, c_old, N) else 0
            
            # Temporarily update N to check new status? 
            # Or calculate directly.
            # New status for (i, j) with color c_new:
            # N[c_new][i] currently does NOT have j. We are adding it.
            # N[c_new][j] currently does NOT have i. We are adding it.
            # Common neighbors of i and j in color c_new AFTER move:
            # (N[c_new][i] | (1<<j)) & (N[c_new][j] | (1<<i))
            # Note: i is not in N[c_new][i] and j is not in N[c_new][j] initially.
            # After move:
            # N_new_c_new_i = N[c_new][i] | (1 << j)
            # N_new_c_new_j = N[c_new][j] | (1 << i)
            # Intersection:
            # (N[c_new][i] & N[c_new][j]) | (N[c_new][i] & (1<<i)) | ((1<<j) & N[c_new][j]) | ((1<<j) & (1<<i))
            # Since i not in N[c_new][i], term 2 is 0.
            # Since j not in N[c_new][j], term 3 is 0.
            # Term 4 is 0 (i!=j).
            # So intersection is N[c_new][i] & N[c_new][j].
            # Wait, does the new edge (i,j) create a common neighbor?
            # A common neighbor k must be connected to i with c_new and to j with c_new.
            # The edge (i,j) itself is not a neighbor.
            # So yes, the common neighbors are exactly the old common neighbors in color c_new.
            # UNLESS i or j were already connected? No, it's a complete graph, each pair has exactly one edge.
            # So N[c_new][i] does not contain j.
            # So the set of common neighbors for (i,j) in color c_new is simply N[c_new][i] & N[c_new][j].
            
            new_viol_ij = 1 if popcount(N[c_new][i] & N[c_new][j]) >= 2 else 0
            
            delta += new_viol_ij - old_viol_ij
            
            # 2. Edges (i, w) for w != j
            # Affected if color(i, w) is c_old or c_new.
            for w in range(n):
                if w == i or w == j:
                    continue
                idx_ew = get_idx(min(i, w), max(i, w), n)
                c_ew = colors[idx_ew]
                
                if c_ew == c_old:
                    # N[c_old][i] loses j.
                    # Old intersection: N[c_old][i] & N[c_old][w]
                    # New intersection: (N[c_old][i] & ~(1<<j)) & N[c_old][w]
                    # Old common = N[c_old][i] & N[c_old][w]
                    # New common = Old common & ~(1<<j)
                    # Violation changes only if j was a common neighbor and removing it drops count from 2 to 1 (or >2 to >1? No, if count was 2, removing 1 makes it 1, so violation removed. If count was 3, removing 1 makes it 2, still violation).
                    old_common = N[c_old][i] & N[c_old][w]
                    old_viol = 1 if popcount(old_common) >= 2 else 0
                    new_common = old_common & ~(1 << j)
                    new_viol = 1 if popcount(new_common) >= 2 else 0
                    delta += new_viol - old_viol
                    
                elif c_ew == c_new:
                    # N[c_new][i] gains j.
                    # Old intersection: N[c_new][i] & N[c_new][w]
                    # New intersection: (N[c_new][i] | (1<<j)) & N[c_new][w]
                    # = Old intersection | (N[c_new][w] & (1<<j))
                    # If j is in N[c_new][w], then j becomes a new common neighbor.
                    old_common = N[c_new][i] & N[c_new][w]
                    old_viol = 1 if popcount(old_common) >= 2 else 0
                    # Check if j is neighbor of w in color c_new
                    if N[c_new][w] & (1 << j):
                        new_common = old_common | (1 << j)
                    else:
                        new_common = old_common
                    new_viol = 1 if popcount(new_common) >= 2 else 0
                    delta += new_viol - old_viol

            # 3. Edges (j, w) for w != i
            for w in range(n):
                if w == i or w == j:
                    continue
                idx_ew = get_idx(min(j, w), max(j, w), n)
                c_ew = colors[idx_ew]
                
                if c_ew == c_old:
                    # N[c_old][j] loses i.
                    old_common = N[c_old][j] & N[c_old][w]
                    old_viol = 1 if popcount(old_common) >= 2 else 0
                    new_common = old_common & ~(1 << i)
                    new_viol = 1 if popcount(new_common) >= 2 else 0
                    delta += new_viol - old_viol
                    
                elif c_ew == c_new:
                    # N[c_new][j] gains i.
                    old_common = N[c_new][j] & N[c_new][w]
                    old_viol = 1 if popcount(old_common) >= 2 else 0
                    if N[c_new][w] & (1 << i):
                        new_common = old_common | (1 << i)
                    else:
                        new_common = old_common
                    new_viol = 1 if popcount(new_common) >= 2 else 0
                    delta += new_viol - old_viol

            # Acceptance
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                # Apply move
                N[c_old][i] &= ~(1 << j)
                N[c_old][j] &= ~(1 << i)
                N[c_new][i] |= (1 << j)
                N[c_new][j] |= (1 << i)
                colors[idx] = c_new
                violations += delta
                
                if violations == 0:
                    return colors, 0
                    
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors[:]
            
            T *= cooling
            if T < 0.01:
                T = 0.01

        return best_colors, best_violations

    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to find larger n
    # Start from 12 up to 29
    # Allocate time based on n? Or fixed time per n?
    # If we find n=12, we can use it to seed n=13? Hard to extend.
    # Just run SA for each n.
    
    # Time budget: 140s total.
    # Let's try n=12, 13, ..., 26.
    # If we find a valid one, update best.
    
    for n in range(12, 27):
        if time.time() > deadline:
            break
            
        # Try a few restarts for this n
        # Time per n: 140 / 15 ~ 9s?
        # But we should prioritize smaller n first as they are easier.
        # If we fail n=12, maybe n=13 is also hard?
        # Actually, if n=12 is valid, n=11 is valid.
        # The constraint is "Maximize n".
        
        # Strategy: Run SA for n. If found, save. 
        # If not found within time slice, move to next n? 
        # Or stick with n until time runs out?
        # Sticking with one n is better because SA needs time to converge.
        # But if n=12 is impossible (it's not, 12 is likely possible), we waste time.
        # Let's give 5s to each n in a loop, but if we find valid, break.
        # If we don't find valid for n=12 in 5s, try n=13? 
        # No, if n=12 is hard, n=13 is harder.
        # So we should try n=12, 13... until we find one or time runs out.
        # But if n=12 is very hard, we might miss n=26 which is algebraic?
        # No, 26 is the target.
        
        # Let's try a few restarts for the current n.
        found_valid = False
        attempts = 0
        time_limit_for_n = 10.0
        start_n_time = time.time()
        
        while time.time() - start_n_time < time_limit_for_n:
            if time.time() > deadline:
                break
                
            colors, viol = sa_search(n, max_time=5.0)
            attempts += 1
            
            if viol == 0:
                best_n = n
                best_colors = colors
                found_valid = True
                break
            else:
                # Keep best found so far for this n?
                # If we don't find valid, we can't use it.
                pass
        
        if found_valid:
            # If we found n, try n+1?
            continue
        else:
            # If we didn't find n, maybe n+1 is also hard.
            # But we should keep trying higher n if we have time?
            # No, if n is hard, n+1 is harder.
            # So if we fail n=12, we probably fail n=13.
            # However, random search might fail n=12 by chance.
            # Let's stick to increasing n only if we succeed?
            # Or just try all n in order, giving each a time slice.
            # If we fail 12, we move to 13? That seems wrong.
            # Better: Try n=12. If success, try 13.
            # If fail 12, try 12 again with more time?
            # Let's just loop n=12..26 and give each 10s.
            # If we find 12, we update best_n=12.
            # Then we try 13. If we find 13, best_n=13.
            # If we fail 13, we still have best_n=12.
            # This is safe.
            pass

    return {'n': best_n, 'colors': best_colors}
