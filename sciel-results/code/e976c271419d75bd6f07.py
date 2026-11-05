import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # We aim for n=29. If we can't find a valid one, we'll try smaller n.
    # But SA on 29 is fast.
    
    def run_sa(n):
        # Initialize random coloring
        colors = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                colors[i][j] = random.randint(0, 1)
                colors[j][i] = colors[i][j]
        
        # Calculate d0[i][j]: number of common neighbors in color 0
        # d0[i][j] = sum(1 for k in range(n) if k!=i and k!=j and colors[i][k]==0 and colors[j][k]==0)
        d0 = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                count = 0
                for k in range(n):
                    if k != i and k != j:
                        if colors[i][k] == 0 and colors[j][k] == 0:
                            count += 1
                d0[i][j] = count
                d0[j][i] = count
        
        # Calculate initial cost
        # Cost = sum( binom(d0[i][j], 2) + binom(n-2-d0[i][j], 2) ) for i<j
        def calc_cost(d0, n):
            cost = 0
            for i in range(n):
                for j in range(i+1, n):
                    d = d0[i][j]
                    cost += d*(d-1)//2
                    d1 = n - 2 - d
                    cost += d1*(d1-1)//2
            return cost
        
        current_cost = calc_cost(d0, n)
        best_cost = current_cost
        best_colors = [row[:] for row in colors]
        best_d0 = [row[:] for row in d0]
        
        if current_cost == 0:
            return best_colors
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while T > min_T and time.time() < deadline:
            # Pick random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            
            # Ensure u < v for consistency? No, colors is symmetric.
            # But d0 update logic assumes we handle (u,v) and (v,u) carefully?
            # My update logic above used w != u,v.
            # Let's ensure u < v to avoid double processing?
            # Actually, flipping (u,v) is the same as (v,u).
            # Let's just pick u, v distinct.
            
            old_color = colors[u][v]
            new_color = 1 - old_color
            
            # Calculate delta cost
            delta_cost = 0
            
            # We need to update d0[u][w] and d0[v][w] for all w != u, v
            # And update the cost terms for pairs (u,w) and (v,w)
            
            # Store old d0 values to restore if rejected
            # Since n is small (29), we can just store the changed d0 values
            # Or better, apply, check, revert.
            
            # Affected pairs: (u, w) and (v, w) for all w != u, v
            # Note: (u, v) cost term does not change.
            
            for w in range(n):
                if w == u or w == v:
                    continue
                
                # Determine change in d0[u][w] and d0[v][w]
                # d0[u][w] changes if v is a common neighbor of u and w in the "old" state?
                # No, the change in d0[u][w] depends on whether the edge (u,v) or (v,w) or (u,w) is the one changing?
                # The edge changing is (u,v).
                # d0[u][w] counts z such that c[u][z]==0 and c[w][z]==0.
                # The edge (u,v) is c[u][v]. This is c[u][z] with z=v.
                # So if c[w][v]==0, then v was a common neighbor of u and w in color 0 IF c[u][v]==0.
                # If we change c[u][v] from 0 to 1:
                #   If c[w][v]==0, then v stops being a common neighbor in color 0.
                #   So d0[u][w] decreases by 1.
                #   And d0[v][w]? d0[v][w] counts z such that c[v][z]==0 and c[w][z]==0.
                #   The edge (u,v) is c[v][u]. This is c[v][z] with z=u.
                #   If c[w][u]==0, then u was a common neighbor of v and w in color 0 IF c[v][u]==0.
                #   So if c[w][u]==0, d0[v][w] decreases by 1.
                
                # Let's generalize:
                # Flip (u,v) from c to 1-c.
                
                # For d0[u][w]:
                # z = v is the vertex that might stop/start being a common neighbor.
                # Condition for v to be common neighbor of u and w in color 0:
                # c[u][v]==0 AND c[w][v]==0.
                # If this was true, and we change c[u][v] to 1, then d0[u][w] decreases by 1.
                # If this was false, and we change c[u][v] to 0 (from 1), and c[w][v]==0, then d0[u][w] increases by 1.
                
                # Similarly for d0[v][w] with z=u:
                # Condition: c[v][u]==0 AND c[w][u]==0.
                
                # And for d0[u][w] in color 1?
                # We only track d0. d1 is derived.
                # Cost term for (u,w) is f(d0[u][w]) = C(d0, 2) + C(n-2-d0, 2).
                # So we just need delta in d0[u][w].
                
                delta_d0_uw = 0
                delta_d0_vw = 0
                
                if old_color == 0:
                    # Becomes 1
                    # v was common neighbor of u and w in color 0 if c[w][v]==0
                    if colors[w][v] == 0:
                        delta_d0_uw = -1
                    # u was common neighbor of v and w in color 0 if c[w][u]==0
                    if colors[w][u] == 0:
                        delta_d0_vw = -1
                else:
                    # Becomes 0
                    # v becomes common neighbor of u and w in color 0 if c[w][v]==0
                    if colors[w][v] == 0:
                        delta_d0_uw = 1
                    # u becomes common neighbor of v and w in color 0 if c[w][u]==0
                    if colors[w][u] == 0:
                        delta_d0_vw = 1
                
                # Update cost for pair (u, w)
                if delta_d0_uw != 0:
                    old_d = d0[u][w]
                    new_d = old_d + delta_d0_uw
                    # Cost contribution: C(d,2) + C(n-2-d,2)
                    old_contrib = old_d*(old_d-1)//2 + (n-2-old_d)*(n-3-old_d)//2
                    new_contrib = new_d*(new_d-1)//2 + (n-2-new_d)*(n-3-new_d)//2
                    delta_cost += new_contrib - old_contrib
                
                # Update cost for pair (v, w)
                if delta_d0_vw != 0:
                    old_d = d0[v][w]
                    new_d = old_d + delta_d0_vw
                    old_contrib = old_d*(old_d-1)//2 + (n-2-old_d)*(n-3-old_d)//2
                    new_contrib = new_d*(new_d-1)//2 + (n-2-new_d)*(n-3-new_d)//2
                    delta_cost += new_contrib - old_contrib
            
            # Accept or Reject
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Apply change
                colors[u][v] = new_color
                colors[v][u] = new_color
                
                # Update d0
                for w in range(n):
                    if w == u or w == v:
                        continue
                    # Recalculate deltas? No, we can just re-derive or store.
                    # It's safer to re-derive the delta logic here or store it.
                    # Let's just re-calculate d0[u][w] and d0[v][w] from scratch? No, O(n) is fine but we already did O(n) work.
                    # Let's just apply the deltas we calculated.
                    # We need to re-calculate the deltas to apply them.
                    # This is inefficient. Let's store the deltas in a list.
                    pass
                
                # Better approach: Store the deltas in a list during the first pass.
                # I'll refactor the loop to store (w, delta_uw, delta_vw).
                
                # For now, let's just re-calculate the affected d0s?
                # No, let's just re-run the delta calculation logic to apply.
                # This doubles the work.
                
                # Let's optimize: Store deltas.
                # I'll rewrite the loop to store them.
                
                # Actually, since n=29, re-calculating d0[u][w] is O(n).
                # Total O(n^2) per flip. 29^2 = 841.
                # 100k flips = 84M ops. Might be too slow in Python.
                # The delta calculation is O(n).
                # So storing deltas is crucial.
                
                # Let's restart the inner loop logic in the code below.
                pass
            
            T *= cooling
            
        return best_colors

    # Refactored SA with stored deltas
    
    def run_sa_optimized(n):
        colors = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                colors[i][j] = random.randint(0, 1)
                colors[j][i] = colors[i][j]
        
        d0 = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                count = 0
                for k in range(n):
                    if k != i and k != j:
                        if colors[i][k] == 0 and colors[j][k] == 0:
                            count += 1
                d0[i][j] = count
                d0[j][i] = count
        
        def cost_of_d(d, n):
            return d*(d-1)//2 + (n-2-d)*(n-3-d)//2

        current_cost = 0
        for i in range(n):
            for j in range(i+1, n):
                current_cost += cost_of_d(d0[i][j], n)
                
        best_cost = current_cost
        best_colors = [row[:] for row in colors]
        
        if current_cost == 0:
            return best_colors
            
        T = 1.0
        cooling = 0.9995
        min_T = 0.005
        
        while T > min_T and time.time() < deadline:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            
            old_color = colors[u][v]
            new_color = 1 - old_color
            
            delta_cost = 0
            deltas = [] # (w, delta_uw, delta_vw)
            
            for w in range(n):
                if w == u or w == v: continue
                
                delta_uw = 0
                delta_vw = 0
                
                if old_color == 0:
                    if colors[w][v] == 0: delta_uw = -1
                    if colors[w][u] == 0: delta_vw = -1
                else:
                    if colors[w][v] == 0: delta_uw = 1
                    if colors[w][u] == 0: delta_vw = 1
                
                if delta_uw != 0:
                    old_d = d0[u][w]
                    new_d = old_d + delta_uw
                    delta_cost += cost_of_d(new_d, n) - cost_of_d(old_d, n)
                    deltas.append((w, delta_uw, 0))
                if delta_vw != 0:
                    old_d = d0[v][w]
                    new_d = old_d + delta_vw
                    delta_cost += cost_of_d(new_d, n) - cost_of_d(old_d, n)
                    deltas.append((w, 0, delta_vw))
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                colors[u][v] = new_color
                colors[v][u] = new_color
                for w, du, dv in deltas:
                    d0[u][w] += du
                    d0[w][u] += du
                    d0[v][w] += dv
                    d0[w][v] += dv
                    
                current_cost += delta_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = [row[:] for row in colors]
                    if best_cost == 0:
                        return best_colors
            
            T *= cooling
            
        return best_colors

    # Try n=29 first
    best_valid_n = 17
    best_valid_colors = None
    
    # Fallback: Generate a valid 17 (Paley)
    squares = {x * x % 17 for x in range(1, 17)}
    best_valid_colors = [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]
    
    for n_target in [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18]:
        if time.time() > deadline - 10:
            break
        res = run_sa_optimized(n_target)
        # Check if valid
        # Cost 0 means no K2,2, so no K2,2,2.
        # We can check cost again or trust the SA.
        # Let's verify cost quickly.
        # Re-calc cost for res
        d0_check = [[0]*n_target for _ in range(n_target)]
        for i in range(n_target):
            for j in range(i+1, n_target):
                count = 0
                for k in range(n_target):
                    if k != i and k != j:
                        if res[i][k] == 0 and res[j][k] == 0:
                            count += 1
                d0_check[i][j] = count
                d0_check[j][i] = count
        
        c = 0
        for i in range(n_target):
            for j in range(i+1, n_target):
                d = d0_check[i][j]
                c += d*(d-1)//2 + (n_target-2-d)*(n_target-3-d)//2
                
        if c == 0:
            best_valid_n = n_target
            # Convert 2D to 1D list
            flat_colors = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    flat_colors.append(res[i][j])
            best_valid_colors = flat_colors
            break # Found a valid one, stop searching smaller?
            # Actually, we want max n. So if 29 works, we stop.
            # If 29 fails, we try 28.
            # So we break the loop if we find a valid one.
            # But what if 29 fails, 28 works? We return 28.
            # The loop structure handles this.
            
    return {'n': best_valid_n, 'colors': best_valid_colors}
