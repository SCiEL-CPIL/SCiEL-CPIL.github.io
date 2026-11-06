import time
import random
import numpy as np

def _paley(n, p):
    """Paley coloring using prime p >= n."""
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def _count_k222_np(M, n):
    """
    Count K2,2,2 subgraphs in graph M (adjacency matrix).
    M is a numpy array of 0s and 1s.
    """
    count = 0
    # Iterate over all non-edges (u, v)
    # These are candidates for being in the same part
    for u in range(n):
        for v in range(u + 1, n):
            if M[u, v] == 1:
                continue
            
            # Common neighbors of u and v
            # S = { w | M[u,w]=1 and M[v,w]=1 }
            # We can use bit operations or numpy
            # For n=29, simple loop is fine, but let's use numpy for intersection
            # N_u = M[u], N_v = M[v]
            # S_mask = N_u & N_v
            # But we need to exclude u and v themselves?
            # M[u,u]=0, M[v,v]=0.
            # M[u,v]=0, M[v,u]=0.
            # So u and v are not in S.
            
            common = M[u] & M[v]
            # Get indices of 1s in common
            verts = np.where(common)[0]
            m = len(verts)
            if m < 4:
                continue
            
            # Iterate over pairs (w1, w2) in verts
            # These form a candidate part 2
            for i in range(m - 1):
                w1 = verts[i]
                # w1, w2 must be non-adjacent in M
                for j in range(i + 1, m):
                    w2 = verts[j]
                    if M[w1, w2] == 1:
                        continue
                    
                    # Now we have part 1 = {u,v}, part 2 = {w1,w2}
                    # We need to find part 3 = {z1,z2} from remaining verts
                    # such that all edges between part 2 and part 3 are 1
                    
                    # Remaining verts
                    rem = list(verts)
                    rem.remove(w1)
                    rem.remove(w2)
                    
                    # Iterate over pairs in rem
                    for k in range(len(rem) - 1):
                        z1 = rem[k]
                        for l in range(k + 1, len(rem)):
                            z2 = rem[l]
                            if M[z1, z2] == 1:
                                continue
                            
                            # Check edges between {w1,w2} and {z1,z2}
                            if (M[w1, z1] == 1 and M[w1, z2] == 1 and 
                                M[w2, z1] == 1 and M[w2, z2] == 1):
                                count += 1
    return count

def _is_valid_np(M, n):
    return _count_k222_np(M, n) == 0

def _simulated_annealing_np(colors, n, max_iters=50000, time_limit=100):
    """
    Simulated annealing using NumPy adjacency matrices.
    """
    start_time = time.time()
    
    # Build initial adjacency matrices
    M0 = np.zeros((n, n), dtype=np.int8)
    M1 = np.zeros((n, n), dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c == 0:
                M0[i, j] = 1
                M0[j, i] = 1
            else:
                M1[i, j] = 1
                M1[j, i] = 1
            idx += 1
            
    current_cost = _count_k222_np(M0, n) + _count_k222_np(M1, n)
    
    if current_cost == 0:
        return colors, True
    
    T0 = 10.0
    T_min = 0.01
    cooling = 0.999
    T = T0
    best_cost = current_cost
    best_colors = colors[:]
    no_improve = 0
    
    num_edges = n * (n - 1) // 2
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick random edge to flip
        idx = random.randint(0, num_edges - 1)
        
        # Find i, j from idx
        i = 0
        base = 0
        while True:
            next_base = (i + 1) * (2 * n - i - 2) // 2
            if idx < next_base:
                break
            i += 1
            base = next_base
        j = i + 1 + (idx - base)
        
        # Flip edge (i, j)
        old_val = M0[i, j]
        if old_val == 1:
            M0[i, j] = 0
            M0[j, i] = 0
            M1[i, j] = 1
            M1[j, i] = 1
            colors[idx] = 1
        else:
            M0[i, j] = 1
            M0[j, i] = 1
            M1[i, j] = 0
            M1[j, i] = 0
            colors[idx] = 0
            
        new_cost = _count_k222_np(M0, n) + _count_k222_np(M1, n)
        delta = new_cost - current_cost
        
        if delta <= 0:
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
                no_improve = 0
                T = max(T0 * 0.5, T_min)
            else:
                no_improve += 1
        else:
            if T > T_min and random.random() < np.exp(-delta / T):
                current_cost = new_cost
                no_improve += 1
            else:
                # Revert
                if old_val == 1:
                    M0[i, j] = 1
                    M0[j, i] = 1
                    M1[i, j] = 0
                    M1[j, i] = 0
                    colors[idx] = 0
                else:
                    M0[i, j] = 0
                    M0[j, i] = 0
                    M1[i, j] = 1
                    M1[j, i] = 1
                    colors[idx] = 1
        
        T *= cooling
        
        if current_cost == 0:
            return colors, True
            
    return best_colors, best_cost == 0

def construct():
    deadline = time.time() + 145
    target_n = 29
    p = 29  # 29 is prime
    
    # 1. Try Paley(29) directly
    colors = _paley(target_n, p)
    if colors:
        M0 = np.zeros((target_n, target_n), dtype=np.int8)
        M1 = np.zeros((target_n, target_n), dtype=np.int8)
        idx = 0
        for i in range(target_n):
            for j in range(i + 1, target_n):
                c = colors[idx]
                if c == 0:
                    M0[i, j] = 1
                    M0[j, i] = 1
                else:
                    M1[i, j] = 1
                    M1[j, i] = 1
                idx += 1
        if _is_valid_np(M0, target_n) and _is_valid_np(M1, target_n):
            return {'n': target_n, 'colors': colors}
    
    # 2. Simulated Annealing from Paley
    if time.time() < deadline - 30:
        colors_sa = colors[:]
        _, success = _simulated_annealing_np(colors_sa, target_n, max_iters=50000, time_limit=60)
        if success:
            return {'n': target_n, 'colors': colors_sa}
    
    # 3. Random restarts with Simulated Annealing
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        colors_rand = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
        _, success = _simulated_annealing_np(colors_rand, target_n, max_iters=20000, time_limit=20)
        if success:
            return {'n': target_n, 'colors': colors_rand}
            
    # 4. Fallback: try smaller n with Paley
    for n in range(28, 10, -1):
        if time.time() > deadline - 5:
            break
        for pp in [17, 19, 23, 29, 31]:
            if pp >= n:
                colors = _paley(n, pp)
                if colors:
                    M0 = np.zeros((n, n), dtype=np.int8)
                    M1 = np.zeros((n, n), dtype=np.int8)
                    idx = 0
                    for i in range(n):
                        for j in range(i + 1, n):
                            c = colors[idx]
                            if c == 0:
                                M0[i, j] = 1
                                M0[j, i] = 1
                            else:
                                M1[i, j] = 1
                                M1[j, i] = 1
                            idx += 1
                    if _is_valid_np(M0, n) and _is_valid_np(M1, n):
                        return {'n': n, 'colors': colors}
    
    # Absolute fallback
    return {'n': 1, 'colors': [0]}
