import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140  # 10s margin
    
    def is_valid(colors, n):
        # Check if no monochromatic triangle
        # colors is list of length n*(n-1)//2
        # Index for (i, j) with i < j: i*(n-1) - i*(i+1)//2 + j - i - 1 ? 
        # Simpler: use a 2D array
        c = np.zeros((n, n), dtype=int)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c[i, j] = c[j, i] = colors[idx]
                idx += 1
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if c[i, j] == c[j, k] == c[i, k]:
                        return False
        return True

    def get_seed_32():
        # Reuse the seed logic
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

    seed_result = get_seed_32()
    
    # Try to find n=50
    n_target = 50
    best_n = 32
    best_colors = seed_result['colors']
    
    for restart in range(100): # Max 100 restarts
        if time.time() > deadline:
            break
            
        # Random init
        colors = [random.randint(0, 3) for _ in range(n_target * (n_target - 1) // 2)]
        
        # Convert to 2D array for fast access
        c = np.zeros((n_target, n_target), dtype=np.int8)
        idx = 0
        for i in range(n_target):
            for j in range(i+1, n_target):
                c[i, j] = c[j, i] = colors[idx]
                idx += 1
        
        # Compute N(u, v, k)
        # N[u, v, k] = count of z such that c[u,z]=k and c[v,z]=k
        N = np.zeros((n_target, n_target, 4), dtype=np.int32)
        
        # Initialize N
        for u in range(n_target):
            for v in range(u+1, n_target):
                for k in range(4):
                    # Count z != u, v such that c[u,z]=k and c[v,z]=k
                    # Vectorized:
                    mask = (c[u] == k) & (c[v] == k)
                    mask[u] = False
                    mask[v] = False
                    N[u, v, k] = np.sum(mask)
                    N[v, u, k] = N[u, v, k]
        
        # Compute M
        M = 0
        for u in range(n_target):
            for v in range(u+1, n_target):
                M += N[u, v, c[u, v]]
        M //= 3
        
        if M == 0:
            return {'n': n_target, 'colors': colors}
        
        # Local Search
        # We will do a number of steps
        steps_per_restart = 50000
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            if M == 0:
                return {'n': n_target, 'colors': colors}
                
            # Pick random edge
            u = random.randint(0, n_target - 1)
            v = random.randint(0, n_target - 1)
            while v == u:
                v = random.randint(0, n_target - 1)
            
            c_old = c[u, v]
            
            # Find best new color
            best_k = -1
            best_val = N[u, v, c_old] + 1 # +1 to force change if all are same?
            # Actually, we want to minimize N[u, v, k]
            # If min N is equal to current, we might still move to escape local min
            min_val = min(N[u, v, :])
            
            candidates = [k for k in range(4) if N[u, v, k] == min_val]
            
            if min_val < N[u, v, c_old]:
                # Always take improvement
                c_new = random.choice(candidates)
            elif min_val == N[u, v, c_old]:
                # Random walk
                if random.random() < 0.5:
                    c_new = random.choice(candidates)
                else:
                    continue
            else:
                # No improvement, skip (or accept with very low prob)
                continue
            
            if c_new == c_old:
                continue
                
            # Update M
            M += (N[u, v, c_new] - N[u, v, c_old]) // 3
            # Note: (A-B) is divisible by 3? 
            # M = 1/3 sum N[u,v,c_uv].
            # Change is 1/3 (N_new - N_old).
            # Is N_new - N_old divisible by 3? No.
            # But M must be integer.
            # So we should store M as integer and update carefully.
            # Actually, let's just recompute M? No, too slow.
            # Let's store M as float? No.
            # The change in M is (N[u,v,c_new] - N[u,v,c_old]) / 3.
            # Since M is integer, the difference must be multiple of 3?
            # No. M = 1/3 * Sum.
            # If Sum changes by D, M changes by D/3.
            # So D must be divisible by 3?
            # Not necessarily. But M is always integer.
            # So D must be divisible by 3.
            # Let's verify.
            # N[u,v,k] = count of z.
            # Change in N[u,v,c_old] and N[u,v,c_new].
            # The sum over all edges changes by D.
            # Since M is integer, D must be divisible by 3.
            
            # Update N
            # For each w != u, v:
            # c_vw = c[v, w]
            # c_uw = c[u, w]
            # N[u, w, c_vw] changes by:
            #   if c_old == c_vw: -1
            #   if c_new == c_vw: +1
            # N[v, w, c_uw] changes by:
            #   if c_old == c_uw: -1
            #   if c_new == c_uw: +1
            
            for w in range(n_target):
                if w == u or w == v:
                    continue
                c_vw = c[v, w]
                c_uw = c[u, w]
                
                if c_old == c_vw:
                    N[u, w, c_vw] -= 1
                    N[w, u, c_vw] -= 1
                if c_new == c_vw:
                    N[u, w, c_vw] += 1
                    N[w, u, c_vw] += 1
                    
                if c_old == c_uw:
                    N[v, w, c_uw] -= 1
                    N[w, v, c_uw] -= 1
                if c_new == c_uw:
                    N[v, w, c_uw] += 1
                    N[w, v, c_uw] += 1
                    
            # Update c
            c[u, v] = c_new
            c[v, u] = c_new
            
        # End local search for this restart
        
    return {'n': best_n, 'colors': best_colors}
