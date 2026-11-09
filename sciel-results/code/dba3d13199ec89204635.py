"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import math

def construct():
    deadline = time.time() + 118.0
    rng = random.Random(12345)

    # ---------- GF(16) arithmetic (Greenwood-Gleason base colouring) ----------
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    def base_color(i, j):
        if i // 16 == j // 16:
            return log[(i % 16) ^ (j % 16)] % 3
        return 3

    # ---------- initial 32-vertex colouring (two copies + bipartite colour 3) ----------
    n0 = 32
    mat0 = [[-1] * n0 for _ in range(n0)]
    for i in range(n0):
        for j in range(i + 1, n0):
            c = base_color(i, j)
            mat0[i][j] = c
            mat0[j][i] = c

    def matrix_to_colors(mat, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(mat[i][j])
        return colors

    def verify(mat, n):
        """Check no monochromatic triangle."""
        for i in range(n):
            for j in range(i + 1, n):
                c = mat[i][j]
                if c < 0:
                    return False
                for k in range(j + 1, n):
                    if mat[i][k] == c and mat[j][k] == c:
                        return False
        return True

    # Count monochromatic triangles efficiently using bitsets
    def count_triangles(mat, n):
        # adj[c][u] is bitmask of vertices connected to u by color c
        adj = [0] * (n * 4)
        for u in range(n):
            for w in range(u + 1, n):
                c = mat[u][w]
                adj[c * n + u] |= (1 << w)
                adj[c * n + w] |= (1 << u)
        
        count = 0
        for c in range(4):
            for u in range(n):
                neighbors = adj[c * n + u]
                # For each pair (v, w) in neighbors, check if (v, w) is color c
                # This is O(n^3) if done naively. 
                # Better: For each u, for each v in neighbors, check intersection of neighbors[u] and neighbors[v]
                # Actually, simpler: 
                # For each color c, we have a graph G_c. We want to count triangles in G_c.
                # Triangles in G_c = sum over u < v < w of [u~v, v~w, w~u in G_c]
                # Using bitsets:
                # For each u, for each v > u such that u~v in G_c:
                #   count += popcount( adj[c*n+u] & adj[c*n+v] & (bitmask of vertices > v) )
                # This is efficient.
                pass # We will implement a faster version in the SA loop if needed, 
                     # but for now, let's use a simple check for validity (0 vs >0) 
                     # and a faster count for SA energy.
        
        # Fast energy calculation for SA
        energy = 0
        for c in range(4):
            # Build adjacency lists for color c
            # We can just iterate triangles
            # For small n (<60), O(n^3) is ~ 200k ops, which is fast enough for a few thousand steps?
            # 51^3 = 132k. 1000 steps = 132M ops. Might be too slow in Python.
            # Let's use the bitset triangle counting.
            for u in range(n):
                neigh_u = adj[c * n + u]
                # Only consider v > u to avoid double counting
                # We need to check triples (u, v, w) with u < v < w
                # Iterate v
                v_mask = neigh_u
                while v_mask:
                    lsb = v_mask & -v_mask
                    v = lsb.bit_length() - 1
                    if v > u:
                        # Check w > v
                        # w must be in neigh_u AND neigh_v AND > v
                        # neigh_u & neigh_v gives common neighbors
                        common = adj[c * n + u] & adj[c * n + v]
                        # Mask out vertices <= v
                        # We want bits > v
                        # (1 << (v+1)) - 1 is mask for 0..v. So ~that is v+1..
                        # Python ints are infinite precision, so we need to be careful or just use modulo/subtraction logic
                        # Simpler: 
                        # common has bits for all common neighbors.
                        # We want count of bits in common that are > v.
                        # common >> (v+1) removes bits 0..v.
                        energy += (common >> (v + 1)).bit_count()
                    v_mask ^= lsb
        return energy

    best_n = n0
    best_mat = [row[:] for row in mat0]

    # ---------- Simulated Annealing to fix/add a vertex ----------
    def sa_add_vertex(mat, n, time_limit, rng):
        """
        Attempts to add vertex n by running SA on a graph of size n+1.
        Returns (new_mat, new_n) if success, else (mat, n).
        """
        new_n = n + 1
        # Initialize new vertex with random colors
        new_mat = [row[:] for row in mat]
        new_mat.append([-1] * new_n)
        for i in range(n):
            c = rng.randint(0, 3)
            new_mat[i].append(c)
            new_mat[new_n - 1][i] = c

        # Initial energy
        current_energy = count_triangles(new_mat, new_n)
        
        if current_energy == 0:
            return new_mat, new_n

        T = 10.0
        T_min = 0.01
        cooling_rate = 0.995
        max_steps = 50000 # Will be limited by time
        
        best_energy = current_energy
        best_mat_copy = [row[:] for row in new_mat]
        
        start_time = time.time()
        
        while T > T_min:
            if time.time() > time_limit:
                break
            
            # Pick a random edge
            u = rng.randint(0, new_n - 1)
            v = rng.randint(0, new_n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            old_color = new_mat[u][v]
            new_color = rng.randint(0, 3)
            if old_color == new_color:
                continue
            
            # Calculate delta energy
            # Changing edge (u,v) from c1 to c2 affects triangles involving (u,v) and a third vertex w.
            # Triangle (u,v,w) is monochromatic c1 if mat[u][w]==c1, mat[v][w]==c1.
            # It becomes monochromatic c2 if mat[u][w]==c2, mat[v][w]==c2.
            # Delta = (# tri c2 formed) - (# tri c1 removed)
            
            # Count triangles removed (color old_color)
            # w must be connected to u and v with old_color
            # We can precompute or just loop w
            # For speed, let's loop w
            removed = 0
            formed = 0
            for w in range(new_n):
                if w == u or w == v:
                    continue
                # Check if (u,v,w) was mono old_color
                if new_mat[u][w] == old_color and new_mat[v][w] == old_color:
                    removed += 1
                # Check if (u,v,w) becomes mono new_color
                if new_mat[u][w] == new_color and new_mat[v][w] == new_color:
                    formed += 1
            
            delta = formed - removed
            
            # Apply change
            new_mat[u][v] = new_color
            new_mat[v][u] = new_color
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_mat_copy = [row[:] for row in new_mat]
                if best_energy == 0:
                    return best_mat_copy, new_n
            
            # Metropolis acceptance
            if delta > 0:
                if rng.random() < math.exp(-delta / T):
                    pass # Accept
                else:
                    # Reject: revert
                    new_mat[u][v] = old_color
                    new_mat[v][u] = old_color
                    current_energy -= delta
            
            T *= cooling_rate
            
        # After SA, check if we found a solution
        if best_energy == 0:
            return best_mat_copy, new_n
        
        # If we didn't find 0, we might have improved the energy of the n+1 graph.
        # But we need a VALID graph of size n+1. 
        # If we can't make it valid, we discard this attempt.
        # However, if the energy is very low, maybe we can use it as a seed for a restart?
        # For now, if not 0, return failure.
        return mat, n

    # ---------- Main loop ----------
    current_mat = [row[:] for row in mat0]
    current_n = n0

    max_target = 51
    consecutive_fails = 0

    while current_n < max_target and time.time() < deadline - 10:
        time_limit = min(deadline - 10, time.time() + 5.0)
        # Try SA to add a vertex
        new_mat, new_n = sa_add_vertex(current_mat, current_n, time_limit, rng)

        if new_n > current_n:
            current_mat = new_mat
            current_n = new_n
            consecutive_fails = 0
            if current_n > best_n:
                best_n = current_n
                best_mat = [row[:] for row in current_mat]
        else:
            consecutive_fails += 1
            if consecutive_fails >= 3:
                # Restart with different random seed for SA
                rng = random.Random(rng.randint(0, 2**31))
                consecutive_fails = 0

    # Final verification
    if verify(best_mat, best_n):
        best_colors = matrix_to_colors(best_mat, best_n)
        return {'n': best_n, 'colors': best_colors}
    else:
        # Fallback to the known 32-vertex solution if something went wrong
        best_colors = matrix_to_colors(mat0, n0)
        return {'n': n0, 'colors': best_colors}

# EVOLVE-BLOCK-END
