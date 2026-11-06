"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Returns a 4-colouring of K_n with no monochromatic triangle, maximising n.
    Base: Greenwood-Gleason K16 (Clebsch graph x3) doubled to K32.
    """
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # --- Build base K16 Greenwood-Gleason colouring ---
    # GF(16) with polynomial x^4 + x + 1 (0b10011)
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

    # Base K16: colour of edge (i,j) = log[i^j] % 3 for i!=j, i,j in 0..15
    base16 = {}
    for i in range(16):
        for j in range(i + 1, 16):
            base16[(i, j)] = log[i ^ j] % 3

    # --- Build K32: two copies of K16, cross edges all colour 3 ---
    def get_color_32(i, j):
        if i // 16 == j // 16:
            return base16[(i % 16, j % 16)]
        else:
            return 3

    # Precompute adjacency for K32
    n32 = 32
    adj = [[0] * n32 for _ in range(n32)]
    for i in range(n32):
        for j in range(i + 1, n32):
            c = get_color_32(i, j)
            adj[i][j] = c
            adj[j][i] = c

    # --- Check if a new vertex can be added ---
    def can_add_vertex(adj, n, color_map):
        """
        Try to find a colour assignment for a new vertex (n) to vertices 0..n-1.
        color_map[i] = colour of edge (n, i) for i in 0..n-1.
        Returns True if valid.
        """
        groups = {0: [], 1: [], 2: [], 3: []}
        for i in range(n):
            groups[color_map[i]].append(i)

        for c in range(4):
            grp = groups[c]
            for a_idx in range(len(grp)):
                for b_idx in range(a_idx + 1, len(grp)):
                    a, b = grp[a_idx], grp[b_idx]
                    if adj[a][b] == c:
                        return False
        return True

    def try_extend(adj, n, max_new=1, budget_s=10.0):
        """
        Try to add up to max_new vertices one at a time.
        Uses randomized local search / hill climbing on the colour assignment.
        """
        current_adj = [row[:] for row in adj]
        current_n = n
        t_start = time.time()

        for _ in range(max_new):
            if time.time() - t_start > budget_s:
                break
            found = False
            # Start with a random valid-ish assignment and try to fix conflicts
            # We use a simple approach: random restarts with a few local moves
            for attempt in range(50000):
                if time.time() > deadline:
                    return None
                # Random assignment
                color_map = [random.randint(0, 3) for _ in range(current_n)]
                
                # Local search: try to fix conflicts by flipping single vertices
                # A conflict is a pair (a,b) with color_map[a] == color_map[b] == c and adj[a][b] == c
                # We'll just check validity and if not, do a few random flips
                if can_add_vertex(current_adj, current_n, color_map):
                    # Add the vertex
                    new_n = current_n + 1
                    new_row = color_map[:] + [0]
                    for i in range(current_n):
                        current_adj[i].append(color_map[i])
                    current_adj.append(new_row)
                    current_n = new_n
                    found = True
                    break
                
                # If not valid, try a few local improvements (random flips)
                # This is a very crude local search
                for _flip in range(10):
                    if time.time() > deadline:
                        return None
                    idx = random.randint(0, current_n - 1)
                    old_c = color_map[idx]
                    new_c = random.randint(0, 3)
                    color_map[idx] = new_c
                    if can_add_vertex(current_adj, current_n, color_map):
                        break
                    color_map[idx] = old_c

            if not found:
                break

        if current_n > n:
            return current_n, current_adj
        return None

    # --- Attempt to extend K32 to K33 ---
    extended = None
    if time.time() < deadline - 5.0:
        extended = try_extend(adj, 32, max_new=1, budget_s=120.0)

    # --- Build final result ---
    if extended is not None:
        n_final, adj_final = extended
        colors = []
        for i in range(n_final):
            for j in range(i + 1, n_final):
                colors.append(adj_final[i][j])
        return {'n': n_final, 'colors': colors}
    else:
        # Return K32
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(adj[i][j])
        return {'n': 32, 'colors': colors}

# EVOLVE-BLOCK-END
