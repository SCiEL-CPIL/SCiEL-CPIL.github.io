import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    n = 40
    best_edges = []
    best_count = 0
    
    # Helper to check if adding (u, v) creates a C6
    # adj: list of int bitmasks
    # d2: list of int bitmasks (neighbors at distance 2)
    def creates_c6(u, v, adj, d2):
        if adj[u] >> v & 1:
            return False # Already an edge, no change
        if adj[v] >> u & 1:
            return False
            
        # Check intersection of distance-2 neighborhoods
        common = d2[u] & d2[v]
        if common == 0:
            return False
        
        # Iterate through potential middle vertices y
        # y is in d2[u] and d2[v]
        # We need x in N(u) and N(y), and z in N(v) and N(y)
        # such that u, x, y, z, v are distinct.
        
        # Get list of y's
        y_list = []
        temp = common
        while temp:
            lsb = temp & -temp
            y = lsb.bit_length() - 1
            y_list.append(y)
            temp -= lsb
        
        for y in y_list:
            # Find x in N(u) \cap N(y)
            # Find z in N(v) \cap N(y)
            # We need x != z, x != v, z != u, x != y, z != y (y is dist 2 so x!=y, z!=y usually holds unless triangle, but C6 is simple)
            # Actually, if x=z, path is u-x-y-x-v, not a simple path.
            # If x=v, path is u-v-y-v-v? No, x is neighbor of u. If x=v, then u-v is edge. But we are adding u-v.
            # Wait, if u-v is not an edge, x cannot be v because v is not in N(u).
            # So x != v and z != u automatically.
            # We just need x != z.
            
            nx = adj[u] & adj[y]
            nz = adj[v] & adj[y]
            
            # Check if there exists x in nx and z in nz such that x != z
            # If nx & nz == 0, then any x in nx and z in nz are distinct.
            if (nx & nz) == 0:
                if nx != 0 and nz != 0:
                    return True
            
            # If nx & nz != 0, there might still be distinct x, z
            # e.g., nx = {1, 2}, nz = {2, 3}. x=1, z=3 works.
            # Only fails if nx is a subset of {z} for all z?
            # Specifically, we need x in nx, z in nz, x != z.
            # This fails only if nx == nz == {k} for some k?
            # Or if nx = {1}, nz = {1}.
            # If nx = {1, 2} and nz = {1, 2}, we can pick x=1, z=2.
            # So it fails only if nx == nz and len(nx) == 1.
            # Or more generally, if nx is a subset of nz and nz is a subset of nx and len < 2?
            # Actually, if there is any x in nx not in nz, we can pick that x and any z in nz.
            # If there is any z in nz not in nx, we can pick any x in nx and that z.
            # It fails only if nx == nz and len(nx) == 1.
            
            if nx == nz and (nx & (nx - 1)) == 0: # nx is a power of 2 (single bit)
                continue
            else:
                return True
                
        return False

    def update_d2(adj, d2, u, v):
        # Add edge u-v
        # d2[u] changes: neighbors of v are now at dist 2 from u (if not dist 1)
        # d2[v] changes: neighbors of u are now at dist 2 from v
        # d2[w] for w in N(u) \cap N(v): u and v are now at dist 2 from w?
        # No, u and v are neighbors of w. So u, v are dist 1 from w.
        # For w in N(u) \ N(v): v is now dist 2 from w.
        # For w in N(v) \ N(u): u is now dist 2 from w.
        # For w not in N(u) U N(v):
        #   If w is neighbor of some x in N(u), and x is neighbor of u...
        #   This update is complex.
        # Given N=40, recalculating d2 from scratch is O(N^3) or O(N^2 * bitops) which is fast.
        pass

    def recalc_d2(adj):
        d2 = [0] * n
        for i in range(n):
            mask = 0
            # d2[i] is union of N(x) for x in N(i), excluding i
            ni = adj[i]
            temp = ni
            while temp:
                lsb = temp & -temp
                x = lsb.bit_length() - 1
                mask |= adj[x]
                temp -= lsb
            mask &= ~ni # Remove direct neighbors
            mask &= ~(1 << i) # Remove self
            d2[i] = mask
        return d2

    def count_edges(adj):
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2

    def random_init(edges_target):
        adj = [0] * n
        possible_edges = []
        for i in range(n):
            for j in range(i+1, n):
                possible_edges.append((i, j))
        random.shuffle(possible_edges)
        for i in range(min(edges_target, len(possible_edges))):
            u, v = possible_edges[i]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def prune_c6s(adj):
        # Remove edges that are part of a C6
        # We can check each edge if removing it helps? No, just check if adding it created a C6?
        # Better: Check if the graph has a C6. If yes, remove a random edge from it.
        # Finding a C6:
        # Check all pairs (u, v) not connected? No, C6 uses existing edges.
        # We can use the creates_c6 logic in reverse?
        # Simpler: Iterate all pairs (u, v) that are edges.
        # Temporarily remove (u, v). Check if creates_c6(u, v) would have been false?
        # No, creates_c6 checks if adding creates a C6.
        # If (u, v) is an edge, and we remove it, does it break a C6?
        # A C6 containing (u, v) exists iff there is a path of length 5 between u and v in G - (u, v).
        # This is equivalent to: in G, is there a path of length 4 between u and v?
        # Wait. If there is a path of length 4 between u and v in G, then adding (u, v) creates a C6?
        # No, the path in G might use (u, v)? No, path of length 4 doesn't use (u, v) if it's simple.
        # If there is a simple path of length 4 between u and v in G, then (u, v) plus that path forms a C6?
        # Only if the path doesn't use (u, v). Since (u, v) is an edge, the path of length 4 is distinct.
        # So, (u, v) is in a C6 iff there is a path of length 4 between u and v in G.
        
        changed = True
        while changed:
            changed = False
            d2 = recalc_d2(adj)
            for u in range(n):
                if changed: break
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        # Check if there is a path of length 4
                        # This is the same check as creates_c6, but we know the edge exists.
                        # creates_c6(u, v, adj, d2) checks if adding (u, v) creates a C6.
                        # If (u, v) is already an edge, creates_c6 returns False immediately in my draft?
                        # Let's modify creates_c6 to work for existing edges too?
                        # Actually, if (u, v) is an edge, and there is a path of length 4, then we have a C6.
                        # So we can just use the logic inside creates_c6.
                        
                        # Re-use logic:
                        common = d2[u] & d2[v]
                        if common == 0:
                            continue
                            
                        y_list = []
                        temp = common
                        while temp:
                            lsb = temp & -temp
                            y = lsb.bit_length() - 1
                            y_list.append(y)
                            temp -= lsb
                            
                        found = False
                        for y in y_list:
                            nx = adj[u] & adj[y]
                            nz = adj[v] & adj[y]
                            # Need x in nx, z in nz, x != z
                            # Also x != v, z != u (guaranteed if simple path logic holds? No, x is neighbor of u. If x=v, then u-v is edge. But we are checking path of length 4. If x=v, path is u-v-y-z-v? No, z is neighbor of v.
                            # If x=v, then v is in N(u).
                            # If z=u, then u is in N(v).
                            # The path is u-x-y-z-v.
                            # If x=v, path is u-v-y-z-v. Not simple.
                            # If z=u, path is u-x-y-u-v. Not simple.
                            # If x=z, path is u-x-y-x-v. Not simple.
                            # So we just need x != z, and x != v, z != u?
                            # If x=v, then v in N(u).
                            # If we remove (u, v), x is no longer v?
                            # This is getting messy.
                            
                            # Simpler: Just check if creates_c6 would be True if we removed the edge and added it back?
                            # No.
                            
                            # Let's just assume if common != 0, it's likely a C6.
                            # For pruning, we can be conservative.
                            if nx and nz:
                                # Check distinctness
                                # x in nx, z in nz
                                # If nx & nz == 0, distinctness of x, z is guaranteed.
                                if (nx & nz) == 0:
                                    found = True
                                    break
                                else:
                                    # Check if there is a pair x, z distinct
                                    if not (nx == nz and (nx & (nx - 1)) == 0):
                                        found = True
                                        break
                                        
                        if found:
                            # Remove edge u-v
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            changed = True
                            break
        return adj

    def try_add_edges(adj):
        improved = True
        while improved:
            improved = False
            d2 = recalc_d2(adj)
            for u in range(n):
                if improved: break
                for v in range(u+1, n):
                    if not ((adj[u] >> v) & 1):
                        if not creates_c6(u, v, adj, d2):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True
                            break
        return adj

    # Main loop
    iterations = 0
    while time.time() - start_time < time_limit:
        iterations += 1
        # Random restart with varying density
        # Target edges: 90 to 110
        target = random.randint(85, 110)
        adj = random_init(target)
        
        # Prune C6s
        adj = prune_c6s(adj)
        
        # Add edges
        adj = try_add_edges(adj)
        
        # Local search: Random swap?
        # Try to remove a random edge and see if we can add 2?
        # Or just try to add more.
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            # Convert adj to list of edges
            best_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (adj[i] >> j) & 1:
                        best_edges.append([i, j])
                        
        # If we hit 105, we can break or continue to see if we can go higher
        if best_count >= 105:
            # Try to push for more
            pass

    return best_edges

# EVOLVE-BLOCK-END
