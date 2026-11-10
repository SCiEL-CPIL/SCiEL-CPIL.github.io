import random
import time
import sys

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_edges = 0
    best_graph = None
    
    def popcount(x):
        return bin(x).count("1")
        
    def is_valid(N):
        # Check all triples
        for u in range(n):
            for v in range(u + 1, n):
                cn_uv = N[u] & N[v]
                for w in range(v + 1, n):
                    if popcount(cn_uv & N[w]) > 2:
                        return False
        return True

    def repair(N):
        # Remove edges to make valid
        # Find violations and remove edges
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Find a violating triple
            for u in range(n):
                for v in range(u + 1, n):
                    cn_uv = N[u] & N[v]
                    for w in range(v + 1, n):
                        common = cn_uv & N[w]
                        if popcount(common) > 2:
                            # Violation found
                            # Remove a random edge incident to u, v, or w 
                            # that connects to a common neighbor
                            # Let's pick a random common neighbor x
                            # and remove edge (u, x)
                            # This reduces common neighbors of {u, v, w} by 1
                            # Note: This might create other issues but we loop
                            
                            # Get list of common neighbors
                            # common is a bitmask
                            # Pick a random bit
                            x = -1
                            mask = common
                            while mask:
                                # Random selection from bits
                                # Simple: just take lowest set bit for determinism in repair, 
                                # or random for diversity
                                # Let's use random
                                # To pick random bit:
                                # Generate random index
                                idx = random.randint(0, popcount(mask) - 1)
                                # Find idx-th set bit
                                temp = mask
                                for _ in range(idx):
                                    temp &= (temp - 1)
                                # temp is now mask with lowest set bit at idx
                                # The bit value is temp & -temp
                                x_bit = temp & -temp
                                x = x_bit.bit_length() - 1
                                break
                            
                            # Remove edge (u, x)
                            # Check if u is connected to x
                            if N[u] & (1 << x):
                                N[u] &= ~(1 << x)
                                N[x] &= ~(1 << u)
                                changed = True
                                break
                    if changed:
                        break
                if changed:
                    break
        return N

    def greedy_add(N):
        # Try to add all missing edges
        added = True
        while added and time.time() < deadline:
            added = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (N[u] & (1 << v)):
                        # Try adding
                        # Check if adding (u, v) creates violation
                        # New N[u] has v, New N[v] has u
                        # Only triples involving u or v are affected
                        # Check triples with u
                        valid = True
                        # Triples {u, x, y}
                        for x in range(n):
                            if x == u: continue
                            for y in range(x + 1, n):
                                if y == u: continue
                                # Common neighbors of u, x, y
                                # N[u] includes v now?
                                # If we add (u, v), v becomes a neighbor of u.
                                # So v is in N'[u].
                                # Is v in N[x] and N[y]?
                                # If yes, v is a new common neighbor.
                                # Also, u becomes neighbor of v.
                                # This affects triples {v, x, y}.
                                
                                # Let's just simulate and check the specific triples
                                # Triples containing u:
                                # {u, x, y} where x,y != u
                                # Common neighbors: (N[u] | {v}) & N[x] & N[y]
                                # This is (N[u] & N[x] & N[y]) | ({v} & N[x] & N[y])
                                # Let C_old = N[u] & N[x] & N[y]
                                # New C = C_old | (1 << v) if (N[x] & (1<<v)) and (N[y] & (1<<v))
                                # If popcount(C_new) > 2, invalid.
                                
                                # It's faster to just check all triples involving u or v
                                pass
                        
                        # Simpler check:
                        # Temporarily add edge
                        N[u] |= (1 << v)
                        N[v] |= (1 << u)
                        
                        # Check validity of triples involving u or v
                        ok = True
                        # Triples with u
                        for x in range(n):
                            if x == u: continue
                            for y in range(x + 1, n):
                                if y == u: continue
                                if popcount(N[u] & N[x] & N[y]) > 2:
                                    ok = False
                                    break
                            if not ok: break
                        # Triples with v (but not u, already checked if x,y != u)
                        # Wait, triples {u, x, y} cover cases where u is in triple.
                        # Triples {v, x, y} where u is NOT in triple.
                        if ok:
                            for x in range(n):
                                if x == v or x == u: continue
                                for y in range(x + 1, n):
                                    if y == v or y == u: continue
                                    if popcount(N[v] & N[x] & N[y]) > 2:
                                        ok = False
                                        break
                                if not ok: break
                        
                        if ok:
                            added = True
                            # Keep edge
                        else:
                            # Revert
                            N[u] &= ~(1 << v)
                            N[v] &= ~(1 << u)
            # End while added
        return N

    # Main Loop
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Random initialization
        N = [0] * n
        p = 0.6 + random.random() * 0.2
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    N[u] |= (1 << v)
                    N[v] |= (1 << u)
        
        # Repair
        N = repair(N)
        
        # Greedy Add
        N = greedy_add(N)
        
        # Count edges
        edge_count = sum(popcount(N[u]) for u in range(n)) // 2
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_graph = N[:]
            
        # If we have plenty of time, do a local search polish on the best graph?
        # Or just keep restarting.
        
        # Optimization: If we find a very dense graph, try to improve it further
        if best_edges > 100:
             # Simple local search on best_graph
             # Try swapping edges?
             pass

    # Return list of edges
    edges = []
    if best_graph:
        for u in range(n):
            for v in range(u + 1, n):
                if best_graph[u] & (1 << v):
                    edges.append([u, v])
    else:
        # Fallback to seed
        edges = [[a, b] for a in range(2) for b in range(2, 20)]
        
    return edges

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(len(res))
