import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_addition(adj, u, v):
        if adj[u] & (1 << v):
            return False
        
        # Check triples containing u: {u, a, b}
        # If v is in N(a) and N(b), then v is added to N({u, a, b}).
        # We need old |N({u, a, b})| <= 1.
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << v)):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << v)):
                    continue
                
                count = (adj[u] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
                        
        # Check triples containing v: {v, a, b}
        # If u is in N(a) and N(b), then u is added to N({v, a, b}).
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << u)):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << u)):
                    continue
                
                count = (adj[v] & adj[a] & adj[b]).bit_count()
                if count > 1:
                    return False
                        
        return True

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_fill(adj, deadline):
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def find_violation(adj):
        """Find a K3,3 violation. Returns (u, v, a, b, c, d) such that {u,v,a} and {b,c,d} form K3,3 or similar.
        Actually, we just need to find any triple {x,y,z} with 3 common neighbors.
        """
        # For each triple (x,y,z), check common neighbors
        # This is O(N^3 * N) which is 20^4 = 160,000 operations. Acceptable.
        for x in range(n):
            for y in range(x + 1, n):
                for z in range(y + 1, n):
                    common = adj[x] & adj[y] & adj[z]
                    if common.bit_count() >= 3:
                        # Extract 3 neighbors
                        neighbors = []
                        mask = common
                        while mask and len(neighbors) < 3:
                            lsb = mask & (-mask)
                            idx = lsb.bit_length() - 1
                            neighbors.append(idx)
                            mask ^= lsb
                        return (x, y, z, neighbors[0], neighbors[1], neighbors[2])
        return None

    def repair_graph(adj, deadline):
        """Remove edges to eliminate K3,3 violations."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            violation = find_violation(adj)
            if violation:
                x, y, z, n1, n2, n3 = violation
                # Remove one random edge from the K3,3 structure
                # The K3,3 is between {x,y,z} and {n1,n2,n3}.
                # Edges are (x,n1), (x,n2), (x,n3), (y,n1), ...
                # Removing any one breaks the K3,3.
                edges_to_consider = [
                    (x, n1), (x, n2), (x, n3),
                    (y, n1), (y, n2), (y, n3),
                    (z, n1), (z, n2), (z, n3)
                ]
                u, v = random.choice(edges_to_consider)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                changed = True
        return adj

    def local_search(adj, deadline):
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        best_adj = current_adj[:]
        best_edges = current_edges
        
        # Greedy fill first
        current_adj = greedy_fill(current_adj, deadline)
        current_edges = count_edges(current_adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = current_adj[:]
            
        # Local search loop
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            # Try to add a random missing edge
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
            if not missing:
                break
            
            u, v = random.choice(missing)
            if is_valid_addition(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                no_improve = 0
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = current_adj[:]
            else:
                # Kick: remove 1-3 random edges
                kick_size = random.choice([1, 1, 2, 2, 3])
                existing = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
                if not existing:
                    break
                
                random.shuffle(existing)
                for _ in range(min(kick_size, len(existing))):
                    a, b = existing.pop()
                    current_adj[a] &= ~(1 << b)
                    current_adj[b] &= ~(1 << a)
                    current_edges -= 1
                    
                # Greedy fill
                current_adj = greedy_fill(current_adj, deadline)
                current_edges = count_edges(current_adj)
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = current_adj[:]
                    no_improve = 0
                else:
                    no_improve += 1
                    
        return best_adj, best_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Multiple restarts
    num_restarts = 1000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with random dense graph
        adj = [0] * n
        for u, v in all_pairs:
            if random.random() < 0.5:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        # Repair
        adj = repair_graph(adj, deadline)
        
        # Local search
        adj, edges = local_search(adj, deadline)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # Final polish on best solution
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            adj, edges = local_search(best_adj, deadline)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj[:]
            else:
                break

    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
