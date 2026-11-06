import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    def is_valid_add(u, v):
        # Check if u and v are already connected
        if (adj[u] >> v) & 1:
            return False
        
        # Check 1: |N(u) \cap N(v)| <= 2
        common = adj[u] & adj[v]
        if common.bit_count() > 2:
            return False
        
        # Check 2: For all w in N(u), adding v to N(u) and u to N(v)
        # must not create a K2,3.
        # Specifically, for any w in N(u), the new common neighbors of w and v
        # are (N(w) \cap N(v)) U {u}.
        # So we need |(N(w) \cap N(v)) U {u}| <= 2.
        # Since u is not in N(v) yet (we are adding it), and u is in N(w) (w in N(u) implies u in N(w) for simple graph? No, N(w) contains u if edge (u,w) exists. Yes.)
        # So u is in N(w). Is u in N(v)? No, not yet.
        # So the set is (N(w) \cap N(v)) U {u}.
        # Size is |N(w) \cap N(v)| + 1 (since u is not in N(v) and u is not in N(w) \cap N(v) because u not in N(v)).
        # Wait, is u in N(w)? Yes, because w in N(u) => edge (u,w) => u in N(w).
        # Is u in N(v)? No.
        # So u is disjoint from N(w) \cap N(v).
        # So size is |N(w) \cap N(v)| + 1.
        # We need |N(w) \cap N(v)| + 1 <= 2 => |N(w) \cap N(v)| <= 1.
        
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            # Check |N(w) \cap N(v)| <= 1
            if (adj[w] & adj[v]).bit_count() > 1:
                return False
            temp ^= lsb
        
        # Check 3: Symmetric check for z in N(v)
        # For any z in N(v), new common neighbors of z and u are (N(z) \cap N(u)) U {v}.
        # Size is |N(z) \cap N(u)| + 1.
        # We need |N(z) \cap N(u)| <= 1.
        
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            z = lsb.bit_length() - 1
            if (adj[z] & adj[u]).bit_count() > 1:
                return False
            temp ^= lsb
            
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges():
        return sum(adj[i].bit_count() for i in range(n)) // 2

    best_count = 0
    best_adj = [0] * n

    def greedy_init():
        nonlocal adj
        adj = [0] * n
        # Deterministic greedy with random tie-breaking
        edges = all_edges[:]
        random.shuffle(edges)
        for u, v in edges:
            if is_valid_add(u, v):
                add_edge(u, v)

    def polish():
        nonlocal adj
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(u, v):
                        add_edge(u, v)
                        improved = True

    def run_sa():
        nonlocal adj, best_count, best_adj
        
        greedy_init()
        polish()
        
        cur_count = count_edges()
        if cur_count > best_count:
            best_count = cur_count
            best_adj = adj[:]
            
        if best_count >= 48:
            return

        T = 0.5
        alpha = 0.9999
        min_T = 0.01
        
        while time.time() < deadline - 5:
            # Identify present and missing edges
            present = []
            missing = []
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    present.append((u, v))
                else:
                    missing.append((u, v))
                    
            if not present or not missing:
                break
                
            move_type = random.random()
            
            if move_type < 0.7 and missing:
                # Add an edge
                u, v = random.choice(missing)
                if is_valid_add(u, v):
                    add_edge(u, v)
                    cur_count += 1
                    if cur_count > best_count:
                        best_count = cur_count
                        best_adj = adj[:]
                        if best_count >= 48:
                            return
            elif move_type < 0.9 and present:
                # Remove an edge
                u, v = random.choice(present)
                # Always accept removal? Or with probability?
                # Removal decreases objective, so accept with prob exp(-delta/T)
                # delta = -1 (loss of 1 edge)
                # prob = exp(1/T) which is > 1. So always accept?
                # In SA for maximization, we accept worse moves with prob exp((new-old)/T).
                # new = cur - 1, old = cur. diff = -1.
                # prob = exp(-1/T).
                if random.random() < math.exp(-1.0 / max(T, min_T)):
                    remove_edge(u, v)
                    cur_count -= 1
            else:
                # Swap: remove one, try to add another
                if present and missing:
                    u1, v1 = random.choice(present)
                    u2, v2 = random.choice(missing)
                    remove_edge(u1, v1)
                    if is_valid_add(u2, v2):
                        add_edge(u2, v2)
                        # Count stays same
                        # Check if this swap enables future improvements?
                        # Just keep it if valid.
                    else:
                        add_edge(u1, v1)
                        
            T = max(T * alpha, min_T)
            
        # Final polish
        polish()
        cur_count = count_edges()
        if cur_count > best_count:
            best_count = cur_count
            best_adj = adj[:]

    # Run multiple restarts
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        run_sa()
        if best_count >= 48:
            break
            
    res = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                res.append([i, j])
    return res

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    adj = [0] * 20
    for u, v in e:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    valid = True
    for i in range(20):
        for j in range(i + 1, 20):
            if (adj[i] & adj[j]).bit_count() > 2:
                valid = False
                print(f"Violation: {i}, {j} have {(adj[i] & adj[j]).bit_count()} common neighbors")
    print("Valid" if valid else "Invalid")
