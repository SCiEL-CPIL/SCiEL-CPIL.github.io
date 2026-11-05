import random
import time
import math

def construct():
    n = 20
    edges = set()
    
    def get_edge(u, v):
        if u > v: u, v = v, u
        return (u, v)

    def can_add(u, v):
        u, v = get_edge(u, v)
        if (u, v) in edges:
            return False
            
        # Check if adding (u, v) creates a K2,3
        # This happens if there exists w such that |N(u) \cap N(w)| becomes 3 or |N(v) \cap N(w)| becomes 3
        # Adding (u, v) adds v to N(u) and u to N(v).
        
        for w in range(n):
            if w == u or w == v:
                continue
            
            # Case 1: v is a neighbor of w. Then v becomes a common neighbor of u and w.
            if get_edge(w, v) in edges:
                # Count current common neighbors of u and w (excluding v)
                cnt = 0
                for x in range(n):
                    if x == u or x == w or x == v:
                        continue
                    if get_edge(u, x) in edges and get_edge(w, x) in edges:
                        cnt += 1
                        if cnt >= 2:
                            return False
                # If cnt == 2, new count is 3 (invalid).
                # If cnt < 2, new count <= 2 (valid).
                
            # Case 2: u is a neighbor of w. Then u becomes a common neighbor of v and w.
            if get_edge(w, u) in edges:
                # Count current common neighbors of v and w (excluding u)
                cnt = 0
                for x in range(n):
                    if x == v or x == w or x == u:
                        continue
                    if get_edge(v, x) in edges and get_edge(w, x) in edges:
                        cnt += 1
                        if cnt >= 2:
                            return False

        return True

    def greedy_init():
        nonlocal edges
        edges = set()
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if can_add(u, v):
                edges.add((u, v))
                
    # Initial Greedy
    greedy_init()
    best_edges = len(edges)
    best_edge_set = set(edges)
    
    # Simulated Annealing
    start_time = time.time()
    time_limit = 110.0
    
    T = 1.0
    T_min = 0.01
    alpha = 0.9999
    
    current_edges = set(best_edge_set)
    current_len = best_edges
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit:
        # Move: Add or Remove
        if random.random() < 0.7:
            # Add
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if get_edge(u, v) in current_edges:
                continue
            
            if can_add(u, v):
                current_edges.add((u, v))
                current_len += 1
                no_improve_count = 0
                if current_len > best_edges:
                    best_edges = current_len
                    best_edge_set = set(current_edges)
            else:
                no_improve_count += 1
        else:
            # Remove
            if not current_edges:
                continue
            u, v = random.choice(list(current_edges))
            current_edges.remove((u, v))
            current_len -= 1
            
            # Accept with probability exp(-1/T)
            # If T is small, prob is small.
            prob = math.exp(-1.0 / T)
            if random.random() >= prob:
                # Reject
                current_edges.add((u, v))
                current_len += 1
            else:
                # Accept removal
                no_improve_count += 1
        
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Kick mechanism
        if no_improve_count > max_no_improve:
            # Remove 10% of edges randomly and re-greedy? 
            # Or just reset T and shuffle?
            # Let's try removing a few random edges to break symmetry
            if current_edges:
                for _ in range(3):
                    if current_edges:
                        u, v = random.choice(list(current_edges))
                        current_edges.remove((u, v))
                        current_len -= 1
            # Re-greedy from current state?
            # Let's just do a local greedy fill
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                if get_edge(u, v) in current_edges:
                    continue
                if can_add(u, v):
                    current_edges.add((u, v))
                    current_len += 1
            if current_len > best_edges:
                best_edges = current_len
                best_edge_set = set(current_edges)
            no_improve_count = 0
            T = 1.0

    result = [[u, v] for u, v in best_edge_set]
    return result

# Test
if __name__ == "__main__":
    #print(construct())
    pass
