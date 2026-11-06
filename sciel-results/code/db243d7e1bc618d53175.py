import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_addition(adj, u, v):
        if v in adj[u]:
            return False
            
        if len(adj[u] & adj[v]) > 2:
            return False
            
        for w in adj[v]:
            if w == u: continue
            if len(adj[u] & adj[w]) > 1:
                return False
                
        for w in adj[u]:
            if w == v: continue
            if len(adj[v] & adj[w]) > 1:
                return False
                
        return True

    def greedy_fill(adj, missing_edges):
        # Returns number of edges added
        if not missing_edges:
            return 0
        miss_list = list(missing_edges)
        random.shuffle(miss_list)
        added = 0
        for u, v in miss_list:
            if is_valid_addition(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                missing_edges.discard((u, v))
                added += 1
        return added

    def run_search(seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        present_edges = set()
        missing_edges = set(all_edges)
        
        # 1. Initial Greedy
        greedy_fill(adj, missing_edges)
        
        best_in_restart = len(present_edges)
        best_adj_in_restart = [set(s) for s in adj]
        best_present_in_restart = set(present_edges)
        
        # 2. Local Search with Kicks
        # We try to improve by removing an edge and re-filling
        # We limit the number of kicks to avoid spending too much time on one restart
        max_kicks = 50
        
        for _ in range(max_kicks):
            # Check time
            if time.time() > deadline - 5.0:
                break
                
            if not present_edges:
                break
                
            # Pick a random edge to remove
            edge_to_remove = random.choice(list(present_edges))
            u, v = edge_to_remove
            
            # Save state
            saved_adj = [set(s) for s in adj]
            saved_present = set(present_edges)
            saved_missing = set(missing_edges)
            
            # Remove edge
            adj[u].remove(v)
            adj[v].remove(u)
            present_edges.remove(edge_to_remove)
            missing_edges.add(edge_to_remove)
            
            # Re-fill
            greedy_fill(adj, missing_edges)
            
            new_count = len(present_edges)
            
            if new_count > best_in_restart:
                best_in_restart = new_count
                best_adj_in_restart = [set(s) for s in adj]
                best_present_in_restart = set(present_edges)
            else:
                # Revert
                adj = saved_adj
                present_edges = saved_present
                missing_edges = saved_missing

        return best_present_in_restart

    # Main Loop
    num_restarts = 0
    while time.time() < deadline:
        seed = num_restarts
        edges_set = run_search(seed)
        count = len(edges_set)
        
        if count > best_count:
            best_count = count
            best_edges = edges_set
            
        num_restarts += 1
        
        if num_restarts % 50 == 0:
             if time.time() > deadline - 10.0:
                 break

    return [list(e) for e in best_edges]
