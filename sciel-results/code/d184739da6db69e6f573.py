import time
import random
import numpy as np

def construct():
    """
    Construct a K2,3-free graph on 20 vertices maximizing edges.
    Uses a local search strategy with edge addition and edge swapping.
    Maintains a common-neighbor count matrix for O(1) validity checks (with O(n) updates).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 20
    
    def is_k23_free(adj_mat, common_mat):
        """
        Verify that the graph is K2,3-free.
        A graph has K2,3 if there exist two vertices u, v with at least 3 common neighbors.
        So we just need to check that for all u != v, common_mat[u, v] <= 2.
        """
        # Exclude diagonal
        mask = ~np.eye(n, dtype=bool)
        if np.max(common_mat[mask]) > 2:
            return False
        return True

    def count_common_neighbors(adj_mat):
        """Compute the common neighbor matrix."""
        return (adj_mat @ adj_mat).astype(np.int8)

    def can_add_edge(u, v, adj_mat, common_mat):
        """Check if edge (u,v) can be added to current graph."""
        if adj_mat[u, v]:
            return False
        # Condition 1: common[u, v] <= 2
        if common_mat[u, v] > 2:
            return False
        
        # Condition 2: For all w in N(v) \ {u}, common[u, w] <= 1
        # Note: u is not in N(v) since adj[u,v]=0
        nv = np.nonzero(adj_mat[v])[0]
        if len(nv) > 0:
            if np.max(common_mat[u, nv]) > 1:
                return False
        
        # Condition 3: For all w in N(u) \ {v}, common[v, w] <= 1
        nu = np.nonzero(adj_mat[u])[0]
        if len(nu) > 0:
            if np.max(common_mat[v, nu]) > 1:
                return False
                
        return True

    def perform_add(u, v, adj_mat, common_mat):
        """Add edge (u,v) to adj and common."""
        # Get neighbors BEFORE modification
        nv = np.nonzero(adj_mat[v])[0]
        nu = np.nonzero(adj_mat[u])[0]
        
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1
        
        # Update common
        # common[u, w] += 1 for w in nv
        # common[v, w] += 1 for w in nu
        
        # nv contains no u (since adj[u,v] was 0)
        # nu contains no v
        if len(nv) > 0:
            common_mat[u, nv] += 1
            common_mat[nv, u] += 1
        if len(nu) > 0:
            common_mat[v, nu] += 1
            common_mat[nu, v] += 1

    def perform_remove(u, v, adj_mat, common_mat):
        """Remove edge (u,v) from adj and common."""
        # Get neighbors BEFORE modification
        # Note: u is in N(v) and v is in N(u) currently
        nv = np.nonzero(adj_mat[v])[0]
        nu = np.nonzero(adj_mat[u])[0]
        
        adj_mat[u, v] = 0
        adj_mat[v, u] = 0
        
        # Update common
        # For w in nv: common[u, w] -= 1
        # For w in nu: common[v, w] -= 1
        
        if len(nv) > 0:
            common_mat[u, nv] -= 1
            common_mat[nv, u] -= 1
        if len(nu) > 0:
            common_mat[v, nu] -= 1
            common_mat[nu, v] -= 1

    def greedy_add(adj_mat, common_mat, deadline):
        """
        Greedily add edges until no more can be added.
        Returns the number of edges added.
        """
        added = 0
        while time.time() < deadline:
            added_any = False
            # Generate list of non-edges
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj_mat[u, v]]
            if not pairs:
                break
            random.shuffle(pairs)
            
            for u, v in pairs:
                if time.time() > deadline:
                    return added
                if can_add_edge(u, v, adj_mat, common_mat):
                    perform_add(u, v, adj_mat, common_mat)
                    added_any = True
                    added += 1
                    # After adding, the list of non-edges is stale, 
                    # but we just continue checking the rest. 
                    # can_add_edge will return False for (u,v) now.
                    
            if not added_any:
                break
        return added

    def repair(adj_mat, common_mat, deadline):
        """
        Remove edges to make the graph K2,3-free if it's not.
        """
        if is_k23_free(adj_mat, common_mat):
            return adj_mat, common_mat
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if common_mat[u, v] > 2:
                        common_neighbors = np.nonzero(adj_mat[u] & adj_mat[v])[0]
                        if len(common_neighbors) == 0:
                            continue
                        w = random.choice(common_neighbors.tolist())
                        perform_remove(u, w, adj_mat, common_mat)
                        changed = True
                        break
                if changed:
                    break
        return adj_mat, common_mat

    best_count = -1
    best_adj = None
    
    # Try multiple restarts
    for seed in range(100):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with random sparse graph or empty
        # Start with 5 disjoint K4s as a strong base
        adj = np.zeros((n, n), dtype=np.int8)
        for k in range(5):
            for i in range(4):
                for j in range(i + 1, 4):
                    u = 4 * k + i
                    v = 4 * k + j
                    adj[u, v] = 1
                    adj[v, u] = 1
                    
        common = count_common_neighbors(adj)
        
        # Greedy addition
        greedy_add(adj, common, deadline)
        
        current_count = int(np.sum(adj) // 2)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj.copy()
            
        # Simulated Annealing / Hill Climbing
        T = 1.0
        cooling = 0.99
        min_T = 0.01
        
        while time.time() < deadline and T > min_T:
            # Pick a random edge to remove
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
            if not edges:
                break
            
            rem_u, rem_v = random.choice(edges)
            
            # Backup state
            adj_backup = adj.copy()
            common_backup = common.copy()
            old_count = current_count
            
            # Perform move: remove edge
            perform_remove(rem_u, rem_v, adj, common)
            
            # Try to recover by greedy adding
            greedy_add(adj, common, deadline)
            
            new_count = int(np.sum(adj) // 2)
            delta = new_count - old_count
            
            # Acceptance criterion
            if delta > 0:
                accept = True
            elif delta == 0:
                # Accept equal moves with some probability to explore
                accept = random.random() < 0.5
            else:
                # Accept worse moves based on temperature
                if T > 0:
                    accept_prob = np.exp(delta / T)
                    accept = random.random() < accept_prob
                else:
                    accept = False
            
            if accept:
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj.copy()
            else:
                # Revert
                adj = adj_backup
                common = common_backup
            
            T *= cooling
            
        if not is_k23_free(adj, common):
            adj, common = repair(adj, common, deadline)
            if not is_k23_free(adj, common):
                continue

        cnt = int(np.sum(adj) // 2)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj.copy()

    # If we didn't find any valid graph (shouldn't happen), return empty
    if best_adj is None:
        return []
        
    # Convert best_adj to list of edges
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                best_edges.append([u, v])
                
    return best_edges
