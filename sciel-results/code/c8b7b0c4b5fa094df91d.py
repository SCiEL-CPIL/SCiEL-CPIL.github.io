# EVOLVE-BLOCK-START
import random
import itertools

def construct():
    n = 15
    all_edges = list(itertools.combinations(range(n), 2))
    
    # Bitmask helpers
    def bit(i):
        return 1 << i
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid(adj, degs, n):
        """
        adj: list of int bitmasks
        degs: list of int degrees
        Returns True if no forbidden 3-path exists.
        """
        # Precompute degree groups as bitmasks
        deg_groups = [0] * (n + 1)
        for i in range(n):
            d = degs[i]
            deg_groups[d] |= bit(i)
            
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & bit(v)):
                    continue
                
                # Path s-u-v-t
                # s in adj[u] \ {v}
                # t in adj[v] \ {u, s}
                
                S_mask = adj[u] & ~bit(v)
                T_mask = adj[v] & ~bit(u)
                
                # Check for each degree d
                for d in range(n + 1):
                    A = S_mask & deg_groups[d]
                    B = T_mask & deg_groups[d]
                    
                    if A and B:
                        # Check if there exists s in A, t in B with s != t
                        # Violation if NOT (A == B and popcount(A) == 1)
                        if not (A == B and popcount(A) == 1):
                            return False
        return True

    def get_adj_degs(edge_set, n):
        adj = [0] * n
        degs = [0] * n
        for u, v in edge_set:
            adj[u] |= bit(v)
            adj[v] |= bit(u)
            degs[u] += 1
            degs[v] += 1
        return adj, degs

    best_edges = set()
    best_count = 0
    
    # Greedy add function
    def greedy_add(current_edges, all_edges, n, max_trials=10):
        current = set(current_edges)
        adj, degs = get_adj_degs(current, n)
        if not is_valid(adj, degs, n):
            # Should not happen if we start valid
            pass
            
        improved = True
        while improved:
            improved = False
            # Shuffle missing edges
            missing = [e for e in all_edges if e not in current]
            random.shuffle(missing)
            
            for e in missing:
                new_edges = current | {e}
                new_adj, new_degs = get_adj_degs(new_edges, n)
                if is_valid(new_adj, new_degs, n):
                    current = new_edges
                    adj, degs = new_adj, new_degs
                    improved = True
        return current

    random.seed(42)
    
    # Run multiple trials with random restarts and perturbations
    for trial in range(200):
        # Start with empty or small random
        if trial % 5 == 0:
            start_edges = set()
        else:
            # Start from best and perturb
            if best_count > 0:
                start_edges = set(best_edges)
                # Remove some random edges to escape local optimum
                k = random.randint(3, 10)
                edges_list = list(start_edges)
                random.shuffle(edges_list)
                for i in range(min(k, len(edges_list))):
                    start_edges.remove(edges_list[i])
            else:
                start_edges = set()
                
        # Run greedy
        res = greedy_add(start_edges, all_edges, n)
        
        if len(res) > best_count:
            best_count = len(res)
            best_edges = res

    # Final local search on best solution: try swaps
    # If we remove edge e1 and add e2, is it valid and same size? No, we want to increase.
    # Try: Remove e1, then greedy add.
    # This is already covered by the perturbation loop.
    
    result = [[u, v] for (u, v) in best_edges]
    final_result = []
    for edge in result:
        u = int(edge[0])
        v = int(edge[1])
        final_result.append([u, v])
        
    return final_result
# EVOLVE-BLOCK-END
