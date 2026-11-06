import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 148  # Safety margin before 150s kill
    
    n = 16
    m_target = 65
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def calc_score(adj, deg):
        """Fast bitmask-based score calculation."""
        # Convert adjacency to bitmask representation
        masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i, j]:
                    mask |= (1 << j)
            masks[i] = mask
        
        max_sum = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    # Common neighbors of u and v
                    common = masks[u] & masks[v]
                    # Only consider w > v to avoid duplicates
                    common &= ~((1 << (v+1)) - 1)
                    if common:
                        # Find all w in common
                        w = common
                        while w:
                            # Extract lowest set bit
                            low = w & (-w)
                            wi = low.bit_length() - 1
                            s = deg[u] + deg[v] + deg[wi]
                            if s > max_sum:
                                max_sum = s
                            w &= w - 1
        return max_sum
    
    def generate_k610_structure():
        """Generate K_{6,10} + 5 matching edges structure with random vertex labeling."""
        adj = np.zeros((n, n), dtype=int)
        # Randomly assign 6 vertices to A, 10 to B
        vertices = list(range(n))
        random.shuffle(vertices)
        A = vertices[:6]
        B = vertices[6:]
        
        # Bipartite edges A-B
        for u in A:
            for v in B:
                adj[u, v] = 1
                adj[v, u] = 1
        
        # 5 disjoint edges in B (perfect matching on 10 vertices)
        random.shuffle(B)
        for i in range(5):
            u, v = B[2*i], B[2*i+1]
            adj[u, v] = 1
            adj[v, u] = 1
            
        deg = np.array([np.sum(adj[i]) for i in range(n)])
        return adj, deg
    
    def local_search(adj, deg, max_iter=5000, time_limit=10):
        """Hill climbing with random restarts within a phase."""
        current_score = calc_score(adj, deg)
        best_score = current_score
        best_adj = adj.copy()
        best_deg = deg.copy()
        
        edges_set = set()
        non_edges_set = set()
        for u, v in all_pairs:
            if adj[u, v]:
                edges_set.add((u, v))
            else:
                non_edges_set.add((u, v))
        
        edges_list = list(edges_set)
        non_edges_list = list(non_edges_set)
        
        T = 0.3
        cooling = 0.9995
        no_improve_count = 0
        
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            # Pick random edge to remove and non-edge to add
            if not edges_list or not non_edges_list:
                break
                
            u, v = random.choice(edges_list)
            x, y = random.choice(non_edges_list)
            
            # Apply swap
            adj[u, v] = 0
            adj[v, u] = 0
            adj[x, y] = 1
            adj[y, x] = 1
            
            deg[u] -= 1
            deg[v] -= 1
            deg[x] += 1
            deg[y] += 1
            
            new_score = calc_score(adj, deg)
            
            if new_score < current_score:
                # Accept improvement
                current_score = new_score
                no_improve_count = 0
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj.copy()
                    best_deg = deg.copy()
                
                # Update sets
                edges_set.remove((u, v))
                non_edges_set.add((u, v))
                edges_set.add((x, y))
                non_edges_set.remove((x, y))
                
                edges_list = list(edges_set)
                non_edges_list = list(non_edges_set)
                
            elif new_score == current_score:
                # Accept equal with higher probability for exploration
                if random.random() < 0.5:
                    no_improve_count += 1
                    edges_set.remove((u, v))
                    non_edges_set.add((u, v))
                    edges_set.add((x, y))
                    non_edges_set.remove((x, y))
                    
                    edges_list = list(edges_set)
                    non_edges_list = list(non_edges_set)
                else:
                    # Revert
                    adj[u, v] = 1
                    adj[v, u] = 1
                    adj[x, y] = 0
                    adj[y, x] = 0
                    
                    deg[u] += 1
                    deg[v] += 1
                    deg[x] -= 1
                    deg[y] -= 1
                    
            else:
                # Metropolis acceptance
                delta = new_score - current_score
                if random.random() < np.exp(-delta / T):
                    no_improve_count += 1
                    edges_set.remove((u, v))
                    non_edges_set.add((u, v))
                    edges_set.add((x, y))
                    non_edges_set.remove((x, y))
                    
                    edges_list = list(edges_set)
                    non_edges_list = list(non_edges_set)
                else:
                    # Revert
                    adj[u, v] = 1
                    adj[v, u] = 1
                    adj[x, y] = 0
                    adj[y, x] = 0
                    
                    deg[u] += 1
                    deg[v] += 1
                    deg[x] -= 1
                    deg[y] -= 1
                    
            T *= cooling
            if T < 0.01:
                T = 0.01
                
            # If stuck for too long, perturb more aggressively
            if no_improve_count > 150:
                # Do a few random perturbations
                num_perturb = random.randint(2, 5)
                for _ in range(num_perturb):
                    if not edges_list or not non_edges_list:
                        break
                    u2, v2 = random.choice(edges_list)
                    x2, y2 = random.choice(non_edges_list)
                    
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0
                    adj[x2, y2] = 1
                    adj[y2, x2] = 1
                    
                    deg[u2] -= 1
                    deg[v2] -= 1
                    deg[x2] += 1
                    deg[y2] += 1
                    
                    edges_set.remove((u2, v2))
                    non_edges_set.add((u2, v2))
                    edges_set.add((x2, y2))
                    non_edges_set.remove((x2, y2))
                    
                    edges_list = list(edges_set)
                    non_edges_list = list(non_edges_set)
                
                current_score = calc_score(adj, deg)
                no_improve_count = 0
                T = 0.3
        
        return best_adj, best_deg, best_score
    
    def polish(adj, deg, time_limit=5):
        """Final high-precision polish pass with stricter acceptance."""
        current_score = calc_score(adj, deg)
        best_score = current_score
        best_adj = adj.copy()
        best_deg = deg.copy()
        
        edges_set = set()
        non_edges_set = set()
        for u, v in all_pairs:
            if adj[u, v]:
                edges_set.add((u, v))
            else:
                non_edges_set.add((u, v))
        
        edges_list = list(edges_set)
        non_edges_list = list(non_edges_set)
        
        T = 0.1
        cooling = 0.9999
        no_improve_count = 0
        iter_count = 0
        
        while time.time() - start_time < time_limit:
            if not edges_list or not non_edges_list:
                break
            
            iter_count += 1
            
            # Try multiple candidate swaps and pick the best
            best_candidate = None
            best_candidate_score = current_score
            
            for trial in range(5):
                if not edges_list or not non_edges_list:
                    break
                u, v = random.choice(edges_list)
                x, y = random.choice(non_edges_list)
                
                # Apply swap temporarily
                adj[u, v] = 0
                adj[v, u] = 0
                adj[x, y] = 1
                adj[y, x] = 1
                
                deg[u] -= 1
                deg[v] -= 1
                deg[x] += 1
                deg[y] += 1
                
                trial_score = calc_score(adj, deg)
                
                if trial_score < best_candidate_score:
                    best_candidate_score = trial_score
                    best_candidate = (u, v, x, y)
                
                # Revert for next trial
                adj[u, v] = 1
                adj[v, u] = 1
                adj[x, y] = 0
                adj[y, x] = 0
                
                deg[u] += 1
                deg[v] += 1
                deg[x] -= 1
                deg[y] -= 1
            
            if best_candidate is None:
                # No improvement found, try Metropolis
                if edges_list and non_edges_list:
                    u, v = random.choice(edges_list)
                    x, y = random.choice(non_edges_list)
                    
                    adj[u, v] = 0
                    adj[v, u] = 0
                    adj[x, y] = 1
                    adj[y, x] = 1
                    
                    deg[u] -= 1
                    deg[v] -= 1
                    deg[x] += 1
                    deg[y] += 1
                    
                    new_score = calc_score(adj, deg)
                    
                    if new_score <= current_score or random.random() < np.exp(-(new_score - current_score) / T):
                        current_score = new_score
                        edges_set.remove((u, v))
                        non_edges_set.add((u, v))
                        edges_set.add((x, y))
                        non_edges_set.remove((x, y))
                        
                        edges_list = list(edges_set)
                        non_edges_list = list(non_edges_set)
                    else:
                        adj[u, v] = 1
                        adj[v, u] = 1
                        adj[x, y] = 0
                        adj[y, x] = 0
                        
                        deg[u] += 1
                        deg[v] += 1
                        deg[x] -= 1
                        deg[y] -= 1
                
                no_improve_count += 1
                T *= cooling
                if T < 0.005:
                    T = 0.005
                
                if no_improve_count > 100:
                    break
            else:
                u, v, x, y = best_candidate
                # Apply the best candidate
                adj[u, v] = 0
                adj[v, u] = 0
                adj[x, y] = 1
                adj[y, x] = 1
                
                deg[u] -= 1
                deg[v] -= 1
                deg[x] += 1
                deg[y] += 1
                
                current_score = best_candidate_score
                no_improve_count = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj.copy()
                    best_deg = deg.copy()
                
                edges_set.remove((u, v))
                non_edges_set.add((u, v))
                edges_set.add((x, y))
                non_edges_set.remove((x, y))
                
                edges_list = list(edges_set)
                non_edges_list = list(non_edges_set)
        
        return best_adj, best_deg, best_score
    
    # Run multiple independent searches with different initializations
    best_overall_score = float('inf')
    best_overall_adj = None
    best_overall_deg = None
    
    num_restarts = 10
    time_per_restart = (deadline - (time.time() - start_time) - 8) / num_restarts  # Reserve 8s for polish
    time_per_restart = min(time_per_restart, 12)  # Cap per restart
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Generate a diverse initial structure
        if restart == 0:
            # Use the known good K_{6,10} structure
            adj, deg = generate_k610_structure()
        elif restart == 1:
            # K_{7,9} with some internal edges
            adj = np.zeros((n, n), dtype=int)
            vertices = list(range(n))
            random.shuffle(vertices)
            A = vertices[:7]
            B = vertices[7:]
            
            for u in A:
                for v in B:
                    adj[u, v] = 1
                    adj[v, u] = 1
            
            # Add edges to reach 65 total (need 65 - 63 = 2 more)
            random.shuffle(B)
            adj[B[0], B[1]] = 1
            adj[B[1], B[0]] = 1
            
            deg = np.array([np.sum(adj[i]) for i in range(n)])
        elif restart == 2:
            # K_{5,11} with more internal edges
            adj = np.zeros((n, n), dtype=int)
            vertices = list(range(n))
            random.shuffle(vertices)
            A = vertices[:5]
            B = vertices[5:]
            
            for u in A:
                for v in B:
                    adj[u, v] = 1
                    adj[v, u] = 1
            
            # Need 65 - 55 = 10 more edges in B
            random.shuffle(B)
            for i in range(5):
                adj[B[2*i], B[2*i+1]] = 1
                adj[B[2*i+1], B[2*i]] = 1
            # Add 5 more random edges in B
            b_edges = set()
            for i in range(len(B)):
                for j in range(i+1, len(B)):
                    b_edges.add((B[i], B[j]))
            b_edges = list(b_edges)
            random.shuffle(b_edges)
            for i in range(5):
                u, v = b_edges[i]
                if not adj[u, v]:
                    adj[u, v] = 1
                    adj[v, u] = 1
            
            deg = np.array([np.sum(adj[i]) for i in range(n)])
        else:
            # For other restarts, start from K_{6,10} but with random perturbations
            adj, deg = generate_k610_structure()
            # Apply a few random swaps to diversify
            edges_set = set((u, v) for u, v in all_pairs if adj[u, v])
            non_edges_set = set((u, v) for u, v in all_pairs if not adj[u, v])
            
            for _ in range(random.randint(3, 7)):
                if not edges_set or not non_edges_set:
                    break
                u, v = random.choice(list(edges_set))
                x, y = random.choice(list(non_edges_set))
                
                adj[u, v] = 0
                adj[v, u] = 0
                adj[x, y] = 1
                adj[y, x] = 1
                
                deg[u] -= 1
                deg[v] -= 1
                deg[x] += 1
                deg[y] += 1
                
                edges_set.remove((u, v))
                non_edges_set.add((u, v))
                edges_set.add((x, y))
                non_edges_set.remove((x, y))
        
        # Run local search
        adj_ls, deg_ls, score_ls = local_search(adj, deg, max_iter=4000, time_limit=time_per_restart)
        
        if score_ls < best_overall_score:
            best_overall_score = score_ls
            best_overall_adj = adj_ls
            best_overall_deg = deg_ls
            
            # Early termination if we hit the theoretical target
            if best_overall_score <= 23:
                break
    
    # Final answer
    if best_overall_adj is None:
        # Fallback to simple structure
        adj, deg = generate_k610_structure()
        best_overall_adj = adj
        best_overall_deg = deg
        best_overall_score = calc_score(adj, deg)
    
    # Final polish pass
    if time.time() < deadline - 3:
        polish_time = min(8, deadline - 3 - (time.time() - start_time))
        if polish_time > 1:
            best_overall_adj, best_overall_deg, best_overall_score = polish(
                best_overall_adj, best_overall_deg, time_limit=polish_time
            )
    
    # Reconstruct edge list
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_overall_adj[i, j]:
                final_edges.append([i, j])
                
    return {'n': n, 'edges': final_edges}
