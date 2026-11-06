import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = 118.0  # Safety margin before 120s kill

    def time_left():
        return deadline - (time.time() - start_time)

    def adj_to_orient(n, adj):
        """Convert adj matrix (adj[i][j]=1 means i->j) to orientation list."""
        orient = []
        for i in range(n):
            for j in range(i + 1, n):
                orient.append(0 if adj[i][j] == 1 else 1)
        return orient

    def count_violations_numpy(n, out_mask):
        """Count violated 4-sets using numpy. out_mask is n x n boolean array."""
        # For each 4-set {a,b,c,d}, check if any vertex v (not in {a,b,c,d}) dominates all four.
        # Equivalent: for each v, the set of 4-sets it dominates = C(outdeg(v), 4) sets within its out-neighborhood.
        # A 4-set is violated iff it is NOT contained in any out-neighborhood.
        
        # Use inclusion: total 4-sets - 4-sets covered by at least one vertex.
        # This is hard to vectorize directly. Instead, use the complement:
        # A 4-set {a,b,c,d} is covered iff exists v not in {a,b,c,d} with out_mask[v,a]=out_mask[v,b]=out_mask[v,c]=out_mask[v,d]=True.
        
        # For n <= 35, C(n,4) <= 52360. We can enumerate all 4-sets and check.
        # Vectorized approach: for each vertex v, compute all 4-subsets of its out-neighbors.
        # A 4-set is covered if it appears in at least one vertex's out-4-subsets.
        
        # Efficient method: For each 4-set, check all n vertices.
        # n=31: C(31,4)=31465, times 31 checks = ~975k bit ops. Fast with numpy.
        
        from itertools import combinations
        total = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            # Check if any v not in quad dominates all four
            dominated = False
            for v in range(n):
                if v in quad:
                    continue
                if out_mask[v, a] and out_mask[v, b] and out_mask[v, c] and out_mask[v, d]:
                    dominated = True
                    break
            if not dominated:
                total += 1
        return total

    def count_violations_fast(n, out_mask):
        """Faster count using numpy broadcasting for small n."""
        from itertools import combinations
        total = 0
        # Precompute for speed: out_mask is numpy bool array n x n
        quads = list(combinations(range(n), 4))
        for quad in quads:
            a, b, c, d = quad
            # For each v not in quad, check if v->a, v->b, v->c, v->d
            # Vectorize over v
            valid = np.ones(n, dtype=bool)
            valid[a] = valid[b] = valid[c] = valid[d] = False
            # Check domination
            dom = (out_mask[:, a] & out_mask[:, b] & out_mask[:, c] & out_mask[:, d]) & valid
            if not dom.any():
                total += 1
        return total

    def vectorized_deficit(n, out_mask):
        """
        Compute the 'coverage deficit' objective.
        For each vertex v, let S_v = out-neighborhood of v.
        The number of 4-sets NOT dominated by v = C(n-1,4) - C(|S_v \\ {v}|, 4) ... 
        Actually, v can only dominate 4-sets that don't include v.
        Number of 4-sets v can dominate = C(outdeg(v), 4) where outdeg excludes self.
        But we want the total number of uncovered 4-sets.
        
        Alternative objective (easier to optimize):
        deficit = sum over all 4-sets Q of [1 if Q is uncovered else 0]
        
        For gradient-based flip selection, we compute for each edge (u,v):
        delta_deficit if we flip u->v to v->u.
        
        Flipping u->v to v->u:
        - u loses v from out-neighborhood
        - v gains u in out-neighborhood
        
        This affects which 4-sets are covered.
        
        For a practical approach with numpy:
        Compute the "coverage count" for each 4-set (how many vertices dominate it).
        A 4-set is violated if coverage_count == 0.
        
        For n<=35, we can maintain a coverage array of size C(n,4).
        """
        pass

    def full_check_and_count(n, adj):
        """Full verification. Returns (is_valid, violation_count)."""
        out_mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and adj[i][j]:
                    out_mask[i, j] = True
        
        from itertools import combinations
        violations = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            valid = np.ones(n, dtype=bool)
            valid[a] = valid[b] = valid[c] = valid[d] = False
            dom = (out_mask[:, a] & out_mask[:, b] & out_mask[:, c] & out_mask[:, d]) & valid
            if not dom.any():
                violations += 1
        return violations == 0, violations

    def find_near_misses(n, out_mask, max_samples=5000):
        """
        Find 4-sets that are "near-miss" (exactly one vertex misses one edge to dominate).
        Returns list of (quad, v, u) where v almost dominates quad except for edge v->u.
        """
        from itertools import combinations
        near_misses = []
        all_quads = list(combinations(range(n), 4))
        random.shuffle(all_quads)
        
        for quad in all_quads[:max_samples]:
            a, b, c, d = quad
            valid = np.ones(n, dtype=bool)
            valid[a] = valid[b] = valid[c] = valid[d] = False
            
            # For each valid vertex v, count how many of {a,b,c,d} it dominates
            dom_count = out_mask[:, a][valid] + out_mask[:, b][valid] + out_mask[:, c][valid] + out_mask[:, d][valid]
            
            # Find vertices with dom_count == 3 (miss exactly one)
            idx_valid = np.where(valid)[0]
            for pos, v in enumerate(idx_valid):
                if dom_count[pos] == 3:
                    # Find which one is missing
                    if not out_mask[v, a]:
                        near_misses.append((quad, v, a))
                    elif not out_mask[v, b]:
                        near_misses.append((quad, v, b))
                    elif not out_mask[v, c]:
                        near_misses.append((quad, v, c))
                    elif not out_mask[v, d]:
                        near_misses.append((quad, v, d))
            
            # Check if this quad is actually violated
            dom = (out_mask[:, a] & out_mask[:, b] & out_mask[:, c] & out_mask[:, d]) & valid
            if not dom.any():
                # It's violated. Add all near-misses for this quad with higher priority
                pass
        
        return near_misses

    def multi_flip_repair(n, adj, out_mask, max_iters=5000):
        """
        Local search using multi-edge flips and near-miss detection.
        """
        from itertools import combinations
        
        for it in range(max_iters):
            if time_left() < 10:
                break
            
            # Find violated 4-sets and near-miss opportunities
            # Sample some 4-sets to find violations
            found_violation = False
            all_quads = list(combinations(range(n), 4))
            random.shuffle(all_quads)
            
            for quad in all_quads[:200]:
                a, b, c, d = quad
                valid = np.ones(n, dtype=bool)
                valid[a] = valid[b] = valid[c] = valid[d] = False
                
                dom = (out_mask[:, a] & out_mask[:, b] & out_mask[:, c] & out_mask[:, d]) & valid
                if dom.any():
                    continue  # This quad is covered
                
                # Violated quad found. Try to fix it.
                found_violation = True
                
                # For each valid vertex v, find how many edges to flip to make v dominate quad
                idx_valid = np.where(valid)[0]
                best_flip = None
                best_cost = 999
                
                for v in idx_valid:
                    missing = []
                    for x in quad:
                        if not out_mask[v, x]:
                            missing.append(x)
                    cost = len(missing)
                    if cost < best_cost and cost <= 3:  # Allow up to 3 flips
                        best_cost = cost
                        best_flip = (v, missing)
                
                if best_flip is not None:
                    v, missing = best_flip
                    # Flip all missing edges: for each u in missing, flip u->v to v->u
                    for u in missing:
                        if adj[u][v]:  # u->v currently
                            adj[u][v] = 0
                            adj[v][u] = 1
                            out_mask[u, v] = False
                            out_mask[v, u] = True
                        else:
                            # Shouldn't happen, but handle
                            adj[u][v] = 1
                            adj[v][u] = 0
                            out_mask[u, v] = True
                            out_mask[v, u] = False
                    break
            
            if not found_violation:
                # No violations found in sample - do full check
                is_valid, violations = full_check_and_count(n, adj)
                if is_valid:
                    return adj, 0
                # If still violations, try random perturbation
                for _ in range(5):
                    u, v = random.sample(range(n), 2)
                    adj[u][v] = 1 - adj[u][v]
                    adj[v][u] = 1 - adj[v][u]
                    out_mask[u][v] = adj[u][v]
                    out_mask[v][u] = adj[v][u]
        
        # Return current state with violation count
        is_valid, violations = full_check_and_count(n, adj)
        return adj, violations

    def numpy_local_search(n, max_iters=10000):
        """
        Numpy-accelerated local search.
        Objective: minimize number of violated 4-sets.
        Move: flip a single edge or a set of edges.
        """
        # Initialize random tournament
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 0
                else:
                    adj[i][j] = 0
                    adj[j][i] = 1
        
        out_mask = adj.astype(bool)
        
        best_adj = adj.copy()
        best_violations = full_check_and_count(n, adj)[1]
        
        for it in range(max_iters):
            if time_left() < 15:
                break
            
            # Find a violated quad by sampling
            from itertools import combinations
            all_quads = list(combinations(range(n), 4))
            random.shuffle(all_quads)
            
            improved = False
            for quad in all_quads[:100]:
                a, b, c, d = quad
                valid = np.ones(n, dtype=bool)
                valid[a] = valid[b] = valid[c] = valid[d] = False
                
                dom = (out_mask[:, a] & out_mask[:, b] & out_mask[:, c] & out_mask[:, d]) & valid
                if dom.any():
                    continue
                
                # Violated. Try single-edge flips to fix.
                # For each v not in quad, count missing edges
                idx_valid = np.where(valid)[0]
                best_single_flip = None
                best_missing_count = 999
                
                for v in idx_valid:
                    missing = 0
                    miss_list = []
                    for x in quad:
                        if not out_mask[v, x]:
                            missing += 1
                            miss_list.append(x)
                    if missing == 1:
                        # Single flip can fix this quad
                        u = miss_list[0]
                        best_single_flip = (u, v)
                        best_missing_count = 1
                        break
                    elif missing < best_missing_count and missing <= 2:
                        best_missing_count = missing
                        best_single_flip = (miss_list[0], v)  # Partial fix
                
                if best_single_flip is not None:
                    u, v = best_single_flip
                    # Flip u->v to v->u (or vice versa)
                    if adj[u][v]:
                        adj[u][v] = 0
                        adj[v][u] = 1
                        out_mask[u, v] = False
                        out_mask[v, u] = True
                    else:
                        adj[u][v] = 1
                        adj[v][u] = 0
                        out_mask[u, v] = True
                        out_mask[v, u] = False
                    improved = True
                    break
            
            if not improved:
                # Random perturbation to escape local minimum
                for _ in range(3):
                    u, v = random.sample(range(n), 2)
                    adj[u][v] = 1 - adj[u][v]
                    adj[v][u] = 1 - adj[v][u]
                    out_mask[u, v] = bool(adj[u][v])
                    out_mask[v, u] = bool(adj[v][u])
            
            # Periodic full check
            if it % 500 == 0 or it == max_iters - 1:
                _, violations = full_check_and_count(n, adj)
                if violations < best_violations:
                    best_violations = violations
                    best_adj = adj.copy()
                if violations == 0:
                    return adj, 0
        
        return best_adj, best_violations

    # === Main Search ===
    
    # Fallback: Paley(83)
    p = 83
    squares = set(x * x % p for x in range(1, p))
    adj_paley = [[0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                adj_paley[i][j] = 1
            else:
                adj_paley[j][i] = 1
    best_n = p
    best_orient = adj_to_orient(p, adj_paley)
    
    # Try smaller n with numpy local search
    # Start from the theoretical lower bound of 31
    for n in range(31, 45):
        if time_left() < 25:
            break
        
        # Multiple restarts
        found = False
        for restart in range(8):
            if time_left() < 15:
                break
            
            random.seed(12345 + restart * 7919)
            np.random.seed(12345 + restart * 7919)
            
            adj, violations = numpy_local_search(n, max_iters=3000)
            
            if violations == 0:
                # Verify
                is_valid, _ = full_check_and_count(n, adj)
                if is_valid:
                    best_n = n
                    best_orient = adj_to_orient(n, adj.tolist())
                    found = True
                    break
        
        if found:
            break
    
    # If we found something smaller than 83, use it
    # Otherwise return Paley(83)
    
    return {'n': best_n, 'orientation': best_orient}
