import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmasks(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        conflict = [0] * m
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                    continue
                if (u2 in adj.get(u1, ()) or v2 in adj.get(u1, ()) or
                    u2 in adj.get(v1, ()) or v2 in adj.get(v1, ())):
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        return conflict

    def greedy_independent_set(conflict, candidates, rng):
        """Greedy independent set with random tie-breaking."""
        result = 0
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            # Randomly select a vertex from remaining candidates
            # Weight by inverse degree for better quality
            temp_c = c
            best_v = -1
            best_score = -1
            count = 0
            while temp_c:
                lsb2 = temp_c & -temp_c
                cv = lsb2.bit_length() - 1
                deg = bin(c & conflict[cv]).count('1')
                score = 1.0 / (deg + 1) + rng.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_v = cv
                temp_c ^= lsb2
            if best_v == -1:
                break
            result |= (1 << best_v)
            c &= ~conflict[best_v]
            c &= ~(1 << best_v)
        return result

    def find_max_independent_set(conflict, candidates, time_limit):
        """
        Find a large independent set using Branch and Bound with bitsets.
        Returns the bitmask of the found independent set.
        """
        if candidates == 0:
            return 0
        
        m = len(conflict)
        
        # Initial lower bound via greedy
        greedy_set = greedy_independent_set(conflict, candidates, random.Random(42))
        best_size = bin(greedy_set).count('1')
        best_set = greedy_set
        
        if best_size == 0:
            return 0
        
        # If greedy already found all, return
        if best_size == bin(candidates).count('1'):
            return candidates
        
        def popcount(x):
            return bin(x).count('1')
        
        def dfs(P, X, size):
            nonlocal best_set, best_size
            if time.time() > time_limit:
                return
                
            if P == 0:
                if size > best_size:
                    best_size = size
                    best_set = X
                return
            
            # Upper bound
            if size + popcount(P) <= best_size:
                return
            
            # Select vertex with max degree in P
            max_deg = -1
            v = -1
            temp_P = P
            while temp_P:
                lsb = temp_P & -temp_P
                candidate_v = lsb.bit_length() - 1
                deg = popcount(P & conflict[candidate_v])
                if deg > max_deg:
                    max_deg = deg
                    v = candidate_v
                temp_P ^= lsb
                
            if v == -1:
                return
            
            # Branch 1: Include v
            new_P = P & ~conflict[v] & ~(1 << v)
            dfs(new_P, X | (1 << v), size + 1)
            
            # Branch 2: Exclude v
            new_P = P & ~(1 << v)
            dfs(new_P, X, size)

        dfs(candidates, 0, 0)
        
        return best_set

    def local_search_improve(conflict, color, m, color_id, time_limit):
        """
        Try to reduce color_id by moving edges from the last color class
        to earlier colors using local search.
        """
        if color_id == 0:
            return color, color_id
        
        last_color = color_id - 1
        last_edges = [i for i in range(m) if color[i] == last_color]
        
        if not last_edges:
            return color, color_id - 1
        
        # Try to move each edge to an earlier color
        trial_color = color[:]
        
        for idx in last_edges:
            # Check if this edge can be moved to any earlier color
            moved = False
            nb = conflict[idx]
            for c in range(last_color):
                # Check if color c is available for idx
                # c is available if no neighbor of idx has color c
                valid = True
                temp_nb = nb
                while temp_nb:
                    lsb = temp_nb & -temp_nb
                    j = lsb.bit_length() - 1
                    if trial_color[j] == c:
                        valid = False
                        break
                    temp_nb ^= lsb
                if valid:
                    trial_color[idx] = c
                    moved = True
                    break
            
            if not moved:
                # Try swapping with a neighbor
                # Find a neighbor j that has color c < last_color
                # and idx could take color j, and j could take last_color
                temp_nb = nb
                swapped = False
                while temp_nb and not swapped:
                    lsb = temp_nb & -temp_nb
                    j = lsb.bit_length() - 1
                    if trial_color[j] < last_color:
                        # Check if idx can take color[j] and j can take last_color
                        cj = trial_color[j]
                        
                        # Check if idx can take cj
                        valid_idx = True
                        temp_nb2 = nb & ~(1 << j)  # exclude j itself
                        while temp_nb2:
                            lsb2 = temp_nb2 & -temp_nb2
                            k = lsb2.bit_length() - 1
                            if trial_color[k] == cj:
                                valid_idx = False
                                break
                            temp_nb2 ^= lsb2
                        
                        if valid_idx:
                            # Check if j can take last_color
                            valid_j = True
                            temp_nb3 = conflict[j] & ~(1 << idx)
                            while temp_nb3:
                                lsb3 = temp_nb3 & -temp_nb3
                                k = lsb3.bit_length() - 1
                                if trial_color[k] == last_color:
                                    valid_j = False
                                    break
                                temp_nb3 ^= lsb3
                            
                            if valid_j:
                                # Swap
                                trial_color[idx] = cj
                                trial_color[j] = last_color
                                swapped = True
                    
                    temp_nb ^= lsb
                
                if not swapped:
                    # Can't move this edge, abandon the entire trial
                    return color, color_id
        
        # All edges from last color were moved
        return trial_color, color_id - 1

    def solve_instance(n, d, seed, time_budget):
        edges = instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict = build_conflict_bitmasks(edges, adj)
        
        color = [-1] * m
        remaining = (1 << m) - 1
        color_id = 0
        
        inst_deadline = start_time + time_budget
        
        # Main loop: iteratively find large independent sets
        while remaining:
            if time.time() > inst_deadline - 2:
                # Fallback: assign unique colors to remaining
                c_rem = remaining
                while c_rem:
                    lsb = c_rem & -c_rem
                    idx = lsb.bit_length() - 1
                    if color[idx] == -1:
                        color[idx] = color_id
                        color_id += 1
                    c_rem ^= lsb
                break
            
            # Find large independent set with time-limited B&B
            bnb_time = min(inst_deadline - 3 - time.time(), 1.5)
            if bnb_time < 0.1:
                bnb_time = 0.1
            
            iso_set = find_max_independent_set(conflict, remaining, time.time() + bnb_time)
            
            if iso_set == 0:
                lsb = remaining & -remaining
                iso_set = lsb
            
            # Assign color
            c_set = iso_set
            while c_set:
                lsb = c_set & -c_set
                idx = lsb.bit_length() - 1
                color[idx] = color_id
                c_set ^= lsb
            
            remaining &= ~iso_set
            color_id += 1
            
            # Try local search to reduce colors
            if time.time() < inst_deadline - 5 and color_id > 1:
                color, color_id = local_search_improve(conflict, color, m, color_id, inst_deadline - 5)
                
                # Update remaining based on new color assignment
                remaining = 0
                for i in range(m):
                    if color[i] < 0:
                        remaining |= (1 << i)
                
                # If colors were reduced, the last color class might be empty
                # and we need to continue with the correct color_id
        
        return color

    out = []
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    
    # Allocate time budget: give more time to harder instances
    # Instance 3 (d=6) is hardest, give it more time
    base_budget = 40
    for i, spec in enumerate(specs):
        n, d, seed = spec
        budget = base_budget + i * 5  # 40, 45, 50
        out.append(solve_instance(n, d, seed, budget))
    
    while len(out) < 3:
        out.append([])
    
    return out
