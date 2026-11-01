import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate candidate vectors with norm^2 = 4
    # Weight 1: one entry ±2 -> norm^2 = 4
    # Weight 2: two entries ±1 -> norm^2 = 2 (scale by sqrt(2)? No, keep integer)
    # Actually, let's use:
    # Type A: weight 1, entries ±2 -> norm^2 = 4
    # Type B: weight 2, entries ±1 -> norm^2 = 2 (these are smaller)
    # Type C: weight 4, entries ±1 -> norm^2 = 4
    
    # For the kissing number problem, we need min pairwise distance >= max norm
    # If max norm is 2 (norm^2=4), then min distance must be >= 2
    # Distance^2 = |v|^2 + |w|^2 - 2*v.w >= 4
    # For vectors with norm^2=4: 4+4-2*v.w >= 4 => v.w <= 2
    # For vectors with norm^2=2: 2+2-2*v.w >= 4 => v.w <= 0
    # Mixed: 4+2-2*v.w >= 4 => v.w <= 1
    
    # Let's focus on vectors with norm^2 = 4 for simplicity, but include weight-2 as well
    # and handle the mixed constraints properly.
    
    # Actually, let's use a unified approach: all vectors have norm^2 = 4
    # Weight 1: ±2 in one position
    # Weight 4: ±1 in four positions
    
    # But we can also use weight 2 vectors with entries ±1 (norm^2=2) and just ensure
    # the distance constraint holds. Let's include them and check constraints properly.
    
    # Generate all candidates:
    candidates = []
    
    # Type 1: weight 1, value ±2 (norm^2 = 4)
    for pos in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[pos] = 2 * sign
            candidates.append(v)
    
    # Type 2: weight 2, values ±1 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    
    # Type 3: weight 4, values ±1 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    n = len(candidates)
    mat = np.array(candidates, dtype=np.int64)
    
    # Compute norms squared
    norms_sq = np.sum(mat * mat, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Compute pairwise dot products
    dots = mat @ mat.T
    
    # Conflict: distance^2 < max_norm_sq
    # dist^2 = norm_i^2 + norm_j^2 - 2*dot(i,j) < max_norm_sq
    # => dot(i,j) > (norm_i^2 + norm_j^2 - max_norm_sq) / 2
    conflict = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = norms_sq[i] + norms_sq[j] - 2 * dots[i, j]
            if dist_sq < max_norm_sq:
                conflict[i, j] = True
                conflict[j, i] = True
    
    # Build adjacency lists as bitsets for fast operations
    # Use Python integers as bitsets
    adj_bits = [0] * n
    for i in range(n):
        bits = 0
        for j in range(n):
            if conflict[i, j]:
                bits |= (1 << j)
        adj_bits[i] = bits
    
    # Also store neighbor lists for quick access
    neighbors = [np.array([j for j in range(n) if conflict[i, j]], dtype=np.int32) for i in range(n)]
    
    def bit_count(x):
        return bin(x).count('1')
    
    def greedy_init(rng):
        """Greedy independent set initialization with random tie-breaking."""
        in_set = set()
        available = set(range(n))
        
        while available and time.time() < deadline - 10:
            # Pick vertex with minimum degree in available subgraph
            best_v = None
            best_deg = float('inf')
            for v in available:
                deg = len([u for u in available if conflict[v, u]])
                if deg < best_deg or (deg == best_deg and rng.random() < 0.5):
                    best_deg = deg
                    best_v = v
            
            if best_v is None:
                break
                
            in_set.add(best_v)
            # Remove best_v and its neighbors from available
            to_remove = {best_v}
            for u in neighbors[best_v]:
                if u in available:
                    to_remove.add(u)
            available -= to_remove
        
        return in_set
    
    def local_search(in_set, max_time=15.0):
        """Local search with 1-for-0, 0-for-1, and 2-for-1 swap moves."""
        in_set = set(in_set)
        current_size = len(in_set)
        best_size = current_size
        best_set = set(in_set)
        
        T = 2.0
        cooling = 0.999
        min_T = 0.1
        sa_deadline = time.time() + max_time
        stuck = 0
        
        # Precompute conflict counts for vertices not in set
        def compute_conflict_counts():
            counts = {}
            for v in in_set:
                for u in neighbors[v]:
                    if u not in in_set:
                        counts[u] = counts.get(u, 0) + 1
            return counts
        
        conflict_counts = compute_conflict_counts()
        
        while time.time() < sa_deadline:
            # Try moves
            move_type = rng.random()
            
            if move_type < 0.3:
                # 0-for-1: add a vertex not in set
                not_in_set = [v for v in range(n) if v not in in_set]
                if not not_in_set:
                    continue
                    
                # Find vertex with minimum conflicts with in_set
                best_v = None
                best_conf = float('inf')
                for v in not_in_set:
                    c = conflict_counts.get(v, 0)
                    if c < best_conf:
                        best_conf = c
                        best_v = v
                
                if best_v is None:
                    continue
                    
                # Determine which vertices to remove
                to_remove = [u for u in neighbors[best_v] if u in in_set]
                delta = 1 - len(to_remove)
                
                if delta > 0 or (delta == 0 and rng.random() < 0.5) or (delta < 0 and rng.random() < np.exp(delta / T)):
                    for u in to_remove:
                        in_set.remove(u)
                        # Update conflict counts
                        for w in neighbors[u]:
                            if w in in_set:
                                conflict_counts[w] = conflict_counts.get(w, 0) - 1
                            else:
                                if w in conflict_counts:
                                    conflict_counts[w] = conflict_counts.get(w, 0) - 1
                                    if conflict_counts[w] == 0:
                                        del conflict_counts[w]
                    
                    in_set.add(best_v)
                    # Update conflict counts for new vertex
                    for w in neighbors[best_v]:
                        if w not in in_set:
                            conflict_counts[w] = conflict_counts.get(w, 0) + 1
                    
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_set = set(in_set)
                        stuck = 0
                    else:
                        stuck += 1
                        
            elif move_type < 0.6:
                # 1-for-0: remove a vertex from set (rarely helpful, but for SA)
                if not in_set:
                    continue
                v = list(in_set)[rng.integers(len(in_set))]
                
                # Removing v frees up its neighbors
                freed = [u for u in neighbors[v] if u not in in_set]
                delta = -1 + len(freed)  # This is wrong; removing one vertex doesn't add others automatically
                
                # Actually, just remove and see if we can add back later
                if rng.random() < 0.1:  # Rarely remove
                    in_set.remove(v)
                    for w in neighbors[v]:
                        if w not in in_set:
                            if w in conflict_counts:
                                conflict_counts[w] = conflict_counts.get(w, 0) - 1
                                if conflict_counts[w] == 0:
                                    del conflict_counts[w]
                    current_size -= 1
                    stuck += 1
                    
            else:
                # 2-for-1 swap: remove two vertices, add one
                if len(in_set) < 2:
                    continue
                in_list = list(in_set)
                v1_idx = rng.integers(len(in_list))
                v2_idx = rng.integers(len(in_list) - 1)
                if v2_idx >= v1_idx:
                    v2_idx += 1
                v1, v2 = in_list[v1_idx], in_list[v2_idx]
                
                # Find a vertex to add that conflicts with both v1 and v2
                candidates_add = [u for u in range(n) if u not in in_set and conflict[u, v1] and conflict[u, v2]]
                if not candidates_add:
                    continue
                    
                best_u = None
                best_gain = float('-inf')
                for u in candidates_add:
                    # Gain = 1 (add u) - (number of other vertices in in_set that conflict with u, excluding v1, v2)
                    other_conflicts = [w for w in in_set if w != v1 and w != v2 and conflict[u, w]]
                    gain = 1 - len(other_conflicts)
                    if gain > best_gain:
                        best_gain = gain
                        best_u = u
                
                if best_u is None or best_gain < 0:
                    continue
                    
                other_conflicts = [w for w in in_set if w != v1 and w != v2 and conflict[best_u, w]]
                delta = 1 - len(other_conflicts)  # We remove v1, v2 (2 vertices) and add best_u (1 vertex)
                # Wait, we're removing v1 and v2, and adding best_u
                # Net change = 1 - 2 = -1, plus we free up neighbors of v1 and v2
                # This is complex; let's just do it and check validity
                
                # Remove v1, v2 and their conflicts with best_u
                to_remove = {v1, v2}
                for w in other_conflicts:
                    to_remove.add(w)
                
                # Check if adding best_u is valid after removing to_remove
                remaining = in_set - to_remove
                if all(not conflict[best_u, w] for w in remaining):
                    # Accept with probability
                    actual_delta = len(remaining) + 1 - len(in_set)
                    if actual_delta > 0 or (actual_delta == 0 and rng.random() < 0.5) or (actual_delta < 0 and rng.random() < np.exp(actual_delta / T)):
                        for w in to_remove:
                            in_set.remove(w)
                            for z in neighbors[w]:
                                if z not in in_set:
                                    if z in conflict_counts:
                                        conflict_counts[z] = conflict_counts.get(z, 0) - 1
                                        if conflict_counts[z] == 0:
                                            del conflict_counts[z]
                        
                        in_set.add(best_u)
                        for z in neighbors[best_u]:
                            if z not in in_set:
                                conflict_counts[z] = conflict_counts.get(z, 0) + 1
                        
                        current_size = len(in_set)
                        if current_size > best_size:
                            best_size = current_size
                            best_set = set(in_set)
                            stuck = 0
                        else:
                            stuck += 1
            
            if stuck > 2000:
                T = max(T, 1.5)
                stuck = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return best_set, best_size
    
    # Main search loop with multiple restarts
    rng = np.random.default_rng(42)
    best_set = set()
    best_size = 0
    
    n_restarts = 0
    max_restarts = 200
    
    while time.time() < deadline - 15 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Greedy initialization
        in_set = greedy_init(rng)
        current_size = len(in_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = set(in_set)
        
        # Local search
        remaining = deadline - 20 - time.time()
        if remaining < 3:
            break
        
        ls_time = min(remaining * 0.7, 20.0)
        if ls_time < 2:
            ls_time = 2.0
        
        ls_set, ls_size = local_search(in_set, max_time=ls_time)
        
        if ls_size > best_size:
            best_size = ls_size
            best_set = set(ls_set)
        
        # Polish with another local search pass
        remaining = deadline - 15 - time.time()
        if remaining > 5 and ls_size > current_size:
            polish_time = min(remaining * 0.5, 10.0)
            if polish_time > 2:
                ps_set, ps_size = local_search(ls_set, max_time=polish_time)
                if ps_size > best_size:
                    best_size = ps_size
                    best_set = set(ps_set)
    
    # Convert back to array
    if best_size == 0:
        # Fallback: just use the original seed program's result
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result_indices = list(best_set)
    result = mat[result_indices]
    
    return result
