import numpy as np
import time
import heapq
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    k_field = 3  # alpha^2 = 3 in F_7
    n_points = q * q  # 2401
    
    # Precompute field arithmetic using integer representation
    # F_49 = F_7[alpha], alpha^2 = 3
    # Element represented as a + b*alpha -> integer a + b*7
    
    def to_int(a, b):
        return a + b * p
    
    def to_pair(x):
        return (x % p, (x // p) % p)
    
    # Precompute addition and multiplication tables
    f_add = np.zeros((q, q), dtype=np.int32)
    f_mul = np.zeros((q, q), dtype=np.int32)
    f_neg = np.zeros(q, dtype=np.int32)
    
    for x in range(q):
        xa, xb = to_pair(x)
        f_neg[x] = to_int((-xa) % p, (-xb) % p)
        for y in range(q):
            ya, yb = to_pair(y)
            f_add[x, y] = to_int((xa + ya) % p, (xb + yb) % p)
            real = (xa * ya + xb * yb * k_field) % p
            imag = (xa * yb + xb * ya) % p
            f_mul[x, y] = to_int(real, imag)
    
    # Directions: (1, m) for m in F_49, and (0, 1)
    directions = []
    for m in range(q):
        directions.append((1, m))
    directions.append((0, 1))
    n_dir = len(directions)  # 50
    
    # Point indexing: point (x, y) in F_49^2 -> index x * q + y
    def point_to_idx(x, y):
        return x * q + y
    
    def idx_to_point(idx):
        return (idx // q, idx % q)
    
    # Precompute for each point and direction, the list of other points on the line
    # line_members[idx][d] = list of (other_idx) for the line through idx with direction d
    line_members = [[[] for _ in range(n_dir)] for _ in range(n_points)]
    
    for idx in range(n_points):
        x, y = idx_to_point(idx)
        for d in range(n_dir):
            vx, vy = directions[d]
            members = []
            for t in range(1, q):
                tv_x = f_mul[t, vx]
                tv_y = f_mul[t, vy]
                nx = f_add[x, tv_x]
                ny = f_add[y, tv_y]
                nidx = point_to_idx(nx, ny)
                members.append(nidx)
            line_members[idx][d] = members
    
    # Build initial N using Hermitian unital construction
    def build_initial_in_N():
        in_N = np.ones(n_points, dtype=bool)
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(to_int((u * u) % p, v))
        
        for x in range(q):
            xa, xb = to_pair(x)
            trace_x = (2 * xa) % p
            for y in range(q):
                ya, yb = to_pair(y)
                norm_y = (ya * ya - yb * yb * k_field) % p
                if (trace_x + norm_y) % p == 0:
                    if y not in y_special:
                        in_N[point_to_idx(x, y)] = False
        return in_N
    
    # Compute not_in_N_count: for each point and direction, how many points on the line (other than the point itself) are NOT in N
    def compute_not_in_N_count(in_N):
        not_in_N_count = np.zeros((n_points, n_dir), dtype=np.int32)
        for d in range(n_dir):
            for idx in range(n_points):
                count = 0
                for nidx in line_members[idx][d]:
                    if not in_N[nidx]:
                        count += 1
                not_in_N_count[idx, d] = count
        return not_in_N_count
    
    # Compute cover_count: for each point, how many directions have not_in_N_count == 0
    def compute_cover_count(not_in_N_count):
        cover_count = np.zeros(n_points, dtype=np.int32)
        for idx in range(n_points):
            for d in range(n_dir):
                if not_in_N_count[idx, d] == 0:
                    cover_count[idx] += 1
        return cover_count
    
    # Efficient state tracking for greedy removal
    class RemovalState:
        def __init__(self, in_N):
            self.in_N = in_N.copy()
            self.not_in_N_count = compute_not_in_N_count(self.in_N)
            self.cover_count = compute_cover_count(self.not_in_N_count)
            # For each point, track the minimum not_in_N_count across directions
            # and how many directions achieve that minimum
            self.min_c = np.zeros(n_points, dtype=np.int32)
            self.count_min = np.zeros(n_points, dtype=np.int32)
            self._update_min_stats()
        
        def _update_min_stats(self):
            for idx in range(n_points):
                if not self.in_N[idx]:
                    continue
                mn = 999999
                cnt = 0
                for d in range(n_dir):
                    c = self.not_in_N_count[idx, d]
                    if c < mn:
                        mn = c
                        cnt = 1
                    elif c == mn:
                        cnt += 1
                self.min_c[idx] = mn
                self.count_min[idx] = cnt
        
        def _recompute_min_stats_for(self, idx):
            mn = 999999
            cnt = 0
            for d in range(n_dir):
                c = self.not_in_N_count[idx, d]
                if c < mn:
                    mn = c
                    cnt = 1
                elif c == mn:
                    cnt += 1
            self.min_c[idx] = mn
            self.count_min[idx] = cnt
        
        def remove_point(self, idx):
            """Remove point idx from N. Returns True if valid, False otherwise."""
            if not self.in_N[idx]:
                return False
            
            # Check if removal is valid:
            # For each (p_idx, d) affected by removing idx, if not_in_N_count[p_idx, d] was 0,
            # then cover_count[p_idx] decreases. If it becomes 0, invalid.
            # Actually, we need to check: after removal, for each point p on lines through idx,
            # the not_in_N_count[p, d] increases by 1 (since idx is no longer in N).
            # This means lines that previously had idx as the only missing point now have idx+1 missing.
            # Wait, I need to re-think the validity condition.
            
            # A point x is covered if there exists a direction d such that ALL other points on the line
            # through x in direction d are in N.
            # not_in_N_count[x, d] = number of points on line through x in dir d (excluding x) NOT in N.
            # x is covered in direction d if not_in_N_count[x, d] == 0.
            # x is covered if cover_count[x] >= 1.
            
            # When we remove idx from N:
            # For each point p and direction d such that idx is on the line through p in direction d
            # (i.e., idx is in line_members[p][d]), not_in_N_count[p, d] increases by 1.
            # If not_in_N_count[p, d] was 0, it becomes 1, so cover_count[p] decreases by 1.
            # If cover_count[p] was 1, it becomes 0, which is invalid.
            
            # So we need to check: for all p where idx is in line_members[p][d],
            # if not_in_N_count[p, d] == 0 and cover_count[p] == 1, then removal is invalid.
            
            # To do this efficiently, we precompute for each idx, the list of (p, d) where idx is in line_members[p][d]
            # This is the 'affects' mapping.
            
            # Let's compute it on the fly or precompute it.
            # For efficiency, let's precompute affects outside.
            
            return True  # Placeholder, will be replaced
    
    # Precompute affects: for each point r, list of (p, d) such that r is in line_members[p][d]
    affects = [[] for _ in range(n_points)]
    for r in range(n_points):
        for p in range(n_points):
            for d in range(n_dir):
                if r in line_members[p][d]:
                    affects[r].append((p, d))
    
    # More efficient: for each r, for each d, the points p such that r is on the line through p in direction d
    # Actually, line_members[p][d] contains all other points on the line through p in direction d.
    # So r is in line_members[p][d] iff p is in line_members[r][d] (by symmetry of lines).
    # Wait, not exactly. The line through p in direction d contains p and all p + t*v for t in F_q.
    # r is on this line iff r = p + t*v for some t != 0, which means p = r + (-t)*v, so p is on the line through r in direction d.
    # So yes, r is in line_members[p][d] iff p is in line_members[r][d].
    
    # So affects[r] = [(p, d) for d in range(n_dir) for p in line_members[r][d]]
    for r in range(n_points):
        for d in range(n_dir):
            for p in line_members[r][d]:
                affects[r].append((p, d))
    
    # Now implement the removal state properly
    class RemovalState:
        def __init__(self, in_N):
            self.in_N = in_N.copy()
            self.not_in_N_count = compute_not_in_N_count(self.in_N)
            self.cover_count = compute_cover_count(self.not_in_N_count)
        
        def can_remove(self, idx):
            """Check if removing idx from N keeps all points covered."""
            if not self.in_N[idx]:
                return False
            for p, d in affects[idx]:
                if self.not_in_N_count[p, d] == 0 and self.cover_count[p] == 1:
                    return False
            return True
        
        def remove(self, idx):
            """Remove idx from N. Assumes can_remove(idx) is True."""
            if not self.in_N[idx]:
                return
            for p, d in affects[idx]:
                old = self.not_in_N_count[p, d]
                self.not_in_N_count[p, d] = old + 1
                if old == 0:
                    self.cover_count[p] -= 1
            self.in_N[idx] = False
    
    # Priority queue based greedy removal
    def greedy_removal(in_N, deadline, time_budget_per_restart=50):
        state = RemovalState(in_N)
        
        # Compute initial cost for each point in N
        # Cost = number of (p, d) in affects[idx] where not_in_N_count[p, d] == 0
        # We want to remove points with low cost first
        cost_cache = {}
        heap = []
        
        for idx in range(n_points):
            if state.in_N[idx]:
                cost = 0
                for p, d in affects[idx]:
                    if state.not_in_N_count[p, d] == 0:
                        cost += 1
                cost_cache[idx] = cost
                heapq.heappush(heap, (cost, idx))
        
        removed_count = 0
        start_time = time.time()
        
        while heap:
            if time.time() > deadline:
                break
            if time.time() - start_time > time_budget_per_restart:
                break
            
            cost, idx = heapq.heappop(heap)
            
            # Lazy deletion: check if idx is still in N and cost is still valid
            if not state.in_N[idx]:
                continue
            if cost_cache.get(idx, -1) != cost:
                # Recompute cost
                new_cost = 0
                for p, d in affects[idx]:
                    if state.not_in_N_count[p, d] == 0:
                        new_cost += 1
                cost_cache[idx] = new_cost
                heapq.heappush(heap, (new_cost, idx))
                continue
            
            # Check if removal is valid
            if not state.can_remove(idx):
                # Mark as invalid for now, don't push back (it might become valid later)
                cost_cache[idx] = -1
                continue
            
            # Remove the point
            state.remove(idx)
            removed_count += 1
            cost_cache[idx] = -1
            
            # Update costs for affected points
            # Points p in affects[idx] may have changed not_in_N_count, so their costs may change
            # For each p in affects[idx], recompute cost and push to heap
            updated_p = set()
            for p, d in affects[idx]:
                if state.in_N[p] and p not in updated_p:
                    updated_p.add(p)
                    new_cost = 0
                    for pp, dd in affects[p]:
                        if state.not_in_N_count[pp, dd] == 0:
                            new_cost += 1
                    cost_cache[p] = new_cost
                    heapq.heappush(heap, (new_cost, p))
        
        return state.in_N, removed_count
    
    # Main function
    start_time = time.time()
    deadline = start_time + 620  # Safety margin
    
    # Build initial N
    in_N = build_initial_in_N()
    initial_removed = n_points - np.sum(in_N)
    
    # Run greedy removal with multiple restarts
    best_in_N = in_N.copy()
    best_removed = initial_removed
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # For each restart, use a slightly different initial N or randomization
        # Here we just run the same greedy but with different heap tie-breaking
        in_N_restart = build_initial_in_N()
        
        # Add some randomization: randomly remove a few points first
        if restart > 0:
            candidates = [i for i in range(n_points) if in_N_restart[i]]
            n_rand = min(10, len(candidates) // 10)
            for _ in range(n_rand):
                if candidates:
                    idx = random.choice(candidates)
                    in_N_restart[idx] = False
                    candidates.remove(idx)
        
        removed_in_N, removed_count = greedy_removal(in_N_restart, deadline, time_budget_per_restart=100)
        total_removed = initial_removed + removed_count
        
        if total_removed > best_removed:
            best_removed = total_removed
            best_in_N = removed_in_N.copy()
    
    # Additional passes: try to remove more points from best_in_N
    state = RemovalState(best_in_N)
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        changed = True
        while changed:
            if time.time() > deadline - 5:
                break
            changed = False
            for idx in range(n_points):
                if not state.in_N[idx]:
                    continue
                if state.can_remove(idx):
                    state.remove(idx)
                    changed = True
    
    best_in_N = state.in_N
    best_removed = n_points - np.sum(best_in_N)
    
    # Build result
    N_indices = [i for i in range(n_points) if best_in_N[i]]
    
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for idx, i in enumerate(N_indices):
        x, y = idx_to_point(i)
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        result[idx, 0, 0] = xa
        result[idx, 0, 1] = xb
        result[idx, 1, 0] = ya
        result[idx, 1, 1] = yb
    
    return result
