import time
from random import Random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = t0 + 110.0  # Safety margin

    def gen_instance(n, d, seed):
        rng = Random(seed)
        degree = [0] * n
        edges = set()
        # Attempt to generate a random graph with max degree d
        # The original code used a specific generation method, we must replicate it exactly 
        # because the task specifies "bounded_degree_graph(n, d, seed)" and the verifier 
        # likely uses the same logic or the provided logic in the seed is the definition.
        # The seed program's gen_instance is:
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges):
        m = len(edges)
        # vertex -> list of edge indices
        v2e = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            v2e[u].append(i)
            v2e[v].append(i)
        
        conflict = [set() for _ in range(m)]
        
        # 1. Share a vertex
        for u, lst in v2e.items():
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    conflict[lst[a]].add(lst[b])
                    conflict[lst[b]].add(lst[a])
        
        # 2. Joined by an edge (distance 2)
        # For each edge (u, v), any edge incident to u (not (u,v)) conflicts with any edge incident to v (not (u,v))
        # Note: If (u,v) is in the graph, then an edge (u, x) and (v, y) are connected by the path x-u-v-y.
        # The definition of strong edge coloring: no two edges in the same color class share a vertex 
        # or are joined by an edge. 
        # "Joined by an edge" means there is an edge connecting a vertex of e1 to a vertex of e2.
        
        for i, (u, v) in enumerate(edges):
            lu = v2e[u]
            lv = v2e[v]
            for a in lu:
                if a == i:
                    continue
                for b in lv:
                    if b == i or b == a:
                        continue
                    conflict[a].add(b)
                    conflict[b].add(a)
                    
        return conflict

    def greedy_initial(edges, conflict):
        m = len(edges)
        # Simple First Fit
        colour = [-1] * m
        used = 0
        # Precompute forbidden vertices for each colour class? 
        # For simplicity, just check conflicts directly. O(m^2 * deg) might be slow for 80 nodes.
        # m is approx n*d/2. For n=80, d=6, m=240. m^2 = 57600. This is fine.
        
        for i in range(m):
            for c in range(used):
                # Check if i can be colour c
                ok = True
                for j in conflict[i]:
                    if colour[j] == c:
                        ok = False
                        break
                if ok:
                    colour[i] = c
                    break
            if colour[i] == -1:
                colour[i] = used
                used += 1
        return colour, used

    def count_conflicts(colour, conflict):
        m = len(colour)
        cnt = 0
        for i in range(m):
            ci = colour[i]
            for j in conflict[i]:
                if j > i: # count each pair once
                    if colour[j] == ci:
                        cnt += 1
        return cnt

    def local_search_reduce_palette(edges, conflict, init_colour, K_target, rng, time_limit):
        m = len(edges)
        # Initialize state
        # Map existing colours to 0..K_target-1
        # If init_colour has more colours, we need to merge them randomly or greedily into K_target
        state = [0] * m
        for i in range(m):
            c = init_colour[i]
            if c < K_target:
                state[i] = c
            else:
                state[i] = rng.randrange(K_target)
        
        current_conflicts = count_conflicts(state, conflict)
        best_state = state[:]
        best_conflicts = current_conflicts
        
        if best_conflicts == 0:
            return best_state, 0
            
        start = time.time()
        T = 1.0
        T_min = 0.01
        alpha = 0.995
        
        while time.time() - start < time_limit:
            if T < T_min:
                break
            # Pick a random edge
            i = rng.randrange(m)
            old_c = state[i]
            new_c = rng.randrange(K_target)
            if new_c == old_c:
                continue
            
            # Calculate delta in conflicts
            # Remove i from old_c: conflicts with j where colour[j]==old_c and j in conflict[i]
            # Add i to new_c: conflicts with j where colour[j]==new_c and j in conflict[i]
            
            delta = 0
            for j in conflict[i]:
                cj = state[j]
                if cj == old_c:
                    delta -= 1 # removing a conflict
                if cj == new_c:
                    delta += 1 # adding a conflict
            
            state[i] = new_c
            current_conflicts += delta
            
            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                best_state = state[:]
                if best_conflicts == 0:
                    return best_state, 0
                
            # Acceptance
            if delta < 0:
                pass # Always accept improvements
            else:
                # Metropolis criterion
                # We are minimizing conflicts.
                # P(accept) = exp(-delta / T)
                if rng.random() < (2.718281828 ** (-delta / T)):
                    pass
                else:
                    # Reject
                    state[i] = old_c
                    current_conflicts -= delta
            
            T *= alpha
            
        return best_state, best_conflicts

    def solve_instance(n, d, seed):
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        conflict = build_conflict(edges)
        
        # 1. Get initial valid colouring
        init_colour, K_init = greedy_initial(edges, conflict)
        
        # 2. Try to reduce palette size
        K_current = K_init
        best_colour = init_colour
        rng = Random(seed + 1000)
        
        # Time budget per instance
        inst_start = time.time()
        
        # We try to reduce K by 1 iteratively
        # If we fail, we might retry with different random seeds/initializations
        attempts_per_K = 3
        
        while K_current > 1:
            K_target = K_current - 1
            success = False
            for attempt in range(attempts_per_K):
                if time.time() > DEADLINE - 5.0:
                    break
                # Time limit for this local search attempt
                # Allocate remaining time divided by expected steps? 
                # Let's give 1-2 seconds per attempt
                time_limit = 1.5 
                if time.time() + time_limit > DEADLINE - 2.0:
                    time_limit = max(0.1, DEADLINE - 2.0 - time.time())
                    
                new_colour, conf = local_search_reduce_palette(
                    edges, conflict, best_colour, K_target, rng, time_limit
                )
                
                if conf == 0:
                    # Success! We found a valid colouring with K_target colours
                    K_current = K_target
                    best_colour = new_colour
                    success = True
                    break
            if not success:
                # Failed to reduce K by 1. 
                # We could try harder, but for now, we stop reducing.
                # A more advanced approach would be to keep searching for K_target
                # with more time, but we have 3 instances.
                break
                
        return best_colour

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        c = solve_instance(*spec)
        out.append(c)
        
    return out
