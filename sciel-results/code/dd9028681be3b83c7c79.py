"""Direct bootstrap for R(5,5): extend Paley(37) with smooth-objective SA search."""
import time
import random
import math

# EVOLVE-BLOCK-START

DEADLINE = time.time() + 270


def _build_paley37():
    """Return adjacency bitset list for Paley graph on 37 vertices (vertices 0..36)."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _find_k4_sets(adj, n):
    """Find all K4 cliques in graph with n vertices. Return list of 4-tuples."""
    k4_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j]
            for k in range(j + 1, n):
                if not (common & (1 << k)):
                    continue
                common2 = common & adj[k]
                for l in range(k + 1, n):
                    if common2 & (1 << l):
                        k4_list.append((i, j, k, l))
    return k4_list


def _find_ind4_sets(adj, n):
    """Find all independent 4-sets in graph with n vertices. Return list of 4-tuples."""
    ind4_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                continue
            non_common = ~(adj[i] | adj[j])
            for k in range(j + 1, n):
                if not (non_common & (1 << k)):
                    continue
                non_common2 = non_common & ~(adj[i] | adj[j] | adj[k])
                for l in range(k + 1, n):
                    if non_common2 & (1 << l):
                        ind4_list.append((i, j, k, l))
    return ind4_list


def _try_add_vertex_sa(adj, n, k4_list, ind4_list, time_limit=10.0):
    """
    Try to add vertex n (new index = n) by choosing edges to 0..n-1.
    Uses SA with smooth log-sum-exp objective.
    Returns new_adj or None if failed.
    """
    new_n = n + 1
    
    # Precompute: for each vertex, which K4s and ind4s it belongs to
    k4_by_vertex = [[] for _ in range(n)]
    for idx, k4 in enumerate(k4_list):
        for v in k4:
            k4_by_vertex[v].append(idx)
    
    ind4_by_vertex = [[] for _ in range(n)]
    for idx, ind4 in enumerate(ind4_list):
        for v in ind4:
            ind4_by_vertex[v].append(idx)
    
    # Precompute masks
    k4_masks = []
    for k4 in k4_list:
        mask = 0
        for v in k4:
            mask |= (1 << v)
        k4_masks.append(mask)
    
    ind4_masks = []
    for ind4 in ind4_list:
        mask = 0
        for v in ind4:
            mask |= (1 << v)
        ind4_masks.append(mask)
    
    # Smooth objective: sum of log(1 + exp(-beta * violation_indicator))
    # For K4: violation when all 4 bits are 1 -> penalty when product of bits is 1
    # For ind4: violation when all 4 bits are 0 -> penalty when product of (1-bits) is 1
    # We use a soft version: for each constraint, compute a "soft violation" score
    
    beta = 5.0  # sharpness of the soft penalty
    
    def soft_cost(state_mask):
        """Compute smooth cost using log-sum-exp style penalties."""
        cost = 0.0
        for mask in k4_masks:
            # Count how many of the 4 bits are set
            cnt = bin(state_mask & mask).count('1')
            # If all 4 set, full penalty; if 3 set, partial; etc.
            # Use: penalty = exp(-beta * (4 - cnt)) when cnt < 4, else 1
            # Actually simpler: penalty = 1 if cnt==4, else exp(-beta*(4-cnt))
            if cnt == 4:
                cost += 1.0
            else:
                cost += math.exp(-beta * (4 - cnt))
        for mask in ind4_masks:
            # For ind4: violation when all 4 bits are 0
            cnt = bin(state_mask & mask).count('1')
            # If cnt==0, full penalty; if cnt==1, partial; etc.
            if cnt == 0:
                cost += 1.0
            else:
                cost += math.exp(-beta * cnt)
        return cost
    
    def hard_violations(state_mask):
        """Count actual hard violations."""
        cost = 0
        for mask in k4_masks:
            if (state_mask & mask) == mask:
                cost += 1
        for mask in ind4_masks:
            if (state_mask & mask) == 0:
                cost += 1
        return cost
    
    # Greedy initialization: start with all 0s, then greedily set bits
    # to minimize soft cost
    def greedy_init():
        state = 0
        # Start with balanced random
        for i in range(n):
            if random.random() < 0.5:
                state |= (1 << i)
        
        # Local improvement: try flipping each bit, keep if improves
        improved = True
        while improved and time.time() < DEADLINE - 5:
            improved = False
            for v in range(n):
                old_cost = soft_cost(state)
                new_state = state ^ (1 << v)
                new_cost = soft_cost(new_state)
                if new_cost < old_cost - 1e-9:
                    state = new_state
                    improved = True
        return state, soft_cost(state)
    
    best_state = None
    best_hard = float('inf')
    best_soft = float('inf')
    
    start_time = time.time()
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize
        state, cur_soft = greedy_init()
        cur_hard = hard_violations(state)
        
        if cur_hard == 0:
            best_state = state
            break
        
        if cur_hard < best_hard or (cur_hard == best_hard and cur_soft < best_soft):
            best_hard = cur_hard
            best_soft = cur_soft
            best_state = state
        
        # SA with smooth objective
        temp = 2.0
        cooling = 0.997
        min_temp = 0.05
        steps_per_temp = 5
        
        while temp > min_temp:
            for _ in range(steps_per_temp):
                v = random.randint(0, n - 1)
                old_soft = soft_cost(state)
                new_state = state ^ (1 << v)
                new_soft = soft_cost(new_state)
                delta = new_soft - old_soft
                
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    state = new_state
                    cur_soft = new_soft
                    cur_hard = hard_violations(state)
                    if cur_hard == 0:
                        best_state = state
                        break
                    if cur_hard < best_hard or (cur_hard == best_hard and cur_soft < best_soft):
                        best_hard = cur_hard
                        best_soft = cur_soft
                        best_state = state
            
            if best_hard == 0:
                break
            temp *= cooling
        
        if best_hard == 0:
            break
    
    if best_hard == 0:
        new_adj = adj[:] + [0]
        for v in range(n):
            if best_state & (1 << v):
                new_adj[new_n] |= (1 << v)
                new_adj[v] |= (1 << new_n)
        return new_adj
    
    return None


def construct():
    adj = _build_paley37()
    n = 37
    
    best_n = 37
    best_adj = adj[:]
    
    # Precompute K4 and independent 4-sets for the base graph
    k4_list = _find_k4_sets(adj, n)
    ind4_list = _find_ind4_sets(adj, n)
    
    # Try to extend the graph by adding vertices one at a time
    for target_n in range(38, 43):
        if time.time() > DEADLINE:
            break
        
        # Recompute k4 and ind4 for current graph
        k4_list = _find_k4_sets(adj, n)
        ind4_list = _find_ind4_sets(adj, n)
        
        # Allocate time for this step
        time_remaining = DEADLINE - time.time()
        if time_remaining < 15:
            break
        
        # Try SA with a time limit
        new_adj = _try_add_vertex_sa(adj, n, k4_list, ind4_list, time_limit=min(10.0, time_remaining - 20))
        
        if new_adj is None and time.time() < DEADLINE - 20:
            # Retry with longer time
            new_adj = _try_add_vertex_sa(adj, n, k4_list, ind4_list, time_limit=min(20.0, DEADLINE - time.time() - 10))
        
        if new_adj is not None:
            adj = new_adj
            n = target_n
            best_n = n
            best_adj = adj[:]
        else:
            break
    
    # Build edge list from best_adj
    n_best = best_n
    edges = []
    for i in range(n_best):
        for j in range(i + 1, n_best):
            if best_adj[i] & (1 << j):
                edges.append((i, j))
    
    return n_best, edges

# EVOLVE-BLOCK-END
