import random
import itertools
import math

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def construct():
    # Precompute edge endpoints
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
            
    # Map edge index to (u, v)
    edge_to_uv = {idx: edges[idx] for idx in range(NUM_EDGES)}
    
    # Precompute constraints for each edge
    # For each edge e=(u,v), and for each pair (x,y) from remaining vertices,
    # we have 2 cycles involving e.
    constraints = [[] for _ in range(NUM_EDGES)]
    for e_idx in range(NUM_EDGES):
        u, v = edges[e_idx]
        others = [k for k in range(N) if k != u and k != v]
        for x, y in itertools.combinations(others, 2):
            # Cycle 1: u-v-x-y-u -> edges: e, (v,x), (x,y), (y,u)
            e1 = edge_idx(v, x)
            e2 = edge_idx(x, y)
            e3 = edge_idx(y, u)
            constraints[e_idx].append((e1, e2, e3))
            # Cycle 2: u-v-y-x-u -> edges: e, (v,y), (y,x), (x,u)
            e1 = edge_idx(v, y)
            e2 = edge_idx(y, x)
            e3 = edge_idx(x, u)
            constraints[e_idx].append((e1, e2, e3))

    # State
    colors = [random.randint(0, 119) for _ in range(NUM_EDGES)]
    
    # Helper to calculate violations for a single edge change
    # We need to check all cycles involving the changed edge.
    
    def get_violation_count(colors):
        # This is O(Edges * ConstraintsPerEdge)
        # But we can update incrementally.
        # For SA, we need Delta.
        pass

    # To make SA efficient, we pre-calculate the "potential" violations.
    # Actually, checking validity of a move is:
    # For each cycle involving the changed edge, check if the 4 colors are distinct.
    
    # Let's implement a function to check if a specific move (edge e, color c_new) is valid given current colors.
    # And count how many violations it introduces/removes.
    
    def check_move_validity(e_idx, new_color, colors):
        """
        Returns (is_valid, num_violations_caused)
        is_valid: True if the move results in 0 violations for all cycles involving e_idx.
        Note: We assume other cycles are already valid or we track global violations.
        For SA, we track global violations.
        This function calculates the DELTA in violations for cycles involving e_idx.
        """
        # We need to know the old color to calculate delta.
        # But for simplicity in the first step, let's just calculate total violations for the state.
        # That is too slow for SA if we do it every step.
        
        # Better: Maintain a global violation count.
        # When we change edge e, only cycles involving e change status.
        # Let's return the list of cycles involving e and their status.
        pass

    # Let's stick to a simpler, faster SA variant:
    # 1. Start with a valid configuration (e.g., all 0s).
    # 2. Only accept moves that keep the configuration VALID.
    # 3. If stuck (no valid move increases distinct count), allow "illegal" moves that decrease distinct count but might lead to a better basin?
    # No, the constraint is hard.
    
    # Alternative: 
    # Use the current program's logic but improve the search strategy.
    # The current program is a random walk.
    # Let's use **Simulated Annealing on the objective: -Distinct + Penalty*Violations**.
    
    # Re-initialize
    colors = [0] * NUM_EDGES
    freq = [0] * 120
    freq[0] = NUM_EDGES
    distinct = 1
    
    # Pre-compute cycles for fast violation checking
    # We have `constraints` list.
    
    def count_violations(colors):
        """
        Counts total number of 4-cycles with 4 distinct colors.
        """
        v_count = 0
        for e_idx in range(NUM_EDGES):
            c0 = colors[e_idx]
            for c1, c2, c3 in constraints[e_idx]:
                # Cycle edges: e_idx, c1, c2, c3
                if len({c0, colors[c1], colors[c2], colors[c3]}) == 4:
                    v_count += 1
        return v_count

    # Initial violations
    # Since all are 0, violations = 0.
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    best_colors = colors[:]
    best_distinct = 1
    
    # We need to be careful with time. 60s limit.
    # Number of steps ~ 50,000 to 100,000.
    
    for step in range(100000):
        e_idx = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e_idx]
        new_color = random.randint(0, 119)
        
        if new_color == old_color:
            continue
            
        # Calculate change in distinct count
        # freq[old_color] will decrease, freq[new_color] will increase
        old_freq = freq[old_color]
        new_freq = freq[new_color]
        
        delta_distinct = 0
        if old_freq == 1:
            delta_distinct -= 1
        if new_freq == 0:
            delta_distinct += 1
            
        # Calculate change in violations
        # Only cycles involving e_idx change.
        # We need to check each cycle involving e_idx.
        delta_violations = 0
        
        for c1, c2, c3 in constraints[e_idx]:
            # Old state
            old_set = {old_color, colors[c1], colors[c2], colors[c3]}
            old_v = 1 if len(old_set) == 4 else 0
            
            # New state
            new_set = {new_color, colors[c1], colors[c2], colors[c3]}
            new_v = 1 if len(new_set) == 4 else 0
            
            delta_violations += (new_v - old_v)
            
        # Energy change
        # Objective: Maximize Distinct, Minimize Violations
        # E = -Distinct + Lambda * Violations
        # Delta E = -Delta Distinct + Lambda * Delta Violations
        Lambda = 10.0
        delta_E = -delta_distinct + Lambda * delta_violations
        
        # Acceptance
        accepted = False
        if delta_E < 0:
            accepted = True
        else:
            if T > 0:
                prob = math.exp(-delta_E / T)
                if random.random() < prob:
                    accepted = True
            
        if accepted:
            # Apply move
            colors[e_idx] = new_color
            freq[old_color] -= 1
            freq[new_color] += 1
            distinct += delta_distinct
            
            if distinct > best_distinct:
                best_distinct = distinct
                best_colors = colors[:]
                
        # Cooling
        T *= cooling
        if T < T_min:
            T = T_min

    return best_colors
