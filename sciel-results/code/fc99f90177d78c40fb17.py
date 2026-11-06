import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices u, v, w, x.
    # The cycles are (u,v,w,x), (u,v,x,w), (u,w,v,x) in terms of edge sets.
    # Edges are stored as (min, max).
    
    all_cycles = []
    edge_to_cycles = defaultdict(list)
    
    vertices = list(range(n))
    for comb in itertools.combinations(vertices, 4):
        u, v, w, x = comb
        # Cycle 1: u-v-w-x-u
        c1 = [(u,v), (v,w), (w,x), (x,u)]
        # Cycle 2: u-v-x-w-u
        c2 = [(u,v), (v,x), (x,w), (w,u)]
        # Cycle 3: u-w-v-x-u
        c3 = [(u,w), (w,v), (v,x), (x,u)]
        
        for cycle in [c1, c2, c3]:
            # Normalize edges
            norm_cycle = []
            for e in cycle:
                norm_e = (min(e[0], e[1]), max(e[0], e[1]))
                norm_cycle.append(norm_e)
            # Check for duplicates in the cycle itself (shouldn't happen with distinct vertices)
            if len(set(norm_cycle)) == 4:
                cycle_idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(cycle_idx)

    # State
    # edge_colors: dict (u,v) -> color
    # valid_edges: set of (u,v)
    
    # We will use a Max-SAT-like local search approach.
    # Variables: For each edge e, we have a binary variable "present" and a color variable.
    # Objective: Maximize sum(present_e) - penalty * sum(bad_cycles)
    # Actually, simpler: 
    # 1. Start with a dense random assignment (all edges present, random colors).
    # 2. Define a cost function: cost = -num_edges + lambda * num_bad_cycles.
    # 3. Local search: 
    #    - Flip color of a random edge.
    #    - Remove a random edge.
    #    - Add a random edge (if not present) with random color.
    #    - If cost improves, accept. Else, accept with probability exp(-delta/T) (simulated annealing).
    
    # However, checking cost change requires efficient updates.
    # Let's maintain:
    # - edge_colors: dict
    # - bad_cycle_count: int
    # - For each cycle, we can check if it's bad quickly.
    # - When an edge changes, we only need to re-check cycles containing that edge.
    
    # Let's try a simpler heuristic first that worked well in inspiration 1:
    # Start dense (p=0.9), repair by removing edges in bad cycles.
    # Then try to re-add them.
    
    # But the directive suggests a structural change: Max-SAT / Global Search.
    
    # Let's implement a robust Simulated Annealing on the space of edge sets and colorings.
    
    def random_state():
        """Generate a random initial state."""
        state = {}
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.8:  # Start dense
                    state[(u,v)] = random.randint(0, num_colors - 1)
        return state
    
    def count_bad_cycles(state):
        """Count the number of non-rainbow 4-cycles in the given state."""
        bad = 0
        for cycle in all_cycles:
            if all(e in state for e in cycle):
                colors = [state[e] for e in cycle]
                if len(set(colors)) < 4:
                    bad += 1
        return bad
    
    def get_bad_cycles(state):
        """Return list of indices of bad cycles."""
        bad_indices = []
        for i, cycle in enumerate(all_cycles):
            if all(e in state for e in cycle):
                colors = [state[e] for e in cycle]
                if len(set(colors)) < 4:
                    bad_indices.append(i)
        return bad_indices
    
    # We will maintain a list of bad cycle indices to speed up checks?
    # Actually, with n=20, number of 4-cycles is C(20,4)*3 = 4845*3 = 14535.
    # Checking all of them every step is too slow for SA.
    # We need incremental updates.
    
    # Let's maintain:
    # - state: dict edge -> color
    # - bad_count: int
    # - For each edge, we don't store cycle status, we just re-evaluate affected cycles.
    
    # To make this fast, we need to know which cycles are currently bad.
    # Let's store a boolean array is_bad_cycle of length len(all_cycles).
    # And bad_count.
    
    # Initialize state
    state = random_state()
    
    is_bad = [False] * len(all_cycles)
    bad_count = 0
    
    # Initial evaluation
    for i, cycle in enumerate(all_cycles):
        if all(e in state for e in cycle):
            colors = [state[e] for e in cycle]
            if len(set(colors)) < 4:
                is_bad[i] = True
                bad_count += 1
    
    def evaluate_cycle(i, state):
        """Return True if cycle i is bad in state."""
        cycle = all_cycles[i]
        if not all(e in state for e in cycle):
            return False
        colors = [state[e] for e in cycle]
        return len(set(colors)) < 4
    
    def update_cycle_status(i, state, is_bad, bad_count):
        """Update is_bad[i] and bad_count based on current state."""
        new_bad = evaluate_cycle(i, state)
        if new_bad and not is_bad[i]:
            is_bad[i] = True
            bad_count += 1
        elif not new_bad and is_bad[i]:
            is_bad[i] = False
            bad_count -= 1
        return bad_count
    
    # Objective function for SA:
    # We want to maximize edges, subject to bad_count == 0.
    # Let's define score = num_edges * 1000 - bad_count * 1000000
    # Or simpler: minimize cost = bad_count * 10000 - num_edges
    
    def get_cost(state, bad_count):
        return bad_count * 10000 - len(state)
    
    current_cost = get_cost(state, bad_count)
    best_state = dict(state)
    best_cost = current_cost
    best_bad_count = bad_count
    best_num_edges = len(state)
    
    T = 100.0
    T_min = 1.0
    cooling = 0.995
    
    # Precompute list of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    while time.time() < deadline:
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Change color of an existing edge
            if not state:
                continue
            e = random.choice(list(state.keys()))
            old_color = state[e]
            new_color = random.randint(0, num_colors - 1)
            if new_color == old_color:
                continue
            
            # Update state
            state[e] = new_color
            
            # Update affected cycles
            old_bad_count = bad_count
            for i in edge_to_cycles[e]:
                update_cycle_status(i, state, is_bad, bad_count)
            
            new_cost = get_cost(state, bad_count)
            delta = new_cost - current_cost
            
            if delta <= 0 or random.random() < pow(2.718, -delta / T):
                # Accept
                current_cost = new_cost
                # Update best
                if bad_count == 0 and len(state) > best_num_edges:
                    best_state = dict(state)
                    best_cost = current_cost
                    best_bad_count = bad_count
                    best_num_edges = len(state)
            else:
                # Reject
                state[e] = old_color
                for i in edge_to_cycles[e]:
                    update_cycle_status(i, state, is_bad, bad_count)
                # bad_count should be back to old_bad_count, but let's trust the update
            
        elif move_type < 0.8:
            # Remove an existing edge
            if not state:
                continue
            e = random.choice(list(state.keys()))
            old_color = state[e]
            del state[e]
            
            old_bad_count = bad_count
            for i in edge_to_cycles[e]:
                update_cycle_status(i, state, is_bad, bad_count)
            
            new_cost = get_cost(state, bad_count)
            delta = new_cost - current_cost
            
            if delta <= 0 or random.random() < pow(2.718, -delta / T):
                # Accept
                current_cost = new_cost
                if bad_count == 0 and len(state) > best_num_edges:
                    best_state = dict(state)
                    best_cost = current_cost
                    best_bad_count = bad_count
                    best_num_edges = len(state)
            else:
                # Reject
                state[e] = old_color
                for i in edge_to_cycles[e]:
                    update_cycle_status(i, state, is_bad, bad_count)
            
        else:
            # Add a missing edge
            # Find missing edges
            missing = [e for e in all_edges if e not in state]
            if not missing:
                continue
            e = random.choice(missing)
            new_color = random.randint(0, num_colors - 1)
            state[e] = new_color
            
            old_bad_count = bad_count
            for i in edge_to_cycles[e]:
                update_cycle_status(i, state, is_bad, bad_count)
            
            new_cost = get_cost(state, bad_count)
            delta = new_cost - current_cost
            
            if delta <= 0 or random.random() < pow(2.718, -delta / T):
                # Accept
                current_cost = new_cost
                if bad_count == 0 and len(state) > best_num_edges:
                    best_state = dict(state)
                    best_cost = current_cost
                    best_bad_count = bad_count
                    best_num_edges = len(state)
            else:
                # Reject
                del state[e]
                for i in edge_to_cycles[e]:
                    update_cycle_status(i, state, is_bad, bad_count)
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Optional: If we have a valid solution (bad_count == 0), we can try to aggressively add edges
        # by restarting with higher T or just continuing SA.
        
        # If we are stuck at bad_count > 0 for a long time, maybe restart?
        # For now, let's just run SA.
        
    # Final verification and cleanup
    # Ensure best_state is valid
    # The problem requires EVERY 4-cycle to be rainbow.
    # If best_bad_count > 0, the solution is invalid and scores 0.
    # We should only return a valid solution.
    # If we don't have a valid solution, we should return the best valid one found.
    # If we never found a valid one, we return an empty graph or a small valid graph?
    # The task says "Maximize the number of edges". An empty graph is valid (0 edges).
    # But we want to maximize.
    
    # Let's do a final greedy repair on best_state if it's not valid?
    # No, best_state is the best *valid* state found.
    # If best_bad_count > 0, it means we never found a valid state?
    # Wait, my update logic updates best_state only if bad_count == 0.
    # So if best_bad_count > 0, it means we initialized best_bad_count to initial bad_count,
    # and never found a valid state.
    # In that case, best_state is just the initial state, which might be invalid.
    # We must return a VALID state.
    
    # So, if best_bad_count > 0, we need to find a valid state.
    # Let's run a simple greedy repair on the current state or best_state.
    
    def greedy_repair(state):
        """Remove edges until state is valid."""
        state = dict(state)
        while True:
            bad_indices = get_bad_cycles(state)
            if not bad_indices:
                break
            # Pick a random bad cycle
            i = random.choice(bad_indices)
            cycle = all_cycles[i]
            # Remove a random edge from this cycle
            e = random.choice(cycle)
            if e in state:
                del state[e]
        return state
    
    if best_bad_count > 0:
        # We need a valid solution.
        # Let's try to repair the current state.
        repaired = greedy_repair(state)
        if len(repaired) > len(best_state): # best_state might be invalid or smaller
            best_state = repaired
            best_bad_count = 0
            best_num_edges = len(repaired)
        else:
            # If best_state was invalid, repair it?
            # Actually, if best_bad_count > 0, best_state is not necessarily the one we should repair.
            # Let's just repair the current state and see if it's better than any valid state we might have?
            # We didn't track any valid state if best_bad_count > 0.
            # So repaired is our only valid candidate.
            best_state = repaired
            best_bad_count = 0
            best_num_edges = len(repaired)
    
    # Final check: is best_state valid?
    # It should be, by construction of greedy_repair or the SA acceptance criteria.
    # Let's double check.
    final_bad = count_bad_cycles(best_state)
    if final_bad != 0:
        # Emergency repair
        best_state = greedy_repair(best_state)
    
    result = []
    for (u, v), c in best_state.items():
        result.append([u, v, c])
        
    return result
