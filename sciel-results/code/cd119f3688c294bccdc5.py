import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles
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
            norm_cycle = []
            for e in cycle:
                norm_e = (min(e[0], e[1]), max(e[0], e[1]))
                norm_cycle.append(norm_e)
            if len(set(norm_cycle)) == 4:
                cycle_idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(cycle_idx)

    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # State: dict edge -> color
    # We maintain a list of bad cycle indices for quick access? 
    # Or just recompute bad_count when needed?
    # Given N=20, recomputing all bad cycles is O(14535 * 4) ~ 60k ops.
    # In a repair loop, this might be called many times.
    # Incremental update is better.
    
    def get_bad_cycles(state):
        """Return list of indices of bad cycles."""
        bad_indices = []
        for i, cycle in enumerate(all_cycles):
            if all(e in state for e in cycle):
                colors = [state[e] for e in cycle]
                if len(set(colors)) < 4:
                    bad_indices.append(i)
        return bad_indices

    def count_bad_cycles(state):
        """Count bad cycles efficiently? Or just len(get_bad_cycles)?"""
        # For simplicity in the main loop, we might just use get_bad_cycles 
        # if the number of steps is low. 
        # But for expansion, we need to evaluate many candidates.
        # Let's use a helper to check if adding an edge creates bad cycles.
        
        pass

    def evaluate_color_for_edge(e, c, state):
        """
        Returns the number of bad cycles created by adding edge e with color c to state.
        Assumes e is NOT in state.
        """
        bad_count = 0
        # Only cycles containing e can become bad.
        for i in edge_to_cycles[e]:
            cycle = all_cycles[i]
            # Check if all other edges in cycle are present
            if all(ed in state for ed in cycle if ed != e):
                # Construct colors
                colors = []
                for ed in cycle:
                    if ed == e:
                        colors.append(c)
                    else:
                        colors.append(state[ed])
                if len(set(colors)) < 4:
                    bad_count += 1
        return bad_count

    def repair(state):
        """
        Remove edges until state is valid.
        Strategy: Find the edge in the most bad cycles and remove it.
        """
        while True:
            bad_indices = get_bad_cycles(state)
            if not bad_indices:
                break
            
            # Count frequency of edges in bad cycles
            edge_freq = defaultdict(int)
            for i in bad_indices:
                cycle = all_cycles[i]
                for e in cycle:
                    edge_freq[e] += 1
            
            # Pick edge with max frequency
            if not edge_freq:
                break # Should not happen if bad_indices not empty
                
            max_freq = max(edge_freq.values())
            candidates = [e for e, f in edge_freq.items() if f == max_freq]
            e_to_remove = random.choice(candidates)
            
            # Optional: Try to recolor instead of removing?
            # If we remove, we lose an edge. If we recolor, we might keep it.
            # Let's try recoloring first.
            # Find a color for e_to_remove that minimizes bad cycles.
            best_color = None
            min_bad_after = len(bad_indices) # Current bad count
            
            # We need to know which cycles are currently bad.
            # We have bad_indices.
            # If we change color of e_to_remove, only cycles containing e_to_remove change status.
            
            # Let's just remove for simplicity and robustness in this iteration.
            # Recoloring logic is complex to verify quickly without full state update.
            # However, removing is guaranteed to reduce bad cycles by at least 1 
            # (actually by the number of bad cycles containing that edge).
            
            del state[e_to_remove]
            
            if time.time() > deadline:
                break

    def expand(state):
        """
        Add missing edges if they don't create bad cycles.
        """
        improved = True
        while improved:
            improved = False
            # Shuffle missing edges to avoid bias
            missing = [e for e in all_edges if e not in state]
            random.shuffle(missing)
            
            for e in missing:
                # Try all colors
                best_c = None
                # We want a color that creates 0 bad cycles.
                # Since state is currently valid (0 bad cycles), 
                # adding e with color c creates bad cycles iff 
                # evaluate_color_for_edge(e, c, state) > 0.
                
                # We can just pick the first color that works.
                # Or shuffle colors.
                colors = list(range(num_colors))
                random.shuffle(colors)
                
                for c in colors:
                    if evaluate_color_for_edge(e, c, state) == 0:
                        state[e] = c
                        improved = True
                        break # Try next missing edge
                
                if time.time() > deadline:
                    return

    # Main Loop: Restart with different initial densities
    best_state = {}
    best_num_edges = 0
    
    # Try a few initial densities
    densities = [0.95, 0.9, 0.85, 0.8]
    
    for d in densities:
        if time.time() > deadline:
            break
            
        # Init state
        state = {}
        for e in all_edges:
            if random.random() < d:
                state[e] = random.randint(0, num_colors - 1)
        
        # Repair
        repair(state)
        
        # Expand
        expand(state)
        
        # Check validity
        if not get_bad_cycles(state):
            if len(state) > best_num_edges:
                best_num_edges = len(state)
                best_state = dict(state)
        
        # If we have time, do some random kicks and re-repair/expand
        # Kick: Remove 5 random edges
        if time.time() < deadline - 20:
            for _ in range(5):
                if state:
                    e = random.choice(list(state.keys()))
                    del state[e]
            repair(state)
            expand(state)
            if not get_bad_cycles(state):
                if len(state) > best_num_edges:
                    best_num_edges = len(state)
                    best_state = dict(state)

    # Final fallback if best_state is empty or invalid
    if not best_state:
        # Return a trivial valid graph (e.g., empty or single edge)
        # Empty graph is valid.
        pass

    result = []
    for (u, v), c in best_state.items():
        result.append([u, v, c])
        
    return result
