import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s kill limit
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles and their edge sets
    all_cycles = []
    edge_to_cycles = defaultdict(list)
    
    for u, v, w, x in itertools.combinations(range(n), 4):
        # Three distinct 4-cycles on 4 vertices (edge sets)
        cycles = [
            [(u, v), (v, w), (w, x), (x, u)],
            [(u, v), (v, x), (x, w), (w, u)],
            [(u, w), (w, v), (v, x), (x, u)],
        ]
        for cycle in cycles:
            norm_cycle = tuple(sorted((min(e), max(e)) for e in cycle))
            if len(set(norm_cycle)) == 4:
                idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(idx)
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def can_add_edge(e, c, edge_colors):
        """Check if adding edge e with color c keeps all 4-cycles containing e rainbow."""
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            # Check if all other edges in cycle are present
            all_present = True
            other_colors = []
            for oe in cycle:
                if oe == e:
                    continue
                if oe not in edge_colors:
                    all_present = False
                    break
                other_colors.append(edge_colors[oe])
            if not all_present:
                continue
            if c in other_colors:
                return False
            if len(set(other_colors)) < 3:
                return False
        return True
    
    def find_valid_color(e, edge_colors):
        """Find a valid color for edge e, or None."""
        # Collect colors used in cycles containing e
        used_in_cycles = set()
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            for oe in cycle:
                if oe != e and oe in edge_colors:
                    used_in_cycles.add(edge_colors[oe])
        
        # Try unused colors first, then others
        unused = [c for c in range(num_colors) if c not in used_in_cycles]
        used = [c for c in range(num_colors) if c in used_in_cycles]
        # Deterministic order for reproducibility and systematic search
        for c in unused + used:
            if can_add_edge(e, c, edge_colors):
                return c
        return None
    
    # Phase 1: Construct a high-quality initial solution using greedy approach
    edge_colors = {}
    
    # Greedy construction with randomization for diversity
    random.shuffle(all_edges)
    for e in all_edges:
        if time.time() > deadline:
            break
        c = find_valid_color(e, edge_colors)
        if c is not None:
            edge_colors[e] = c
    
    best_count = len(edge_colors)
    best_state = dict(edge_colors)
    
    def save_if_better():
        nonlocal best_count, best_state
        if len(edge_colors) > best_count:
            best_count = len(count for count in [len(edge_colors)])
            best_state = dict(edge_colors)
    
    # Phase 2: Lagrangian-inspired weight-guided local search
    # We assign weights to edges based on how many valid colors they have
    # Edges with fewer options get higher priority
    
    def compute_edge_weights(edge_colors):
        """Compute weights for missing edges based on constraint tightness."""
        weights = {}
        missing_set = set(e for e in all_edges if e not in edge_colors)
        for e in missing_set:
            valid_colors = 0
            for c in range(num_colors):
                if can_add_edge(e, c, edge_colors):
                    valid_colors += 1
            # Higher weight = more constrained = more urgent to place
            weights[e] = num_colors - valid_colors
        return weights
    
    def add_edge_by_weight(edge_colors, missing_set, weights):
        """Add the most constrained edge first."""
        if not missing_set:
            return False
        # Find edge with highest weight
        best_e = None
        best_w = -1
        for e in missing_set:
            w = weights.get(e, 0)
            if w > best_w:
                best_w = w
                best_e = e
        if best_e is None:
            return False
        c = find_valid_color(best_e, edge_colors)
        if c is not None:
            edge_colors[best_e] = c
            missing_set.discard(best_e)
            return True
        return False
    
    # Iterative improvement loop with Lagrangian relaxation
    while time.time() < deadline:
        improved = False
        
        # Compute weights for missing edges
        missing_set = set(e for e in all_edges if e not in edge_colors)
        if not missing_set:
            break
        
        weights = compute_edge_weights(edge_colors)
        
        # Move type 1: Add missing edges in order of constraint tightness
        for _ in range(min(20, len(missing_set))):
            if time.time() > deadline:
                break
            if add_edge_by_weight(edge_colors, missing_set, weights):
                improved = True
                save_if_better()
                # Recompute weights after adding an edge
                missing_set = set(e for e in all_edges if e not in edge_colors)
                if not missing_set:
                    break
                weights = compute_edge_weights(edge_colors)
        
        # Move type 2: Recolor edges to free up constraints
        # Focus on edges that are in many conflicts
        if not improved:
            # Try recoloring edges that appear in many cycles
            edge_cycle_count = {}
            for e in edge_colors:
                edge_cycle_count[e] = len(edge_to_cycles[e])
            
            # Sort by cycle count (most constrained first)
            sorted_edges = sorted(edge_colors.keys(), key=lambda e: -edge_cycle_count[e])
            
            for e_recolor in sorted_edges[:30]:  # Limit to most constrained
                if time.time() > deadline:
                    break
                old_c = edge_colors[e_recolor]
                # Try all colors
                for new_c in range(num_colors):
                    if new_c == old_c:
                        continue
                    # Temporarily remove
                    del edge_colors[e_recolor]
                    if can_add_edge(e_recolor, new_c, edge_colors):
                        # Check if this helps add any missing edges
                        edge_colors[e_recolor] = new_c
                        # Quick check: can we add any missing edge now?
                        missing_set = set(e for e in all_edges if e not in edge_colors)
                        weights = compute_edge_weights(edge_colors)
                        if add_edge_by_weight(edge_colors, missing_set, weights):
                            improved = True
                            save_if_better()
                            break
                        else:
                            # Revert if no immediate benefit
                            edge_colors[e_recolor] = old_c
                    else:
                        # Revert
                        edge_colors[e_recolor] = old_c
        
        # Move type 3: Random perturbation to escape local optima
        if not improved:
            if random.random() < 0.15:
                # Remove a few random edges to allow re-optimization
                for _ in range(3):
                    if not edge_colors:
                        break
                    e = random.choice(tuple(edge_colors.keys()))
                    missing_set = set(e for e in all_edges if e not in edge_colors)
                    missing_set.add(e)
                    del edge_colors[e]
            elif random.random() < 0.05:
                # Larger perturbation
                for _ in range(8):
                    if not edge_colors:
                        break
                    e = random.choice(tuple(edge_colors.keys()))
                    missing_set = set(e for e in all_edges if e not in edge_colors)
                    missing_set.add(e)
                    del edge_colors[e]
    
    # Use the best state found
    edge_colors = best_state
    
    # Final cleanup: ensure validity
    for _ in range(300):
        if time.time() > deadline:
            break
        # Check for conflicts
        conflict_found = False
        for cycle in all_cycles:
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    # Remove an edge
                    e_to_remove = cycle[0]
                    del edge_colors[e_to_remove]
                    conflict_found = True
                    break
        if not conflict_found:
            break
    
    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
    return result
