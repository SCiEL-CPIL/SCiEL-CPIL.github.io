import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s kill limit
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles and edge-to-cycle incidence
    all_cycles = []
    edge_to_cycles = defaultdict(list)
    
    for u, v, w, x in itertools.combinations(range(n), 4):
        # Three distinct 4-cycles on 4 vertices
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
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    def can_add_edge(e, c, edge_colors):
        """Check if adding edge e with color c keeps all 4-cycles containing e rainbow."""
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            other_edges = [oe for oe in cycle if oe != e]
            if not all(oe in edge_colors for oe in other_edges):
                continue
            other_colors = [edge_colors[oe] for oe in other_edges]
            if c in other_colors:
                return False
            if len(set(other_colors)) < 3:
                return False
        return True
    
    def find_valid_color(e, edge_colors):
        """Find a valid color for edge e, or None. Prioritize colors with fewer conflicts."""
        colors = list(range(num_colors))
        # Sort colors by how often they appear in cycles containing e (fewer = better)
        color_pressure = defaultdict(int)
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            for oe in cycle:
                if oe != e and oe in edge_colors:
                    color_pressure[edge_colors[oe]] += 1
        colors.sort(key=lambda c: color_pressure.get(c, 0))
        random.shuffle([c for c in colors if color_pressure.get(c, 0) == 0])
        for c in colors:
            if can_add_edge(e, c, edge_colors):
                return c
        return None
    
    def get_conflict_edges(edge_colors):
        conflict = set()
        for cycle in all_cycles:
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    for e in cycle:
                        conflict.add(e)
        return conflict
    
    # Phase 1: Structured initial solution using a near-complete graph
    # Use a systematic coloring: color = (u * 7 + v * 3) % num_colors for edge (u,v)
    # This tends to avoid color conflicts on 4-cycles
    edge_colors = {}
    for u in range(n):
        for v in range(u + 1, n):
            c = (u * 7 + v * 3) % num_colors
            edge_colors[(u, v)] = c
    
    # Repair: remove edges causing conflicts, prioritizing high-degree edges
    max_repair = 2000
    for _ in range(max_repair):
        if time.time() > deadline:
            break
        conflicts = get_conflict_edges(edge_colors)
        if not conflicts:
            break
        # Remove edge in most conflict cycles
        e_to_remove = max(conflicts, key=lambda e: len(edge_to_cycles[e]))
        del edge_colors[e_to_remove]
    
    # Phase 2: Local search with Lagrangian-inspired dynamic weighting
    best_count = len(edge_colors)
    best_state = dict(edge_colors)
    
    def save_if_better():
        nonlocal best_count, best_state
        if len(edge_colors) > best_count:
            best_count = len(edge_colors)
            best_state = dict(edge_colors)
    
    # Lagrangian multipliers: track conflict pressure per edge
    # High multiplier = edge is in many conflicting cycles = deprioritize
    edge_multiplier = defaultdict(float)
    
    def update_multipliers(edge_colors, learning_rate=0.1):
        """Update Lagrangian multipliers based on current conflict structure."""
        # Decay existing multipliers
        for e in list(edge_multiplier.keys()):
            edge_multiplier[e] *= (1 - learning_rate)
        
        # Compute conflict counts
        conflict_count = defaultdict(int)
        for cycle in all_cycles:
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    for e in cycle:
                        conflict_count[e] += 1
        
        # Update multipliers: increase for edges in conflicts
        for e, count in conflict_count.items():
            edge_multiplier[e] += learning_rate * count
        
        # Clean up small multipliers
        for e in list(edge_multiplier.keys()):
            if edge_multiplier[e] < 0.01:
                del edge_multiplier[e]
    
    def edge_priority(e, edge_colors):
        """Compute priority for adding edge e. Higher = more desirable."""
        if e in edge_colors:
            return -1
        # Base score: number of valid colors available
        valid_count = 0
        for c in range(num_colors):
            if can_add_edge(e, c, edge_colors):
                valid_count += 1
                if valid_count >= 5:  # Early exit
                    break
        # Penalty from Lagrangian multiplier
        penalty = edge_multiplier.get(e, 0)
        # Bonus for edges in many cycles (more potential value)
        cycle_bonus = len(edge_to_cycles[e]) * 0.5
        return valid_count * 2 - penalty + cycle_bonus
    
    # Initial multiplier update
    update_multipliers(edge_colors)
    
    # Iterative improvement loop with Lagrangian-guided moves
    while time.time() < deadline:
        missing = [e for e in all_edges if e not in edge_colors]
        
        # Periodically update multipliers
        if random.random() < 0.1:
            update_multipliers(edge_colors)
        
        # Move type 1: Add a missing edge (prioritized by Lagrangian-informed score)
        if missing:
            # Score missing edges using dynamic priority
            scored_missing = []
            for e in missing:
                score = edge_priority(e, edge_colors)
                if score > 0:
                    scored_missing.append((score, e))
            
            if scored_missing:
                scored_missing.sort(reverse=True)
                # Pick from top-scoring edges with some randomness
                top_k = min(15, len(scored_missing))
                top_edges = [e for _, e in scored_missing[:top_k]]
                e_add = random.choice(top_edges)
                c = find_valid_color(e_add, edge_colors)
                if c is not None:
                    edge_colors[e_add] = c
                    save_if_better()
                    continue
        
        # Move type 2: Recolor an existing edge (fast O(k) check)
        if edge_colors:
            # Prioritize recoloring edges with high multipliers
            weighted_edges = list(edge_colors.keys())
            weights = [1.0 + edge_multiplier.get(e, 0) * 0.5 for e in weighted_edges]
            total_w = sum(weights)
            if total_w > 0:
                r = random.random() * total_w
                cum = 0
                e_recolor = weighted_edges[-1]
                for i, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        e_recolor = weighted_edges[i]
                        break
            else:
                e_recolor = random.choice(weighted_edges)
            
            old_c = edge_colors[e_recolor]
            # Try a few random new colors
            for _ in range(3):
                new_c = random.randint(0, num_colors - 1)
                if new_c != old_c and can_add_edge(e_recolor, new_c, edge_colors):
                    edge_colors[e_recolor] = new_c
                    break
        
        # Move type 3: Swap - remove one edge, add another
        if edge_colors and missing and random.random() < 0.4:
            # Remove edge with highest multiplier (most problematic)
            sorted_by_mult = sorted(edge_colors.keys(), key=lambda e: -edge_multiplier.get(e, 0))
            e_remove = sorted_by_mult[0] if sorted_by_mult else random.choice(list(edge_colors.keys()))
            
            # Add edge with highest priority
            scored_missing = []
            for e in missing:
                score = edge_priority(e, edge_colors)
                if score > 0:
                    scored_missing.append((score, e))
            
            if scored_missing:
                scored_missing.sort(reverse=True)
                e_add = scored_missing[0][1]
            else:
                e_add = random.choice(missing)
            
            old_c = edge_colors[e_remove]
            del edge_colors[e_remove]
            c_add = find_valid_color(e_add, edge_colors)
            if c_add is not None:
                edge_colors[e_add] = c_add
                save_if_better()
                # Try to re-add e_remove
                c_rem = find_valid_color(e_remove, edge_colors)
                if c_rem is not None:
                    edge_colors[e_remove] = c_rem
                    save_if_better()
            else:
                # Revert
                c_rem = find_valid_color(e_remove, edge_colors)
                if c_rem is not None:
                    edge_colors[e_remove] = c_rem
                else:
                    edge_colors[e_remove] = old_c
        
        # Move type 4: Double swap - remove 2, add 2
        if random.random() < 0.15 and len(edge_colors) >= 2 and len(missing) >= 2:
            # Remove two edges with highest multipliers
            sorted_by_mult = sorted(edge_colors.keys(), key=lambda e: -edge_multiplier.get(e, 0))
            e1, e2 = sorted_by_mult[0], sorted_by_mult[1]
            
            # Add two edges with highest priorities
            scored_missing = []
            for e in missing:
                score = edge_priority(e, edge_colors)
                if score > 0:
                    scored_missing.append((score, e))
            
            if len(scored_missing) >= 2:
                scored_missing.sort(reverse=True)
                e3, e4 = scored_missing[0][1], scored_missing[1][1]
            else:
                e3, e4 = random.sample(missing, 2)
            
            c1 = edge_colors[e1]
            c2 = edge_colors[e2]
            del edge_colors[e1]
            del edge_colors[e2]
            c3 = find_valid_color(e3, edge_colors)
            c4 = find_valid_color(e4, edge_colors)
            if c3 is not None and c4 is not None:
                edge_colors[e3] = c3
                edge_colors[e4] = c4
                save_if_better()
                # Try to re-add e1, e2
                c1_new = find_valid_color(e1, edge_colors)
                c2_new = find_valid_color(e2, edge_colors)
                if c1_new is not None:
                    edge_colors[e1] = c1_new
                if c2_new is not None:
                    edge_colors[e2] = c2_new
                save_if_better()
            else:
                # Revert
                edge_colors[e1] = c1
                edge_colors[e2] = c2
    
    # Use the best state found
    edge_colors = best_state
    
    # Final cleanup: ensure validity
    for _ in range(500):
        if time.time() > deadline:
            break
        conflicts = get_conflict_edges(edge_colors)
        if not conflicts:
            break
        e_to_remove = max(conflicts, key=lambda e: len(edge_to_cycles[e]))
        del edge_colors[e_to_remove]
    
    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
    return result
