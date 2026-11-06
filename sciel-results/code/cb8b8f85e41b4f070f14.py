import time
import sys
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Build edge index mapping
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    # Precompute all 4-cycles (3 per 4-vertex subset)
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        # Cycle 2: a-b-d-c-a
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        # Cycle 3: a-c-b-d-a
        cycles.append(((a,c), (c,b), (b,d), (d,a)))

    # Convert to edge indices
    cycle_indices = []
    for c in cycles:
        cycle_indices.append(tuple(edge_idx[u][v] for u, v in c))

    # Build edge_to_cycles mapping
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, c_idx in enumerate(cycle_indices):
        for e_idx in c_idx:
            edge_to_cycles[e_idx].append(i)

    # For each cycle, track how many edges have been assigned
    # and their colors
    def check_cycle_valid(cycle_id, colors):
        """Check if a cycle has 4 distinct colors"""
        ci = cycle_indices[cycle_id]
        c0 = colors[ci[0]]
        c1 = colors[ci[1]]
        c2 = colors[ci[2]]
        c3 = colors[ci[3]]
        # All 4 must be distinct for violation
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
        return True

    # Backtracking solver with MRV (Minimum Remaining Values) and diversity-first heuristic
    def solve():
        colors = [-1] * NUM_EDGES  # -1 means unassigned
        # Track for each cycle: how many edges are assigned, and what colors
        # cycle_state[c] = list of (edge_idx, color) for assigned edges in cycle c
        cycle_state = [[] for _ in cycle_indices]
        
        # Track which colors are used
        used_colors = set()
        
        # For efficiency, track for each edge how many cycles it's in
        edge_cycle_count = [len(edge_to_cycles[e]) for e in range(NUM_EDGES)]
        
        # MRV: pick the edge with fewest valid color options
        # Diversity: prefer unused colors first
        
        def get_valid_colors(edge):
            """Get set of colors that don't violate constraints if assigned to edge"""
            valid = []
            for c in range(MAX_COLORS):
                # Check all cycles containing this edge
                conflict = False
                for cid in edge_to_cycles[edge]:
                    ci = cycle_indices[cid]
                    # Get colors of other 3 edges in this cycle
                    other_colors = []
                    for e2 in ci:
                        if e2 != edge:
                            if colors[e2] == -1:
                                # Unassigned - can't determine conflict yet
                                # But we need to check if assigning c would create a rainbow
                                # with the other 3 (some may be unassigned)
                                # For now, assume it could be valid
                                other_colors.append(None)
                            else:
                                other_colors.append(colors[e2])
                    
                    # If all other 3 are assigned, check for rainbow
                    if all(oc is not None for oc in other_colors):
                        # Check if c + other 3 creates 4 distinct colors
                        all_colors = [c] + other_colors
                        if len(set(all_colors)) == 4:
                            conflict = True
                            break
                    else:
                        # Some unassigned - can't determine conflict
                        # Assume valid for now (optimistic)
                        pass
                
                if not conflict:
                    valid.append(c)
            
            return valid

        # Simpler approach: just try colors in order, with diversity bias
        def backtrack(depth, num_used):
            if time.time() > deadline:
                return False, num_used
            
            # Find next unassigned edge (MRV: fewest valid options)
            best_edge = -1
            best_valid = None
            min_count = float('inf')
            
            for e in range(NUM_EDGES):
                if colors[e] == -1:
                    # Count how many cycles this edge is in that are "close" to complete
                    # i.e., have 3 other edges assigned
                    close_count = 0
                    for cid in edge_to_cycles[e]:
                        ci = cycle_indices[cid]
                        assigned = sum(1 for e2 in ci if e2 != e and colors[e2] != -1)
                        if assigned == 3:
                            close_count += 1
                    
                    # MRV: prefer edges that are more constrained
                    # But also consider diversity: prefer edges that can use new colors
                    constraint_score = close_count
                    
                    if constraint_score < min_count:
                        min_count = constraint_score
                        best_edge = e
                        best_valid = None  # Will compute later
            
            if best_edge == -1:
                # All edges assigned
                return True, num_used
            
            e = best_edge
            
            # Get valid colors for this edge
            # Try unused colors first (diversity), then used colors
            unused = [c for c in range(MAX_COLORS) if c not in used_colors]
            used = [c for c in range(MAX_COLORS) if c in used_colors]
            
            # Shuffle for diversity
            random.shuffle(unused)
            random.shuffle(used)
            
            # Try unused colors first
            for c in unused:
                if time.time() > deadline:
                    return False, num_used
                
                # Check if assigning c to e violates any cycle constraint
                valid = True
                for cid in edge_to_cycles[e]:
                    ci = cycle_indices[cid]
                    # Collect colors of all 4 edges in this cycle
                    cycle_colors = []
                    for e2 in ci:
                        if e2 == e:
                            cycle_colors.append(c)
                        else:
                            if colors[e2] == -1:
                                # Unassigned - skip check (optimistic)
                                break
                            cycle_colors.append(colors[e2])
                    
                    if len(cycle_colors) == 4:
                        if len(set(cycle_colors)) == 4:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Assign and recurse
                colors[e] = c
                used_colors.add(c)
                new_num_used = num_used + 1 if c not in used_colors else num_used
                # Wait, we just added c to used_colors, so it's now in the set
                # Need to track differently
                
                # Actually, let me fix the tracking
                # Remove the add and track properly
                used_colors.discard(c)
                
                success, best_score = backtrack(depth + 1, num_used + (1 if c not in used_colors else 0))
                
                if success:
                    # Restore and return
                    used_colors.add(c)  # Restore
                    colors[e] = -1
                    used_colors.discard(c)  # Clean up
                    return True, best_score
                
                # Backtrack
                colors[e] = -1
            
            # Try used colors
            for c in used:
                if time.time() > deadline:
                    return False, num_used
                
                # Check validity
                valid = True
                for cid in edge_to_cycles[e]:
                    ci = cycle_indices[cid]
                    cycle_colors = []
                    for e2 in ci:
                        if e2 == e:
                            cycle_colors.append(c)
                        else:
                            if colors[e2] == -1:
                                break
                            cycle_colors.append(colors[e2])
                    
                    if len(cycle_colors) == 4:
                        if len(set(cycle_colors)) == 4:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                colors[e] = c
                success, best_score = backtrack(depth + 1, num_used)
                
                if success:
                    colors[e] = -1
                    return True, best_score
                
                colors[e] = -1
            
            return False, num_used
        
        return backtrack(0, 0)

    # Run the backtracking solver
    success, score = solve()
    
    if not success or score == 0:
        # Fallback: use a simple valid coloring
        # Hierarchical partition
        colors = [0] * NUM_EDGES
        color_id = 0
        def process_partition(vertices):
            nonlocal color_id
            if len(vertices) <= 1:
                return
            mid = len(vertices) // 2
            left = vertices[:mid]
            right = vertices[mid:]
            for u in left:
                for v in right:
                    idx = edge_idx[u][v]
                    colors[idx] = color_id
            color_id += 1
            process_partition(left)
            process_partition(right)
        process_partition(list(range(N)))
        return colors

    # If we got here but colors weren't fully filled (due to timeout), 
    # we need to fill in the rest or return what we have
    # Check if all edges are assigned
    if -1 in colors:
        # Fill unassigned edges with valid colors
        for e in range(NUM_EDGES):
            if colors[e] == -1:
                # Find a valid color
                found = False
                for c in range(MAX_COLORS):
                    valid = True
                    for cid in edge_to_cycles[e]:
                        ci = cycle_indices[cid]
                        cycle_colors = []
                        for e2 in ci:
                            if e2 == e:
                                cycle_colors.append(c)
                            else:
                                if colors[e2] == -1:
                                    break
                                cycle_colors.append(colors[e2])
                        
                        if len(cycle_colors) == 4:
                            if len(set(cycle_colors)) == 4:
                                valid = False
                                break
                    
                    if valid:
                        colors[e] = c
                        found = True
                        break
                
                if not found:
                    # Last resort: use 0
                    colors[e] = 0

    return colors

if __name__ == "__main__":
    # For local testing
    res = construct()
    print("Distinct colors:", len(set(res)))
