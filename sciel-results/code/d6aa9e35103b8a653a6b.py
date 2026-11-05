import time
import random
import itertools
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for 120s limit

    # Fallback: n=9 seed
    n_seed = 9
    seed_colors = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best = {'n': n_seed, 'colors': seed_colors}

    # First: quickly find a valid n=17 via SA
    r17 = _sa_search(17, deadline - 45.0, 0)
    if r17 is not None:
        best = r17

    # Try to extend to n=18 via backtracking with forward checking
    if time.time() < deadline - 5.0:
        r18 = _extend_to_18_fc(r17, deadline - 5.0)
        if r18 is not None:
            return r18

    return best


def _sa_search(n, deadline, seed=0):
    """Simulated annealing to find a C6-free 4-coloring of K_n."""
    random.seed(seed)
    edge_map, edge_list = _build_edge_map(n)
    num_edges = len(edge_list)
    cycles, edge_to_cycles = _build_cycles(n, edge_map)
    num_cycles = len(cycles)

    if num_cycles == 0:
        return {'n': n, 'colors': [0] * num_edges}

    colors = [random.randint(0, 3) for _ in range(num_edges)]
    cycle_counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
    total_viol = 0
    for c_id in range(num_cycles):
        for e in cycles[c_id]:
            cycle_counts[c_id][colors[e]] += 1
        for c in range(4):
            if cycle_counts[c_id][c] == 6:
                total_viol += 1
                break

    T = 2.0
    cooling = 0.9995
    min_T = 0.05

    while time.time() < deadline:
        if total_viol == 0:
            return {'n': n, 'colors': colors}

        e = random.randint(0, num_edges - 1)
        oc = colors[e]
        nc = random.randint(0, 3)
        if nc == oc:
            continue

        delta = 0
        for c_id in edge_to_cycles[e]:
            cc = cycle_counts[c_id]
            old_v = 1 if cc[oc] == 6 else 0
            cc[oc] -= 1
            cc[nc] += 1
            new_v = 1 if cc[nc] == 6 else 0
            delta += new_v - old_v

        if delta < 0 or random.random() < 2.718281828 ** (-delta / T):
            colors[e] = nc
            total_viol += delta
        else:
            for c_id in edge_to_cycles[e]:
                cc = cycle_counts[c_id]
                cc[oc] += 1
                cc[nc] -= 1
        T = max(min_T, T * cooling)

    if total_viol == 0:
        return {'n': n, 'colors': colors}
    return None


def _extend_to_18_fc(r17, deadline):
    """Try to extend a valid K_17 coloring to K_18 via backtracking with forward checking."""
    if r17 is None:
        return None

    n_old = 17
    n_new = 18
    old_colors = r17['colors']

    # Build edge map for n=18
    edge_map, edge_list = _build_edge_map(n_new)
    num_edges = len(edge_list)
    n17_edges = n_old * (n_old - 1) // 2

    # Build cycles for n=18
    cycles, edge_to_cycles = _build_cycles(n_new, edge_map)

    # The new edges are those involving vertex 17
    # In lexicographic order, these are edges (17, j) for j in 0..16, but since i<j,
    # they are (j, 17) for j in 0..16, which come at the end of the list
    # Actually, edges are (i,j) with i<j. So (0,17), (1,17), ..., (16,17) are the new edges
    # They are the last 17 edges in the list
    
    new_edge_indices = list(range(n17_edges, num_edges))
    
    # Initialize colors: first n17_edges from r17, rest to be determined
    colors = list(old_colors)
    # We'll assign colors to new edges via backtracking
    
    # Precompute for each new edge, which cycles it belongs to
    # and for each such cycle, which other new edges are in that cycle
    
    # For forward checking: when we assign a color to a new edge, we check if it creates a C6
    # A C6 is formed when all 6 edges of a cycle have the same color
    
    # Let's use a different approach: backtracking with constraint propagation
    # For each new edge, try each color, and check if it's consistent with existing assignments
    
    # Build a map: for each new edge index, which cycles contain it
    new_edge_to_cycles = {}
    for e in new_edge_indices:
        new_edge_to_cycles[e] = edge_to_cycles[e]
    
    # For each cycle, track how many of its edges are new edges
    cycle_new_edges = {}
    for c_id in range(len(cycles)):
        new_e_in_cycle = [e for e in cycles[c_id] if e in new_edge_to_cycles]
        if new_e_in_cycle:
            cycle_new_edges[c_id] = new_e_in_cycle
    
    # Backtracking search
    # Order new edges: maybe by most constrained first
    # For simplicity, just use the natural order
    
    def is_valid_assignment(colors, e, c):
        """Check if assigning color c to edge e is valid (doesn't create a C6)."""
        for c_id in new_edge_to_cycles[e]:
            cycle = cycles[c_id]
            # Check if all 6 edges would have color c
            # Some edges are already assigned (old edges or previously assigned new edges)
            all_c = True
            for edge in cycle:
                if edge == e:
                    continue
                if colors[edge] != c:
                    all_c = False
                    break
            if all_c:
                return False
        return True
    
    def backtrack(idx):
        """Backtracking: assign colors to new_edge_indices[idx:]"""
        if time.time() > deadline:
            return False
        
        if idx == len(new_edge_indices):
            # All new edges assigned, check if valid
            # We've been checking incrementally, so should be valid
            return True
        
        e = new_edge_indices[idx]
        
        for c in range(4):
            if is_valid_assignment(colors, e, c):
                colors[e] = c
                if backtrack(idx + 1):
                    return True
                colors[e] = -1  # backtrack
        
        return False
    
    # Initialize new edges to -1 (unassigned)
    for e in new_edge_indices:
        colors[e] = -1
    
    if backtrack(0):
        return {'n': n_new, 'colors': colors}
    
    # If backtracking fails, try a randomized local search approach
    # as a fallback
    random.seed(42)
    for e in new_edge_indices:
        colors[e] = random.randint(0, 3)
    
    # Build cycle counts and violations
    cycle_counts = [[0, 0, 0, 0] for _ in range(len(cycles))]
    total_viol = 0
    for c_id in range(len(cycles)):
        for e in cycles[c_id]:
            cycle_counts[c_id][colors[e]] += 1
        for c in range(4):
            if cycle_counts[c_id][c] == 6:
                total_viol += 1
                break
    
    if total_viol == 0:
        return {'n': n_new, 'colors': colors}
    
    # Local search on new edges only
    T = 1.0
    cooling = 0.999
    min_T = 0.02
    
    while time.time() < deadline:
        if total_viol == 0:
            return {'n': n_new, 'colors': colors}
        
        # Pick a random new edge
        e = random.choice(new_edge_indices)
        oc = colors[e]
        nc = random.randint(0, 3)
        if nc == oc:
            continue
        
        delta = 0
        for c_id in edge_to_cycles[e]:
            cc = cycle_counts[c_id]
            old_v = 1 if cc[oc] == 6 else 0
            cc[oc] -= 1
            cc[nc] += 1
            new_v = 1 if cc[nc] == 6 else 0
            delta += new_v - old_v
        
        if delta < 0 or random.random() < 2.718281828 ** (-delta / T):
            colors[e] = nc
            total_viol += delta
        else:
            for c_id in edge_to_cycles[e]:
                cc = cycle_counts[c_id]
                cc[oc] += 1
                cc[nc] -= 1
        T = max(min_T, T * cooling)
    
    if total_viol == 0:
        return {'n': n_new, 'colors': colors}
    return None


def _build_edge_map(n):
    edge_map = {}
    edge_list = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            edge_list.append((i, j))
            idx += 1
    return edge_map, edge_list


def _build_cycles(n, edge_map):
    cycles = []
    edge_to_cycles = {e: [] for e in range(n * (n - 1) // 2)}

    for subset in itertools.combinations(range(n), 6):
        seen = set()
        for perm in itertools.permutations(subset):
            mn = min(perm)
            mi = perm.index(mn)
            rot = perm[mi:] + perm[:mi]
            if rot[1] > rot[-1]:
                rot = (rot[0],) + rot[:0:-1]
            if rot in seen:
                continue
            seen.add(rot)

            edges = []
            for k in range(6):
                u, v = rot[k], rot[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                edges.append(edge_map[(u, v)])

            cid = len(cycles)
            cycles.append(edges)
            for e in edges:
                edge_to_cycles[e].append(cid)

    return cycles, edge_to_cycles
# EVOLVE-BLOCK-END
