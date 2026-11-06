"""Max-SAT / WalkSAT-style local search for rainbow 4-cycle edge coloring."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n = 20
    num_colors = 20
    # All possible edges: 190
    edge_list = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(edge_list)  # 190

    # Edge index lookup
    edge_idx = {}
    for i, (u, v) in enumerate(edge_list):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i

    # Precompute all 4-cycles: (u, v, w, x) with u < w, v < x, and
    # edges (u,v), (v,w), (w,x), (x,u) all in edge_list
    # A 4-cycle is determined by choosing opposite pairs (u,w) and (v,x)
    # where u,v,w,x are all distinct, and all four edges exist in edge_list.
    # For efficiency, precompute: for each pair of opposite vertices (u,w),
    # the list of common neighbors gives potential (v,x) pairs.

    # State:
    # present[i] = 1 if edge i is present, 0 otherwise
    # color[i] = color of edge i (0..19), only meaningful if present[i]=1

    def compute_all_cycles():
        """Precompute all 4-cycles as tuples of 4 edge indices."""
        cycles = []
        # adj will change, but for precomputation we use the full K20
        # A 4-cycle u-v-w-x-u exists if all 4 edges are in edge_list
        # Since we're on K20, all edges exist, so all 4-cycles of K20 are candidates
        # But we only check cycles where all 4 edges are present in the current state.
        # Precompute the structural 4-cycles (all exist in K20):
        # For each unordered pair of disjoint edges as "opposite" pairs...
        # Actually, enumerate: for u < w (opposite), for each pair v < x of common neighbors
        # of u and w (in K20, that's all vertices except u,w), the cycle is u-v-w-x-u.
        # Edge indices: (u,v), (v,w), (w,x), (x,u)
        for u in range(n):
            for w in range(u + 1, n):
                # Common neighbors of u and w in K20: all vertices except u, w
                common = [v for v in range(n) if v != u and v != w]
                L = len(common)
                for i in range(L):
                    v = common[i]
                    for j in range(i + 1, L):
                        x = common[j]
                        # Cycle: u-v-w-x-u
                        # Edges: (u,v), (v,w), (w,x), (x,u)
                        e1 = edge_idx[(u, v)]
                        e2 = edge_idx[(v, w)]
                        e3 = edge_idx[(w, x)]
                        e4 = edge_idx[(x, u)]
                        cycles.append((e1, e2, e3, e4))
        return cycles

    all_cycles = compute_all_cycles()
    num_cycles = len(all_cycles)

    def count_violations(present, color):
        """Count number of 4-cycles that are not rainbow (among present edges only)."""
        violations = 0
        for (e1, e2, e3, e4) in all_cycles:
            # All 4 edges must be present for this to be a cycle in the graph
            if not (present[e1] and present[e2] and present[e3] and present[e4]):
                continue
            c1 = color[e1]
            c2 = color[e2]
            c3 = color[e3]
            c4 = color[e4]
            if c1 == c2 or c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4 or c3 == c4:
                violations += 1
        return violations

    def count_violations_fast(present, color):
        """Vectorized-ish violation count using numpy for speed."""
        # Convert to numpy for faster access
        p = present
        c = color
        violations = 0
        for (e1, e2, e3, e4) in all_cycles:
            if not (p[e1] and p[e2] and p[e3] and p[e4]):
                continue
            c1 = c[e1]
            c2 = c[e2]
            c3 = c[e3]
            c4 = c[e4]
            if c1 == c2 or c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4 or c3 == c4:
                violations += 1
        return violations

    # WalkSAT parameters
    # State: present (bool array of size m), color (int array of size m)

    def random_state():
        """Generate a random initial state."""
        # Start with a moderately dense random graph
        density = 0.5
        present = np.array([1 if random.random() < density else 0 for _ in range(m)], dtype=np.uint8)
        color = np.array([random.randint(0, num_colors - 1) for _ in range(m)], dtype=np.int32)
        return present, color

    def walksat_step(present, color, temperature):
        """
        One WalkSAT step:
        1. Pick a random violated cycle (if any)
        2. With probability p_walk, flip a random variable in that clause
        3. Otherwise, flip the variable that minimizes the number of newly violated clauses
        Variables: each edge has two "variables": present (0/1) and color (0..19)
        We treat them as: action = (edge_idx, type, new_value)
          type=0: change present
          type=1: change color
        """
        # Find violated cycles
        violated = []
        for (e1, e2, e3, e4) in all_cycles:
            if not (present[e1] and present[e2] and present[e3] and present[e4]):
                continue
            c1 = color[e1]
            c2 = color[e2]
            c3 = color[e3]
            c4 = color[e4]
            if c1 == c2 or c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4 or c3 == c4:
                violated.append((e1, e2, e3, e4))

        if not violated:
            return present, color, 0

        e1, e2, e3, e4 = random.choice(violated)
        cycle_edges = [e1, e2, e3, e4]

        # Generate candidate moves: for each edge in the cycle, try changing present or color
        # Score each move by the change in total violations (approximate: only check affected cycles)

        p_walk = 0.25
        if random.random() < p_walk:
            # Random walk: pick a random edge in cycle and random action
            e = random.choice(cycle_edges)
            action_type = random.randint(0, 1)
            if action_type == 0:
                # Toggle present
                present[e] = 1 - present[e]
            else:
                # Random new color
                old_c = color[e]
                new_c = random.randint(0, num_colors - 1)
                if new_c == old_c:
                    new_c = (new_c + 1) % num_colors
                color[e] = new_c
            new_viol = count_violations_fast(present, color)
            return present, color, new_viol
        else:
            # Greedy: try all single-variable flips in the cycle, pick best
            best_score = count_violations_fast(present, color)
            best_move = None

            for e in cycle_edges:
                # Try toggling present
                old_present = present[e]
                present[e] = 1 - old_present
                score = count_violations_fast(present, color)
                if score < best_score:
                    best_score = score
                    best_move = (e, 'present', 1 - old_present)
                present[e] = old_present

                # Try changing color (try a few random colors)
                old_color = color[e]
                for _ in range(3):
                    new_c = random.randint(0, num_colors - 1)
                    if new_c == old_color:
                        continue
                    color[e] = new_c
                    score = count_violations_fast(present, color)
                    if score < best_score:
                        best_score = score
                        best_move = (e, 'color', new_c)
                    color[e] = old_color

            if best_move is not None:
                e, mtype, mval = best_move
                if mtype == 'present':
                    present[e] = mval
                else:
                    color[e] = mval
            # else: no improvement found, keep state

            return present, color, best_score

    # Objective: maximize edges subject to 0 violations
    # Energy = violations * BIG - edges_present
    BIG = 1000

    def energy(present, color):
        v = count_violations_fast(present, color)
        e = int(np.sum(present))
        return v * BIG - e

    # Multi-restart WalkSAT
    best_present = None
    best_color = None
    best_energy = float('inf')
    best_edges = 0
    best_violations = float('inf')

    num_restarts = 50  # Will be limited by time
    T = 5.0
    T_min = 0.1
    cooling = 0.97
    max_steps_per_restart = 5000

    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        present, color = random_state()
        current_viol = count_violations_fast(present, color)
        current_edges = int(np.sum(present))
        current_energy = current_viol * BIG - current_edges

        T = 5.0

        for step in range(max_steps_per_restart):
            if time.time() > deadline - 3.0:
                break

            if current_viol == 0:
                # Valid! Try to add edges
                missing = [i for i in range(m) if present[i] == 0]
                if not missing:
                    break
                e = random.choice(missing)
                present[e] = 1
                color[e] = random.randint(0, num_colors - 1)
                new_viol = count_violations_fast(present, color)
                new_edges = current_edges + 1
                if new_viol == 0:
                    current_viol = 0
                    current_edges = new_edges
                    current_energy = -new_edges
                    # Update best
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_violations = 0
                        best_present = present.copy()
                        best_color = color.copy()
                        best_energy = current_energy
                else:
                    # Revert
                    present[e] = 0
                    # Try to fix by recoloring nearby edges? Just revert for now
                    pass
                continue

            # Normal WalkSAT step
            present, color, new_viol = walksat_step(present, color, T)
            new_edges = int(np.sum(present))
            new_energy = new_viol * BIG - new_edges

            delta_E = new_energy - current_energy
            if delta_E < 0 or random.random() < np.exp(-delta_E / max(T, 0.01)):
                current_viol = new_viol
                current_edges = new_edges
                current_energy = new_energy

                if current_viol < best_violations or (current_viol == best_violations and current_edges > best_edges):
                    best_violations = current_viol
                    best_edges = current_edges
                    best_present = present.copy()
                    best_color = color.copy()
                    best_energy = current_energy
            # else: revert (already reverted in walksat_step if no move applied)

            T *= cooling
            if T < T_min:
                T = T_min

    # If we found a valid solution (0 violations), great
    # Otherwise, we need a fallback valid solution
    if best_violations > 0 or best_present is None:
        # Fallback: construct a C4-free graph (no 4-cycles at all means constraint is vacuously satisfied)
        # Greedy C4-free construction
        adj = np.zeros((n, n), dtype=np.uint8)
        edges_to_try = list(range(m))
        random.shuffle(edges_to_try)
        for i in edges_to_try:
            u, v = edge_list[i]
            # Check if adding (u,v) creates a C4
            # A C4 through (u,v) exists if there's a common neighbor of u and v that also
            # shares another common neighbor... Actually, simpler:
            # Adding edge (u,v) creates a C4 iff there exist a,b such that
            # a is common neighbor of u and v, and b is common neighbor of u and v,
            # and a!=b, and edges (a,b) or the path... 
            # Standard C4-free check: (u,v) can be added iff for all a != u,v,
            # the number of common neighbors of u and v through a is 0, i.e.,
            # there's no vertex a such that adj[u,a] and adj[v,a] are both 1 AND
            # there's no vertex b such that adj[u,b] and adj[v,b] are both 1 and a!=b.
            # Simpler: count common neighbors of u and v. If >= 2, adding (u,v) creates a C4.
            # Wait, that's not quite right either. Let me think...
            # A 4-cycle containing edge (u,v): u-a-v-b-u where a,b are distinct, a!=u,v, b!=u,v, a!=b.
            # So we need: adj[u,a]=1, adj[a,v]=1, adj[v,b]=1, adj[b,u]=1, with a!=b, a,b not in {u,v}.
            # This means: a is a common neighbor of u and v, and b is a common neighbor of u and v, a!=b.
            # So: if |N(u) ∩ N(v)| >= 2, adding (u,v) creates a C4.
            common = np.sum(adj[u] & adj[v])
            if common < 2:
                adj[u, v] = 1
                adj[v, u] = 1
                # Set present and color
                if best_present is None:
                    best_present = np.zeros(m, dtype=np.uint8)
                    best_color = np.zeros(m, dtype=np.int32)
                best_present[i] = 1
                best_color[i] = 0  # All same color is fine since no C4 exists

        # Count edges in fallback
        fb_edges = int(np.sum(adj)) // 2
        if best_present is None or fb_edges > best_edges:
            best_edges = fb_edges
            best_violations = 0
            # Rebuild best_present from adj
            if best_present is None:
                best_present = np.zeros(m, dtype=np.uint8)
                best_color = np.zeros(m, dtype=np.int32)
            else:
                # Reset
                best_present = np.zeros(m, dtype=np.uint8)
                best_color = np.zeros(m, dtype=np.int32)
            for i in range(m):
                u, v = edge_list[i]
                if adj[u, v]:
                    best_present[i] = 1
                    best_color[i] = 0

    # Final verification and output
    if best_present is None:
        # Shouldn't happen, but safety
        return [[0, 1, 0]]

    result = []
    for i in range(m):
        if best_present[i]:
            u, v = edge_list[i]
            c = int(best_color[i])
            result.append([u, v, c])

    return result

# EVOLVE-BLOCK-END
