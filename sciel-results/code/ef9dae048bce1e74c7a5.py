import time
import random
import sys
import numpy as np


def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4 using incremental
    bitset-based local search. Maximizes n.
    """
    deadline = time.time() + 140.0  # Safety margin for 150s limit

    # Known seed solution for n=11 (cyclic coloring)
    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()

    # Precompute edge index mapping
    def build_edge_map(n):
        """Returns list of (u, v) for each edge in lexicographic order."""
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        return edges

    def edge_index(i, j, n):
        """O(1) edge index for (i,j) with i<j in lexicographic order."""
        # Number of edges before vertex i: i*(i-1)/2
        # Within vertex i's block, j is at offset (j - i - 1)
        return i * (i - 1) // 2 + (j - i - 1)

    def is_valid(n, colors):
        """Verify no monochromatic C4."""
        if len(colors) != n * (n - 1) // 2:
            return False
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return False
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common & (common - 1):
                        return False
        return True

    class IncrementalCost:
        """
        Maintains per-color neighbor bitmasks and total C4-violation count.
        Supports O(n) incremental updates when one edge color changes.
        """
        def __init__(self, n, colors):
            self.n = n
            self.colors = colors
            self.neighbors = [[0] * n for _ in range(5)]
            self.violation_count = 0
            self._build()

        def _build(self):
            """Build neighbor bitmasks and compute total violations."""
            n = self.n
            self.neighbors = [[0] * n for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = self.colors[idx]
                    self.neighbors[c][i] |= (1 << j)
                    self.neighbors[c][j] |= (1 << i)
                    idx += 1
            self.violation_count = self._count_violations()

        def _count_violations(self):
            """Count pairs (u,v) with >=2 common neighbors in any color."""
            n = self.n
            count = 0
            for c in range(5):
                nc = self.neighbors[c]
                for u in range(n):
                    nu = nc[u]
                    if nu == 0:
                        continue
                    for v in range(u + 1, n):
                        nv = nc[v]
                        if nv == 0:
                            continue
                        common = nu & nv
                        if common & (common - 1):
                            count += 1
            return count

        def _count_violations_for_color(self, c):
            """Count violations for a single color."""
            n = self.n
            nc = self.neighbors[c]
            count = 0
            for u in range(n):
                nu = nc[u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = nc[v]
                    if nv == 0:
                        continue
                    common = nu & nv
                    if common & (common - 1):
                        count += 1
            return count

        def _violations_involving_vertex_color(self, vertex, c):
            """Count violations (u,v) where u or v == vertex, in color c."""
            n = self.n
            nc = self.neighbors[c]
            nv = nc[vertex]
            count = 0
            for w in range(n):
                if w == vertex:
                    continue
                nw = nc[w]
                if nw == 0:
                    continue
                common = nv & nw
                if common & (common - 1):
                    count += 1
            return count

        def change_edge_color(self, u, v, old_c, new_c):
            """
            Change edge (u,v) from old_c to new_c.
            Returns the change in total violation count.
            Updates state in place.
            """
            n = self.n
            bit_u = 1 << u
            bit_v = 1 << v

            # Remove u from v's old_c neighbors and vice versa
            self.neighbors[old_c][v] &= ~bit_u
            self.neighbors[old_c][u] &= ~bit_v

            # Count violations for old_c involving u or v BEFORE the change
            # Actually we need to count violations that will be affected.
            # Changing (u,v) in old_c: removes edge u-v from color old_c.
            # This affects common-neighbor counts for pairs (u, w) and (v, w) in old_c.
            # Specifically, for any w, common(u,w) in old_c loses bit_v if v was in N_old(u)
            # and common(v,w) in old_c loses bit_u if u was in N_old(v).
            # But we already removed the bits, so we need to count BEFORE removal.

            # Let me redo: count violations in old_c involving u or v BEFORE removal
            old_viol_u = self._violations_involving_vertex_color(u, old_c)
            old_viol_v = self._violations_involving_vertex_color(v, old_c)
            # But (u,v) pair is counted once, so we need to be careful.
            # _violations_involving_vertex_color(u, c) counts all pairs (u,w) for w!=u
            # _violations_involving_vertex_color(v, c) counts all pairs (v,w) for w!=v
            # The pair (u,v) is counted in both. So total unique pairs involving u or v:
            # = old_viol_u + old_viol_v - (1 if (u,v) is a violation in old_c else 0)

            # Check if (u,v) was a violation pair in old_c
            # (u,v) is a violation if they have >=2 common neighbors in old_c
            # Common neighbors of u and v in old_c (before removal):
            # N_old(u) & N_old(v) - but we already removed bits!
            # Let's compute before removal.
            # Actually, let me restructure: compute delta by counting before/after for affected pairs.

            # Simpler approach: count violations for old_c before and after the bit removal,
            # but only for pairs involving u or v.

            # Let me just recompute the affected pairs.
            # Pairs affected in old_c: (u, w) for all w != u, and (v, w) for all w != v.
            # That's 2*(n-1) - 1 = 2n-3 pairs (since (u,v) counted twice).

            # Actually, the cleanest approach:
            # 1. Count violations in old_c involving u or v BEFORE change
            # 2. Remove bits
            # 3. Count violations in old_c involving u or v AFTER change
            # 4. Add bits for new_c
            # 5. Count violations in new_c involving u or v BEFORE adding
            # 6. Count violations in new_c involving u or v AFTER adding
            # Delta = (new_old_after - old_old_before) + (new_new_after - new_new_before)

            # Wait, I already removed the bits. Let me undo and redo properly.

            # Undo removal
            self.neighbors[old_c][v] |= bit_u
            self.neighbors[old_c][u] |= bit_v

            # Count before for old_c
            before_old = self._violations_involving_vertex_color(u, old_c) + \
                         self._violations_involving_vertex_color(v, old_c)
            # Subtract double-count of (u,v) pair
            common_uv = self.neighbors[old_c][u] & self.neighbors[old_c][v]
            if common_uv & (common_uv - 1):
                before_old -= 1

            # Now do the removal
            self.neighbors[old_c][v] &= ~bit_u
            self.neighbors[old_c][u] &= ~bit_v

            # Count after for old_c
            after_old = self._violations_involving_vertex_color(u, old_c) + \
                        self._violations_involving_vertex_color(v, old_c)
            common_uv2 = self.neighbors[old_c][u] & self.neighbors[old_c][v]
            if common_uv2 & (common_uv2 - 1):
                after_old -= 1

            # Count before for new_c (before adding)
            before_new = self._violations_involving_vertex_color(u, new_c) + \
                         self._violations_involving_vertex_color(v, new_c)
            common_uv3 = self.neighbors[new_c][u] & self.neighbors[new_c][v]
            if common_uv3 & (common_uv3 - 1):
                before_new -= 1

            # Add bits for new_c
            self.neighbors[new_c][v] |= bit_u
            self.neighbors[new_c][u] |= bit_v

            # Count after for new_c
            after_new = self._violations_involving_vertex_color(u, new_c) + \
                        self._violations_involving_vertex_color(v, new_c)
            common_uv4 = self.neighbors[new_c][u] & self.neighbors[new_c][v]
            if common_uv4 & (common_uv4 - 1):
                after_new -= 1

            delta = (after_old - before_old) + (after_new - before_new)
            self.violation_count += delta
            return delta

    def solve_local_search(n, time_limit):
        """
        Incremental local search: repeatedly pick a random edge, try changing
        its color, accept if it reduces violations. With occasional random
        restarts and temperature-like acceptance of worse moves.
        """
        total_edges = n * (n - 1) // 2
        edges = build_edge_map(n)

        # Initialize with random coloring
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        ic = IncrementalCost(n, colors)

        if ic.violation_count == 0:
            return colors

        best_cost = ic.violation_count
        best_colors = colors[:]

        # Local search with restarts
        start_time = time.time()
        no_improve_steps = 0
        max_no_improve = 50000

        while time.time() - start_time < time_limit:
            # Check time periodically
            if no_improve_steps > max_no_improve:
                # Restart with new random coloring
                colors = [random.randint(0, 4) for _ in range(total_edges)]
                ic = IncrementalCost(n, colors)
                if ic.violation_count == 0:
                    return colors
                if ic.violation_count < best_cost:
                    best_cost = ic.violation_count
                    best_colors = colors[:]
                no_improve_steps = 0
                continue

            # Pick random edge
            e_idx = random.randint(0, total_edges - 1)
            u, v = edges[e_idx]
            old_c = colors[e_idx]

            # Pick new color
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue

            # Try the change
            delta = ic.change_edge_color(u, v, old_c, new_c)
            colors[e_idx] = new_c

            if delta < 0:
                # Improvement
                no_improve_steps = 0
                if ic.violation_count < best_cost:
                    best_cost = ic.violation_count
                    best_colors = colors[:]
                    if best_cost == 0:
                        return best_colors
            else:
                # Accept with small probability (simulated annealing-like)
                # Use a simple probability based on delta
                if delta <= 2 and random.random() < 0.1:
                    no_improve_steps += 1
                else:
                    # Reject: undo
                    ic.change_edge_color(u, v, new_c, old_c)
                    colors[e_idx] = old_c
                    no_improve_steps += 1

        return best_colors if best_cost == 0 else None

    # Also try a structured initialization based on modular arithmetic
    def structured_init(n):
        """
        Try to initialize with a structured pattern that might be closer to valid.
        Use modular arithmetic: color(i,j) = (i + j) % 5 or similar.
        """
        total_edges = n * (n - 1) // 2
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                # Try different structured patterns
                c = (i + j) % 5
                colors.append(c)
        return colors

    # Strategy: try n from 29 down
    for n in range(29, 11, -1):
        if time.time() > deadline - 5:
            break

        if n >= 27:
            t_alloc = 8.0
        elif n == 26:
            t_alloc = 40.0
        elif n >= 20:
            t_alloc = 5.0
        else:
            t_alloc = 3.0

        t_alloc = min(t_alloc, deadline - time.time() - 2.0)
        if t_alloc <= 0:
            break

        # Try structured init first for a short time
        if n <= 26:
            struct_colors = structured_init(n)
            ic_test = IncrementalCost(n, struct_colors)
            if ic_test.violation_count == 0:
                if is_valid(n, struct_colors):
                    best_n = n
                    best_colors = struct_colors
                    return {'n': best_n, 'colors': best_colors}
            # Use structured init as starting point for local search
            # (modify solve to accept initial colors)
            # For now, just run standard local search
            res = solve_local_search(n, t_alloc)
        else:
            res = solve_local_search(n, t_alloc)

        if res:
            if is_valid(n, res):
                best_n = n
                best_colors = res
                return {'n': best_n, 'colors': best_colors}

    # Return best found
    return {'n': best_n, 'colors': best_colors}


if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
