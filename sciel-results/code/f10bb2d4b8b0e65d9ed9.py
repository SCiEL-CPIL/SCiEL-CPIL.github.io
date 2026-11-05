"""
5-colourings of K_n without monochromatic C4 (R_5(C4) > n).
Combines cyclic seed idea, algebraic construction attempts, and
C4-counter-based local search for repair.
"""
import time
import random

_START = time.time()
_LIMIT = 145.0  # safety margin before 150s kill
_MARGIN = 5.0


def _time_left():
    return _LIMIT - _MARGIN - (time.time() - _START)


def _edge_index(n, i, j):
    """Index of edge (i,j) with i<j in the lexicographic list."""
    if i > j:
        i, j = j, i
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)


def _validate(n, colors):
    """Check that no colour class contains a 4-cycle (C4)."""
    m = n * (n - 1) // 2
    if len(colors) != m:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
    # Build neighbour sets per colour
    nbrs = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            nbrs[c][i].add(j)
            nbrs[c][j].add(i)
            idx += 1
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                if len(nbrs[c][i] & nbrs[c][j]) >= 2:
                    return False
    return True


class C4Counter:
    """
    Maintains, for each colour c and each pair (i,j), the number of
    common neighbours in colour c. A violation exists when any count >= 2.
    Supports fast update when a single edge is recoloured.
    """
    def __init__(self, n, colors):
        self.n = n
        self.colors = colors[:]
        # common[c][i][j] = number of common neighbours of i and j in colour c
        # Only store for i < j
        self.common = [[None] * n for _ in range(5)]
        for c in range(5):
            self.common[c] = [None] * n
            for i in range(n):
                self.common[c][i] = [0] * n
        # Build neighbour lists
        self.nbrs = [[[] for _ in range(n)] for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                self.nbrs[c][i].append(j)
                self.nbrs[c][j].append(i)
                idx += 1
        # Compute common counts
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    cnt = 0
                    si = set(self.nbrs[c][i])
                    for k in self.nbrs[c][j]:
                        if k in si:
                            cnt += 1
                    self.common[c][i][j] = cnt
                    self.common[c][j][i] = cnt
        self.total_violations = 0
        self._count_violations()

    def _count_violations(self):
        self.total_violations = 0
        for c in range(5):
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if self.common[c][i][j] >= 2:
                        self.total_violations += 1

    def change_edge(self, i, j, new_c):
        """Recolour edge (i,j) to new_c, updating all structures."""
        if i > j:
            i, j = j, i
        idx = _edge_index(self.n, i, j)
        old_c = self.colors[idx]
        if old_c == new_c:
            return
        self.colors[idx] = new_c

        # Remove i from nbrs[old_c][j] and j from nbrs[old_c][i]
        # Update common counts for pairs involving i or j in old_c
        self._remove_from_colour(old_c, i, j)
        # Add to new colour
        self._add_to_colour(new_c, i, j)

    def _remove_from_colour(self, c, i, j):
        """Remove edge (i,j) from colour c."""
        # Update common counts: for any k != i,j, the pair (k,i) and (k,j)
        # lose one common neighbour if the other was connected.
        # Specifically:
        # For each k in nbrs[c][i] (k != j): common[c][k][j] decreases by 1
        # For each k in nbrs[c][j] (k != i): common[c][k][i] decreases by 1
        # Also common[c][i][j] itself: count common neighbours of i and j in c
        #   This doesn't change directly from removing edge (i,j), since
        #   common neighbours are vertices connected to both i and j.
        #   Wait: removing edge (i,j) means i is no longer a neighbour of j.
        #   So for any k that is a common neighbour of i and j, the pair (i,j)
        #   still has k as common neighbour. The edge (i,j) itself doesn't
        #   affect common[i][j] directly.
        #   But it DOES affect: for any k, if k was connected to both i and j,
        #   then i is a neighbour of k (in c) and j is a neighbour of k (in c).
        #   When we remove edge (i,j), i is no longer a neighbour of j, but
        #   i is still a neighbour of k. So common[k][i] and common[k][j]
        #   are affected only if the other endpoint was involved.
        #
        # Let me think more carefully:
        # common[c][a][b] = |N_c(a) ∩ N_c(b)|
        # When we remove edge (i,j) from colour c:
        #   - i is removed from N_c(j), and j is removed from N_c(i).
        #   - For any pair (a,b):
        #     If a=i and b=j: common[i][j] = |N_c(i) ∩ N_c(j)|.
        #       N_c(i) lost j, N_c(j) lost i. But j was not in N_c(i) before?
        #       Wait, j IS in N_c(i) because edge (i,j) exists in c.
        #       So N_c(i) ∩ N_c(j): j is in N_c(i), is j in N_c(j)? No.
        #       i is in N_c(j), is i in N_c(i)? No.
        #       So common[i][j] doesn't change from removing (i,j) itself.
        #       Actually it might: if there's some k such that k is in both
        #       N_c(i) and N_c(j), that's unchanged. The removal of (i,j)
        #       removes i from N_c(j) and j from N_c(i). But i was already
        #       in N_c(j) (since edge exists), and j was in N_c(i).
        #       After removal, i is no longer in N_c(j), j no longer in N_c(i).
        #       So for pair (i,j): the intersection N_c(i) ∩ N_c(j) loses
        #       nothing because i ∉ N_c(i) and j ∉ N_c(j) always.
        #       So common[i][j] unchanged.
        #
        #     If a=i, b=k (k≠j): common[i][k] = |N_c(i) ∩ N_c(k)|.
        #       N_c(i) lost j. So if j ∈ N_c(k), then common[i][k] decreases by 1.
        #       j ∈ N_c(k) iff edge (j,k) is in colour c, i.e., k ∈ nbrs[c][j].
        #     Similarly if a=k, b=i: same thing.
        #
        #     If a=j, b=k (k≠i): common[j][k] = |N_c(j) ∩ N_c(k)|.
        #       N_c(j) lost i. So if i ∈ N_c(k), common[j][k] decreases by 1.
        #       i ∈ N_c(k) iff k ∈ nbrs[c][i].
        #
        #     If a=k1, b=k2 (both ≠ i,j): no change.
        #
        # So the updates are:
        #   For each k in nbrs[c][j] (k ≠ i): common[i][k] -= 1, common[k][i] -= 1
        #   For each k in nbrs[c][i] (k ≠ j): common[j][k] -= 1, common[k][j] -= 1

        old_total = self.total_violations

        # For k in nbrs[c][j], k != i: common[i][k] decreases
        for k in list(self.nbrs[c][j]):
            if k == i:
                continue
            old_val = self.common[c][i][k]
            self.common[c][i][k] -= 1
            self.common[c][k][i] -= 1
            new_val = self.common[c][i][k]
            # Update violation count
            if old_val >= 2 and new_val < 2:
                self.total_violations -= 1
            elif old_val < 2 and new_val >= 2:
                self.total_violations += 1

        # For k in nbrs[c][i], k != j: common[j][k] decreases
        for k in list(self.nbrs[c][i]):
            if k == j:
                continue
            old_val = self.common[c][j][k]
            self.common[c][j][k] -= 1
            self.common[c][k][j] -= 1
            new_val = self.common[c][j][k]
            if old_val >= 2 and new_val < 2:
                self.total_violations -= 1
            elif old_val < 2 and new_val >= 2:
                self.total_violations += 1

        # Remove i from nbrs[c][j] and j from nbrs[c][i]
        self.nbrs[c][j].remove(i)
        self.nbrs[c][i].remove(j)

    def _add_to_colour(self, c, i, j):
        """Add edge (i,j) to colour c."""
        # Symmetric to removal:
        # For each k in nbrs[c][j] (before adding i): common[i][k] += 1
        # For each k in nbrs[c][i] (before adding j): common[j][k] += 1
        # Then add i to nbrs[c][j] and j to nbrs[c][i]

        # For k in nbrs[c][j]: common[i][k] increases
        for k in list(self.nbrs[c][j]):
            old_val = self.common[c][i][k]
            self.common[c][i][k] += 1
            self.common[c][k][i] += 1
            new_val = self.common[c][i][k]
            if old_val < 2 and new_val >= 2:
                self.total_violations += 1
            elif old_val >= 2 and new_val < 2:
                self.total_violations -= 1

        # For k in nbrs[c][i]: common[j][k] increases
        for k in list(self.nbrs[c][i]):
            old_val = self.common[c][j][k]
            self.common[c][j][k] += 1
            self.common[c][k][j] += 1
            new_val = self.common[c][j][k]
            if old_val < 2 and new_val >= 2:
                self.total_violations += 1
            elif old_val >= 2 and new_val < 2:
                self.total_violations -= 1

        self.nbrs[c][i].append(j)
        self.nbrs[c][j].append(i)

    def find_violation(self):
        """Find any (c, i, j) with common[c][i][j] >= 2."""
        for c in range(5):
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if self.common[c][i][j] >= 2:
                        # Find two common neighbours
                        common_nbrs = []
                        for k in self.nbrs[c][i]:
                            if k in self.nbrs[c][j]:
                                common_nbrs.append(k)
                                if len(common_nbrs) >= 2:
                                    break
                        if len(common_nbrs) >= 2:
                            return (c, i, j, common_nbrs[:2])
        return None


def _seed_construction():
    """Original K11 seed: split into five Hamiltonian cycles by circular distance 1..5."""
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}


def _cyclic_construction(n):
    """Cyclic 5-colouring using circular distance mapped to colour (d-1)%5."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            d = min(d, n - d)
            c = (d - 1) % 5
            colors.append(c)
    return {'n': n, 'colors': colors}


def _local_search(n, max_iter=200000, seed=42):
    """
    Local search using C4Counter for efficient violation tracking.
    Start with cyclic initialisation, then repair C4 violations.
    """
    random.seed(seed)
    m = n * (n - 1) // 2
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = min(j - i, n - j + i)
            colors.append((d - 1) % 5)

    counter = C4Counter(n, colors)

    for it in range(max_iter):
        if _time_left() < 10:
            break
        if counter.total_violations == 0:
            return {'n': n, 'colors': counter.colors}

        # Find a violation
        viol = counter.find_violation()
        if viol is None:
            # Shouldn't happen if total_violations > 0, but just in case
            if _validate(n, counter.colors):
                return {'n': n, 'colors': counter.colors}
            break

        c, i, j, common = viol
        # The C4 is: i - common[0] - j - common[1] - i (in colour c)
        # Edges in this C4: (i, common[0]), (common[0], j), (j, common[1]), (common[1], i)
        edges_in_c4 = [(i, common[0]), (common[0], j), (j, common[1]), (common[1], i)]
        random.shuffle(edges_in_c4)

        best_improvement = None
        best_new_c = None
        best_edge = None

        for a, b in edges_in_c4:
            if a > b:
                a, b = b, a
            for new_c in range(5):
                if new_c == c:
                    continue
                # Estimate: recolor edge (a,b) from c to new_c
                # This removes it from colour c and adds to new_c
                # Quick check: does this reduce violations?
                # We'll just try it and measure
                counter.change_edge(a, b, new_c)
                if counter.total_violations < 0 or (best_improvement is None or counter.total_violations < best_improvement):
                    best_improvement = counter.total_violations
                    best_new_c = new_c
                    best_edge = (a, b)
                # Undo
                counter.change_edge(a, b, c)

        if best_improvement is not None and best_improvement < counter.total_violations:
            a, b = best_edge
            counter.change_edge(a, b, best_new_c)
        else:
            # No improvement found; try random move to escape local minimum
            a, b = edges_in_c4[0]
            if a > b:
                a, b = b, a
            new_c = (c + 1 + random.randint(0, 3)) % 5
            counter.change_edge(a, b, new_c)

    if counter.total_violations == 0 and _validate(n, counter.colors):
        return {'n': n, 'colors': counter.colors}
    return None


def _try_lazebnik_woldar():
    """
    Attempt the Lazebnik-Woldar algebraic construction for n=26.
    Uses the group Z_2 x Z_13 with a specific colouring rule.
    """
    n = 26
    # Vertices: (x, y) where x in Z_2, y in Z_13
    # Encode vertex v as: x = v % 2, y = v // 2

    # In Z_13, we need to partition the 12 non-zero differences into 5 groups
    # such that no group contains a difference d where 2d or similar creates C4.
    #
    # A known construction: use the multiplicative group Z_13^* which is cyclic of order 12.
    # Let g = 2 be a primitive root mod 13: 2^1=2, 2^2=4, 2^3=8, 2^4=16=3, 2^5=6, 2^6=12,
    # 2^7=24=11, 2^8=22=9, 2^9=18=5, 2^10=10, 2^11=20=7, 2^12=14=1
    #
    # The 12 non-zero elements: {1,2,3,4,5,6,7,8,9,10,11,12}
    # Powers of 2: 2,4,8,3,6,12,11,9,5,10,7,1
    #
    # For the 5-colouring, we need to assign each non-zero difference in Z_13
    # (and handle the dx=0 vs dx=1 cases) to one of 5 colours.
    #
    # Actually, the standard construction for R_5(C4) >= 26 is more nuanced.
    # Let me try a direct approach: for each pair of vertices, compute the
    # difference and assign a colour based on a table.

    # Let's try: for dx=0 (same layer), use difference dy in Z_13
    # For dx=1 (different layers), use difference dy in Z_13
    # We have 12 possible dy values for each case, total 24 edge types.
    # We need to assign 5 colours to these 24 types such that no C4 forms.

    # This is complex. Let me try a simpler heuristic:
    # Use quadratic residues to split into 3 groups, then subdivide.

    # Quadratic residues mod 13: {1, 3, 4, 9, 10, 12}
    # Non-residues: {2, 5, 6, 7, 8, 11}

    # Let's try:
    # dx=0, dy in QR: colour 0
    # dx=0, dy in NQR: colour 1
    # dx=1, dy in QR: colour 2
    # dx=1, dy in NQR: colour 3
    # And we need a 5th colour... this only gives 4.

    # Alternative: split QR and NQR further.
    # QR = {1,3,4,9,10,12}, NQR = {2,5,6,7,8,11}
    # Split QR into {1,3,4} and {9,10,12}
    # Split NQR into {2,5,6} and {7,8,11}

    # That gives 4 groups per dx value, total 8 groups. Need to map to 5 colours.
    # This is getting complicated without knowing the exact construction.

    # Let me try a different approach: use a known valid colouring from literature
    # or just do a local search starting from a reasonable initialisation.

    # For now, try a few simple constructions and validate.

    # Construction: use (dy mod 5) for dx=0 and (dy mod 5 + 2) for dx=1
    colors = []
    for i in range(n):
        x1 = i % 2
        y1 = i // 2
        for j in range(i + 1, n):
            x2 = j % 2
            y2 = j // 2
            dx = (x2 - x1) % 2
            dy = (y2 - y1) % 13
            if dy == 0:
                # Same y, different x (must be dx=1 since i<j and same y means x differs)
                # Actually if y1==y2 then dx must be 1 (since x is 0 or 1)
                c = 4
            else:
                if dx == 0:
                    c = dy % 5
                else:
                    c = (dy % 5 + 2) % 5
            colors.append(c)

    if _validate(n, colors):
        return {'n': n, 'colors': colors}
    return None


def _try_algebraic_variants():
    """Try several algebraic constructions for n=26."""
    n = 26

    # Variant 1: Use Z_26 cyclic with a non-trivial colouring
    # Try colour = f(d) where d = min(j-i, n-(j-i)) and f is a specific function
    # For n=26, d ranges from 1 to 13
    # We need to assign 13 values to 5 colours such that no C4 forms.
    # In a cyclic group, a C4 in colour c means there exist a,b,c,d with
    # a-c = b-d and a-d = b-c (mod 26) in colour c, which simplifies to
    # the four vertices forming a rectangle in the cyclic group.
    # Actually in Z_n, a C4 in colour c exists if there are two distinct
    # differences d1, d2 in colour c such that d1 + d2 is also a difference
    # that creates a cycle. This is complex.

    # Let me just try a few mappings and validate.
    for variant in range(20):
        random.seed(variant)
        # Random assignment of distances 1..13 to colours 0..4
        dist_colors = {}
        for d in range(1, 14):
            dist_colors[d] = random.randint(0, 4)
        # Ensure symmetry: distance d and 26-d get same colour
        colors = []
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                d = min(d, n - d)
                colors.append(dist_colors[d])
        if _validate(n, colors):
            return {'n': n, 'colors': colors}

    return None


def construct():
    """Entry point: return the best 5-colouring found."""
    if _time_left() <= 0:
        return _seed_construction()

    best = _seed_construction()

    # Try cyclic constructions for various n
    for n in [25, 20, 15, 12, 10, 5]:
        if _time_left() < 5:
            break
        res = _cyclic_construction(n)
        if _validate(n, res['colors']):
            if n > best['n']:
                best = res

    # Try algebraic construction for n=26
    if best['n'] < 26 and _time_left() > 15:
        res = _try_lazebnik_woldar()
        if res and res['n'] > best['n']:
            best = res

    # Try algebraic variants
    if best['n'] < 26 and _time_left() > 15:
        res = _try_algebraic_variants()
        if res and res['n'] > best['n']:
            best = res

    # Try local search for n=26
    if best['n'] < 26 and _time_left() > 20:
        res = _local_search(26, max_iter=100000, seed=42)
        if res and res['n'] > best['n']:
            best = res

    # Try local search for n=25 if 26 failed
    if best['n'] < 25 and _time_left() > 20:
        res = _local_search(25, max_iter=100000, seed=42)
        if res and res['n'] > best['n']:
            best = res

    # Try local search for n=24
    if best['n'] < 24 and _time_left() > 15:
        res = _local_search(24, max_iter=100000, seed=42)
        if res and res['n'] > best['n']:
            best = res

    # Try local search for n=23
    if best['n'] < 23 and _time_left() > 15:
        res = _local_search(23, max_iter=100000, seed=42)
        if res and res['n'] > best['n']:
            best = res

    # Try local search for n=22
    if best['n'] < 22 and _time_left() > 10:
        res = _local_search(22, max_iter=100000, seed=42)
        if res and res['n'] > best['n']:
            best = res

    # Try local search for n=21
    if best['n'] < 21 and _time_left() > 10:
        res = _local_search(21, max_iter=100000, seed=42)
        if res and res['n'] > best['n']:
            best = res

    # Try local search for n=20
    if best['n'] < 20 and _time_left() > 10:
        res = _local_search(20, max_iter=100000, seed=42)
        if res and res['n'] > best['n']:
            best = res

    # Try local search for n=19
    if best['n'] < 19 and _time_left() > 10:
        res = _local_search(19, max_iter=100000, seed=42)
        if res and res['n'] > best['n']:
            best = res

    # Try local search for n=18
    if best['n'] < 18 and _time_left() > 10:
        res = _local_search(18, max_iter=100000, seed=42)
        if res and res['n'] > best['n']:
            best = res

    # Try local search for n=17
    if best['n'] < 17 and _time_left() > 10:
        res = _local_search(17, max_iter=100000, seed=42)
        if res and res['n'] > best['n']:
            best = res

    # Try local search for n=16
    if best['n'] < 16 and _time_left() > 10:
        res = _local_search(16, max_iter=100000, seed=42)
        if res and res['n'] > best['n']:
            best = res

    # Try local search for n=15
    if best['n'] < 15 and _time_left() > 10:
        res = _local_search(15, max_iter=100000, seed=42)
        if res and res['n'] > best['n']:
            best = res

    # Try local search for n=14
    if best['n'] < 14 and _time_left() > 10:
        res = _local_search(14, max_iter=100000, seed=42)
        if res and res['n'] > best['n']:
            best = res

    # Try local search for n=13
    if best['n'] < 13 and _time_left() > 10:
        res = _local_search(13, max_iter=100000, seed=42)
        if res and res['n'] > best['n']:
            best = res

    # Try local search for n=12
    if best['n'] < 12 and _time_left() > 10:
        res = _local_search(12, max_iter=100000, seed=42)
        if res and res['n'] > best['n']:
            best = res

    # Final fallback
    if best['n'] < 11:
        best = _seed_construction()

    return best


if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, valid={_validate(res['n'], res['colors'])}")
