"""Five-colouring of K_n with no monochromatic C4. Bitset local search."""
import time
import random
import numpy as np

_DEADLINE = time.time() + 140  # safety margin below 150s kill


def _build_adj(colors, n):
    """Return adj[c] as list of int bitsets, one per vertex."""
    adj = [None] * 5
    for c in range(5):
        adj[c] = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[i * n + j]
            adj[c][i] |= 1 << j
            adj[c][j] |= 1 << i
    return adj


def _count_c4s(adj, n):
    """Total number of C4s across all colour classes (each counted once)."""
    total = 0
    for c in range(5):
        A = adj[c]
        for i in range(n):
            for j in range(i + 1, n):
                if A[i] & (1 << j):
                    total += (A[i] & A[j]).bit_count()
    return total


def _violations(adj, n):
    """Return list of (i, j, c) where colour c on edge (i,j) participates in a C4."""
    out = []
    for c in range(5):
        A = adj[c]
        for i in range(n):
            for j in range(i + 1, n):
                if A[i] & (1 << j):
                    if (A[i] & A[j]).bit_count() > 0:
                        out.append((i, j, c))
    return out


def _delta_if_remove(adj, i, j, c):
    """C4s removed if edge (i,j) is taken out of colour c."""
    A = adj[c]
    return (A[i] & A[j]).bit_count()


def _delta_if_add(adj, i, j, c):
    """C4s created if edge (i,j) is added to colour c (currently absent)."""
    A = adj[c]
    return (A[i] & A[j]).bit_count()


def _try_recolor(adj, n, viols, budget):
    """Try to recolor violating edges to reduce C4 count. Returns (changed, new_violation_count)."""
    viols = list(viols)
    random.shuffle(viols)
    changed = False
    for (i, j, c) in viols:
        if time.time() > budget:
            break
        if not (adj[c][i] & (1 << j)):
            continue  # already moved
        removed = _delta_if_remove(adj, i, j, c)
        if removed == 0:
            continue
        best_gain = removed
        best_c = -1
        for c2 in range(5):
            if c2 == c:
                continue
            if adj[c2][i] & (1 << j):
                continue  # edge already in c2
            created = _delta_if_add(adj, i, j, c2)
            gain = removed - created
            if gain > best_gain:
                best_gain = gain
                best_c = c2
        if best_c == -1:
            continue
        # apply
        adj[c][i] &= ~(1 << j)
        adj[c][j] &= ~(1 << i)
        adj[best_c][i] |= 1 << j
        adj[best_c][j] |= 1 << i
        changed = True
    return changed


def _random_recolor(adj, n, budget):
    """Randomly recolor a few edges to escape local minima."""
    for _ in range(3):
        if time.time() > budget:
            return
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        c = random.randrange(5)
        if adj[c][i] & (1 << j):
            for c2 in range(5):
                if c2 == c:
                    continue
                if not (adj[c2][i] & (1 << j)):
                    adj[c][i] &= ~(1 << j)
                    adj[c][j] &= ~(1 << i)
                    adj[c2][i] |= 1 << j
                    adj[c2][j] |= 1 << i
                    break


def _grow(adj, n, budget):
    """Try to add vertex n by finding a colouring for its n incident edges."""
    new_n = n + 1
    if new_n > 29:
        return None
    # candidate: new vertex index n
    # For each existing vertex i, pick a colour for edge (i, n)
    # Constraint: for any pair (i, j), the set of common neighbours of i and j
    # in the new colour must not include both i and j already... actually we need
    # that adding (i,n) in colour c doesn't create a C4.
    # A C4 is created by adding edge (i,n) in colour c iff there exists j
    # such that both (i,j) and (j,n) are in colour c and i,j have a common
    # neighbour in colour c other than each other... simpler: after adding,
    # check (A'[i] & A'[n]).bit_count() == 0 for all i.
    # We do a greedy/backtracking assignment.
    # For vertex i, colour c is forbidden if there exists j<i with edge(i,j)=c
    # and edge(j, new) = c and they share a common neighbour in c.
    # We assign colours to edges (0,new),(1,new),...,(n-1,new).
    # Use backtracking with constraint propagation.
    assign = [-1] * n  # assign[i] = colour of edge (i, new)
    # Precompute: for each i, which colours are "safe" given current partial assign
    # A colour c is unsafe for edge (i,new) if there exists j with assign[j]==c
    # such that (adj[c][i] & adj[c][j] & ~(1<<i) & ~(1<<j)).bit_count() > 0
    # (i and j share a common neighbour in colour c).
    # We'll do DFS.
    def safe(i, c):
        Ai = adj[c][i]
        for j in range(i):
            if assign[j] == c:
                # common neighbours of i and j in colour c
                if (Ai & adj[c][j]).bit_count() > 0:
                    return False
        return True

    def dfs(i):
        if i == n:
            return True
        if time.time() > budget:
            return False
        for c in range(5):
            if safe(i, c):
                assign[i] = c
                if dfs(i + 1):
                    return True
                assign[i] = -1
        return False

    if dfs(0):
        # apply
        new_adj = [ [a | 0 for a in A] for A in adj ]
        # extend each colour's adj list by one element
        for c in range(5):
            new_adj[c].append(0)
        for i in range(n):
            c = assign[i]
            new_adj[c][i] |= 1 << n
            new_adj[c][n] |= 1 << i
        return new_adj
    return None


def _seed_n11():
    n = 11
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return n, colors


def construct():
    t0 = time.time()
    deadline = _DEADLINE
    random.seed(12345)

    # Start from the known valid K11
    n, colors = _seed_n11()
    adj = _build_adj(colors, n)
    best_n = n
    best_colors = list(colors)
    best_c4 = _count_c4s(adj, n)

    # Phase 1: grow from 11 toward larger n using the grow routine
    # Try multiple random seeds for the grow backtracking to find different embeddings
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        # Re-seed random for variety in grow's tie-breaking (grow is deterministic
        # given adj, but we can perturb by trying to grow from slightly different
        # starting colourings obtained via random recolors that keep validity)
        # Simpler: just try grow; if it fails, do a random recolor that preserves
        # no-C4 (i.e., only recolor if it doesn't create a C4) to vary the structure.
        new_adj = _grow(adj, n, deadline - 5)
        if new_adj is not None:
            n = n + 1
            # rebuild colors from adj
            colors = [0] * (n * n)
            for c in range(5):
                A = new_adj[c]
                for i in range(n):
                    for j in range(i + 1, n):
                        if A[i] & (1 << j):
                            colors[i * n + j] = c
            adj = new_adj
            best_n = n
            best_colors = list(colors)
            best_c4 = 0
        else:
            # Try to perturb current valid coloring to enable future growth
            # Random recolor that preserves C4-freeness
            viols = _violations(adj, n)
            if not viols:
                _random_recolor_safe(adj, n, deadline - 5)
            # occasionally restart grow from scratch with a fresh random K11
            if attempt % 50 == 49:
                # regenerate K11 with a random cyclic shift to vary structure
                pass

    # Phase 2: if we have a valid coloring of size n, try to polish/grow further
    # with a more aggressive local search: keep trying grow, and if stuck,
    # do violation-free random recolors to explore.
    for it in range(5000):
        if time.time() > deadline - 5:
            break
        if n >= best_n and _count_c4s(adj, n) == 0:
            new_adj = _grow(adj, n, deadline - 5)
            if new_adj is not None:
                n = n + 1
                colors = [0] * (n * n)
                for c in range(5):
                    A = new_adj[c]
                    for i in range(n):
                        for j in range(i + 1, n):
                            if A[i] & (1 << j):
                                colors[i * n + j] = c
                adj = new_adj
                best_n = n
                best_colors = list(colors)
                best_c4 = 0
                continue
        # If current coloring has violations (shouldn't if we only grow valid ones),
        # fix them
        viols = _violations(adj, n)
        if viols:
            changed = _try_recolor(adj, n, viols, deadline - 5)
            if changed:
                viols = _violations(adj, n)
                if not viols:
                    # valid again, update best
                    colors = [0] * (n * n)
                    for c in range(5):
                        A = adj[c]
                        for i in range(n):
                            for j in range(i + 1, n):
                                if A[i] & (1 << j):
                                    colors[i * n + j] = c
                    best_n = n
                    best_colors = list(colors)
                    best_c4 = 0
        else:
            # valid; try random recolor to explore
            _random_recolor_safe(adj, n, deadline - 5)
            # after recolor, check validity
            viols = _violations(adj, n)
            if viols:
                # revert by rebuilding from best_colors? We don't have a stack.
                # Instead, just try to fix.
                _try_recolor(adj, n, viols, deadline - 5)
                viols = _violations(adj, n)
                if not viols:
                    colors = [0] * (n * n)
                    for c in range(5):
                        A = adj[c]
                        for i in range(n):
                            for j in range(i + 1, n):
                                if A[i] & (1 << j):
                                    colors[i * n + j] = c
                    best_n = n
                    best_colors = list(colors)
                    best_c4 = 0

    # Final: ensure best_colors is valid and of size best_n
    # Rebuild a clean colors list for best_n
    # We'll just return best_n and best_colors
    # But best_colors might be stale if n changed; ensure consistency
    # Reconstruct colors from a fresh adj built from best_colors to be safe
    # Actually best_colors was updated whenever we had a valid state of size n.
    # Verify best_colors is valid
    adj_check = _build_adj(best_colors, best_n)
    if _count_c4s(adj_check, best_n) != 0:
        # fallback to seed
        best_n, best_colors = _seed_n11()

    return {'n': best_n, 'colors': list(best_colors)}


def _random_recolor_safe(adj, n, budget):
    """Randomly recolor edges only if it doesn't create a C4."""
    for _ in range(5):
        if time.time() > budget:
            return
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        c = random.randrange(5)
        if not (adj[c][i] & (1 << j)):
            continue  # edge not in c
        # find another colour to move to
        for c2 in range(5):
            if c2 == c:
                continue
            if adj[c2][i] & (1 << j):
                continue
            # check if moving creates a C4 in c2
            if _delta_if_add(adj, i, j, c2) > 0:
                continue
            # safe
            adj[c][i] &= ~(1 << j)
            adj[c][j] &= ~(1 << i)
            adj[c2][i] |= 1 << j
            adj[c2][j] |= 1 << i
            break
