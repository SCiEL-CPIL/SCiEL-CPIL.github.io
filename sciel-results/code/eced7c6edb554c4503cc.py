"""HP lattice protein folding: Fragment re-insertion + Reversal + Kink moves with ILS."""
# EVOLVE-BLOCK-START
import time
import random

_T0 = time.time()
_DEADLINE = 205.0  # 210s wall-clock limit, 5s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy_fast(seq, chain):
    """Fast energy calculation."""
    # Precompute H indices
    h_indices = [i for i, c in enumerate(seq) if c == 'H']
    if not h_indices:
        return 0
    
    # Position to index map
    where = {p: i for i, p in enumerate(chain)}
    
    contacts = 0
    for i in h_indices:
        x, y = chain[i]
        # Check 4 neighbors
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = where.get((x + dx, y + dy))
            if j is not None and j > i and seq[j] == 'H':
                contacts += 1
    return -contacts


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n < 2:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            dx = abs(chain[i][0] - chain[i - 1][0])
            dy = abs(chain[i][1] - chain[i - 1][1])
            if dx + dy != 1:
                return False
    return True


def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        x, y = chain[-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                sx, sy = site
                for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (sx + adx, sy + ady)
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and sequence[idx] == "H":
                            gain += 1
            # Free space heuristic
            free = 0
            sx, sy = site
            for adx, ady in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (sx + adx, sy + ady) not in occupied:
                    free += 1
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _try_reinsert_fragment(chain, sequence, rng, max_frag_len=5):
    """
    Attempt to remove a fragment and re-insert it by finding a new path.
    Returns new chain if successful, else None.
    """
    n = len(chain)
    if n < 4:
        return None
    
    # Pick fragment length and start
    L = rng.randint(2, min(max_frag_len, n - 2))
    a = rng.randint(0, n - L - 1)
    b = a + L  # Fragment is chain[a:b]
    
    # Identify anchors
    # The fragment connects chain[a-1] to chain[b] (if a>0 and b<n)
    # We remove chain[a:b].
    # The new gap is between chain[a-1] (or start) and chain[b] (or end).
    # Actually, the fragment *is* the path between the two neighbors.
    # Neighbor 1: chain[a-1] if a>0 else None (if a==0, the fragment starts at 0, so it's anchored at the end? No, if a==0, the fragment is the prefix. We can't re-insert a prefix easily without moving the whole start. Let's restrict a > 0 and b < n to keep it internal.)
    if a == 0 or b == n:
        return None
        
    start_anchor = chain[a-1]
    end_anchor = chain[b]
    
    # The fragment to replace is chain[a:b]
    # We need to find a new path of length L from start_anchor to end_anchor
    # that does not collide with the rest of the chain (chain[:a-1] + chain[b:])
    
    blocked = set(chain[:a-1]) | set(chain[b:])
    
    # Randomized DFS to find path
    # We want a path p0...pL where p0=start_anchor, pL=end_anchor
    # p_i not in blocked for 1 <= i <= L-1
    # p_L = end_anchor is allowed to be in blocked? No, end_anchor is in chain[b:], which is in blocked.
    # So we must ensure the path ends at end_anchor, but intermediate nodes are not in blocked.
    
    def dfs(path, curr):
        if len(path) == L:
            return curr == end_anchor
        if len(path) > L:
            return False
            
        x, y = curr
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if len(path) < L - 1:
                if nxt in blocked:
                    continue
            else:
                # Last step must go to end_anchor
                if nxt != end_anchor:
                    continue
                # Also check if end_anchor is blocked? It is in blocked set.
                # But we are *ending* there, so it's the target.
                # However, if L=1, path is [start, end].
                # If L>1, intermediate nodes must not be blocked.
                pass
            
            # Check if we've already visited nxt in this path?
            # The path is a walk, it can self-intersect? No, the final chain must be self-avoiding.
            # So the new path must not intersect itself or the blocked set.
            if nxt in path:
                continue
                
            path.append(nxt)
            if dfs(path, nxt):
                return True
            path.pop()
        return False
    
    # Start DFS
    path = [start_anchor]
    # Shuffle directions to randomize
    # The DFS above is deterministic order. Let's make it random.
    
    def dfs_random(path, curr, depth):
        if depth == L:
            return curr == end_anchor
        if depth > L:
            return False
            
        x, y = curr
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        rng.shuffle(dirs)
        for dx, dy in dirs:
            nxt = (x + dx, y + dy)
            if depth < L - 1:
                if nxt in blocked:
                    continue
            else:
                if nxt != end_anchor:
                    continue
            
            if nxt in path:
                continue
                
            path.append(nxt)
            if dfs_random(path, nxt, depth + 1):
                return True
            path.pop()
        return False
    
    if dfs_random(path, start_anchor, 1):
        # path is the new fragment
        new_chain = chain[:a-1] + path + chain[b:]
        # Verify validity just in case
        if _is_valid_chain(new_chain):
            return new_chain
    return None


def _reversal_move(chain, rng):
    """Reverse a random segment."""
    n = len(chain)
    if n < 4:
        return None
    a = rng.randint(0, n - 3)
    b = rng.randint(a + 2, n - 1)
    new_chain = chain[:a] + chain[a:b + 1][::-1] + chain[b + 1:]
    if _is_valid_chain(new_chain):
        return new_chain
    return None


def _kink_move(chain, rng):
    """Try to flip a 3-residue segment (kink)."""
    n = len(chain)
    if n < 3:
        return None
    i = rng.randint(1, n - 2)
    p_prev = chain[i-1]
    p_curr = chain[i]
    p_next = chain[i+1]
    
    # Try the two other possible positions for p_curr
    # p_curr is adjacent to p_prev and p_next.
    # The other two neighbors of p_prev that are adjacent to p_next?
    # Actually, if p_prev and p_next are fixed, p_curr can be at one of 2 positions (if p_prev and p_next are distance 2).
    # If p_prev and p_next are distance 1 (straight), p_curr must be at the current pos.
    # If p_prev and p_next are distance 2 (kink), p_curr can be at the "other" corner.
    
    dx_prev = p_prev[0] - p_curr[0]
    dy_prev = p_prev[1] - p_curr[1]
    dx_next = p_next[0] - p_curr[0]
    dy_next = p_next[1] - p_curr[1]
    
    # The vector from curr to prev is (-dx_prev, -dy_prev).
    # The vector from curr to next is (-dx_next, -dy_next).
    # If they are orthogonal, there is another position.
    # Let's just check the 4 neighbors of p_curr.
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        new_p = (p_curr[0] + dx, p_curr[1] + dy)
        if new_p == p_prev or new_p == p_next:
            continue
        # Check if new_p is adjacent to p_prev and p_next
        if abs(new_p[0] - p_prev[0]) + abs(new_p[1] - p_prev[1]) == 1 and \
           abs(new_p[0] - p_next[0]) + abs(new_p[1] - p_next[1]) == 1:
            # Check if new_p is occupied
            # We need to check against the whole chain except p_curr
            # This is O(N).
            occupied = set(chain)
            occupied.discard(p_curr)
            if new_p not in occupied:
                new_chain = chain.copy()
                new_chain[i] = new_p
                if _is_valid_chain(new_chain):
                    return new_chain
    return None


def _local_search(sequence, chain, rng, max_iters=100):
    """Local search with multiple move types."""
    best_chain = list(chain)
    best_energy = _energy_fast(sequence, best_chain)
    n = len(best_chain)
    
    for _ in range(max_iters):
        if _remaining() < 2.0:
            break
            
        # Pick a move
        move_type = rng.choice(['reversal', 'kink', 'fragment'])
        new_chain = None
        
        if move_type == 'reversal':
            new_chain = _reversal_move(best_chain, rng)
        elif move_type == 'kink':
            new_chain = _kink_move(best_chain, rng)
        else:
            new_chain = _try_reinsert_fragment(best_chain, sequence, rng)
            
        if new_chain is None:
            continue
            
        e = _energy_fast(sequence, new_chain)
        if e < best_energy:
            best_chain = new_chain
            best_energy = e
            
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using ILS."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0  # start at 0 (unfolded)
    n = len(sequence)
    
    # Initial growth
    chain = _grow(sequence, dimension, rng)
    if chain is None:
        # Fallback
        best = [(i, 0) for i in range(n)]
        return [list(p) for p in best]
        
    e = _energy_fast(sequence, chain)
    if e < best_energy:
        best_energy = e
        best = chain
        
    # ILS Loop
    while _remaining() > 5.0:
        # Local search
        improved, imp_e = _local_search(sequence, best, rng, max_iters=50)
        if imp_e < best_energy:
            best_energy = imp_e
            best = improved
            
        # Perturbation: Random Fragment Re-insertion or Reversal
        # Try a few perturbations to escape local minima
        for _ in range(5):
            if _remaining() < 5.0:
                break
            # Pick a random move to apply to best
            if rng.random() < 0.5:
                perturbed = _reversal_move(best, rng)
            else:
                perturbed = _try_reinsert_fragment(best, sequence, rng)
                
            if perturbed is not None:
                p_e = _energy_fast(sequence, perturbed)
                # Accept if better or with some probability (Simulated Annealing idea)
                # For simplicity, accept if better or equal, or if it's a "good" worse move?
                # Let's just accept if it's better, or if we are stuck.
                if p_e <= best_energy:
                    best = perturbed
                    best_energy = p_e
                    break
        # If no improvement, try a new growth?
        # Maybe restart growth if stuck for too long?
        # Let's just continue with ILS.
        
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
