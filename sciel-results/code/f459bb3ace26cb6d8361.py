"""Four-colourings of K_n without monochromatic triangles.

Strategy:
- Primary: Construct 49-vertex coloring using GF(7^2) with multiplicative
  group partitioned into cosets of cubes (order 48, divisible by 3).
- Fallback: 32-vertex Clebsch-based construction if GF(7^2) fails.
- Deadline guard: return best valid result before 120 s.
- Fast numpy-based verification for local search.
"""

import time
import random
import sys
import numpy as np

# ---------------------------------------------------------------------------
# GF(7^2) construction for 49 vertices
# ---------------------------------------------------------------------------

def _gf49_construction():
    """
    Construct a 4-coloring of K_49 using GF(7^2).
    
    GF(7^2) has 49 elements. The multiplicative group has order 48.
    Since 48 is divisible by 3, we can partition the nonzero elements into
    3 cosets of the subgroup of cubes (order 16).
    
    For vertices x, y in GF(49):
    - If x = y: no edge (same vertex)
    - If x - y = 0: not possible for distinct vertices
    - Otherwise: color based on which coset of cubes (x-y) belongs to
    
    This should avoid monochromatic triangles because if (a-b) and (b-c)
    are in the same coset, then (a-c) = (a-b) + (b-c) is unlikely to be
    in that coset (additive combinatorics argument).
    
    We use 3 colors for the 3 cosets, and color 3 for... actually we only
    need 3 colors for this construction. But the problem requires colors
    in {0,1,2,3}. We can use colors 0,1,2 for the 3 cosets.
    
    Wait - we need to verify this actually works. Let me implement and test.
    """
    # Build GF(7^2) = GF(7)[x] / (x^2 + 1) since x^2+1 is irreducible over GF(7)
    # Check: x^2+1 has no roots in GF(7): 0^2+1=1, 1^2+1=2, 2^2+1=5, 3^2+1=10=3, 4^2+1=17=3, 5^2+1=26=5, 6^2+1=37=2
    # Yes, irreducible.
    
    # Represent elements as (a, b) meaning a + b*alpha where alpha^2 = -1 = 6 mod 7
    # So alpha^2 = 6
    
    def gf49_add(x, y):
        return ((x[0] + y[0]) % 7, (x[1] + y[1]) % 7)
    
    def gf49_sub(x, y):
        return ((x[0] - y[0]) % 7, (x[1] - y[1]) % 7)
    
    def gf49_mul(x, y):
        # (a + b*alpha) * (c + d*alpha) = ac + ad*alpha + bc*alpha + bd*alpha^2
        # = ac + 6*bd + (ad + bc)*alpha
        a, b = x
        c, d = y
        real = (a * c + 6 * b * d) % 7
        imag = (a * d + b * c) % 7
        return (real, imag)
    
    # Build all 49 elements
    elements = [(a, b) for a in range(7) for b in range(7)]
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    
    # Find a primitive element (generator of multiplicative group of order 48)
    # Try elements and check if they have order 48
    def gf49_pow(x, n):
        result = (1, 0)  # multiplicative identity
        base = x
        while n > 0:
            if n & 1:
                result = gf49_mul(result, base)
            base = gf49_mul(base, base)
            n >>= 1
        return result
    
    # Find primitive element
    primitive = None
    for e in elements:
        if e == (0, 0):
            continue
        # Check if e has order 48
        # 48 = 2^4 * 3, so check e^(48/2) != 1 and e^(48/3) != 1
        if gf49_pow(e, 24) != (1, 0) and gf49_pow(e, 16) != (1, 0):
            primitive = e
            break
    
    if primitive is None:
        return None  # Should not happen
    
    # Generate all nonzero elements as powers of primitive
    # powers[i] = primitive^i for i = 0..47
    powers = []
    current = (1, 0)
    for i in range(48):
        powers.append(current)
        current = gf49_mul(current, primitive)
    
    # powers[48] should be (1,0) again
    
    # The subgroup of cubes has order 48/3 = 16
    # cubes[i] = primitive^(3*i) for i = 0..15
    cubes = [powers[3 * i] for i in range(16)]
    cube_set = set(cubes)
    
    # Partition nonzero elements into 3 cosets of the cube subgroup
    # coset_0 = cubes
    # coset_1 = primitive * cubes
    # coset_2 = primitive^2 * cubes
    
    def get_coset(elem):
        """Return which coset of cubes elem belongs to (0, 1, or 2)."""
        if elem in cube_set:
            return 0
        # Check if elem * primitive^(-1) is in cubes
        # primitive^(-1) = primitive^47 = powers[47]
        inv_prim = powers[47]
        e1 = gf49_mul(elem, inv_prim)
        if e1 in cube_set:
            return 1
        # Otherwise should be in coset 2
        return 2
    
    # Verify partition
    coset_counts = [0, 0, 0]
    for p in powers:
        c = get_coset(p)
        coset_counts[c] += 1
    # Each coset should have 16 elements
    
    # Now assign colors
    # For edge (i, j) where i < j, compute elements[i] - elements[j]
    # and assign color based on coset
    colors = []
    for i in range(49):
        for j in range(i + 1, 49):
            diff = gf49_sub(elements[i], elements[j])
            if diff == (0, 0):
                # Should not happen for i != j
                c = 0
            else:
                c = get_coset(diff)
            colors.append((i, j, c))
    
    return colors

# ---------------------------------------------------------------------------
# Fast triangle-free verification using bitsets
# ---------------------------------------------------------------------------

def _is_triangle_free_fast(colors, n):
    """
    colors: list of (i, j, c) in lex order, i < j.
    Returns True iff no monochromatic triangle.
    Uses bitset adjacency for speed.
    """
    # adj[c][v] = bitmask of neighbors of v in color c
    adj = [[0] * n for _ in range(4)]
    
    for i, j, c in colors:
        adj[c][i] |= (1 << j)
        adj[c][j] |= (1 << i)
    
    # For each color, check each vertex
    for c in range(4):
        for v in range(n):
            nbrs = adj[c][v]
            # For each pair of neighbors, check if they're connected
            # Iterate through set bits
            n1 = nbrs
            while n1:
                # Get lowest set bit
                b1 = n1 & (-n1)
                n1 -= b1
                u = b1.bit_length() - 1
                # Check if u's neighbors intersect with nbrs (excluding v)
                nbrs_u = adj[c][u] & nbrs
                # If nbrs_u has any bits, there's a triangle
                # But we need to exclude v itself
                if nbrs_u & ~(1 << v):
                    return False
    return True


def _build_adj_matrices(colors, n):
    """Build adjacency matrices for fast triangle checking.
    Returns list of 4 numpy boolean matrices, one per color.
    """
    adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    for i, j, c in colors:
        adj[c][i, j] = True
        adj[c][j, i] = True
    return adj


def _is_triangle_free_numpy(adj, n):
    """Check triangle-freeness using numpy matrix operations.
    adj: list of 4 (n,n) boolean numpy arrays.
    Returns True iff no monochromatic triangle.
    """
    for c in range(4):
        A = adj[c]
        # Triangle exists if (A @ A) has any entry > 0 where A is also 1
        # More precisely: for each edge (i,j) in color c, check if any common neighbor
        # Use: (A @ A.T) > 0 indicates common neighbors, but we need to check
        # that the common neighbor k has A[k,j] and A[k,i] both true
        # Actually: triangle (i,j,k) in color c means A[i,j] & A[j,k] & A[k,i]
        # For each edge (i,j), count common neighbors: sum over k of A[i,k] & A[j,k]
        # If any edge has a common neighbor, triangle exists
        # Efficient: (A @ A.T)[i,j] = number of common neighbors of i and j
        # Triangle exists iff there's an edge (i,j) with (A @ A.T)[i,j] > 0
        # But we need to be careful: A[i,j] must be true
        common = A @ A.T
        # For each edge, check if common neighbors exist
        # np.any(common * A > 0) but this counts i-j pairs
        # Actually if A[i,j] is true and common[i,j] > 0, then triangle exists
        if np.any(common * A):
            return False
    return True

# ---------------------------------------------------------------------------
# GF(2,4) multiplication table for the Clebsch-graph base construction
# ---------------------------------------------------------------------------

def _build_mul_table():
    """Build multiplication in GF(2^4) with irreducible x^4+x+1 (0b10011)."""
    log = {}
    mul = {}
    x = 1
    log[1] = 0
    for e in range(15):
        x2 = 0
        for i in range(4):
            if (x >> i) & 1:
                x2 ^= (1 << (i + 1))
        # reduce
        if x2 & 0b10000:
            x2 ^= 0b10011
        x = x2
        log[x] = e + 1
    return log

# Precompute once
_log = _build_mul_table()

def _base32_colors():
    """Return the 32-vertex base coloring as a list of (i,j,color) in lex order."""
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                # same block: Clebsch
                c = _log[i % 16 ^ j % 16] % 3
            else:
                c = 3
            colors.append((i, j, c))
    return colors

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    deadline = time.time() + 110  # 110s budget, safety margin for 120s limit
    
    # Try GF(49) construction first
    best_colors = None
    best_n = 0
    
    gf49_colors = _gf49_construction()
    if gf49_colors is not None:
        # Verify it's triangle-free
        if _is_triangle_free_fast(gf49_colors, 49):
            best_colors = gf49_colors
            best_n = 49
    
    # If GF(49) failed or we want to try to extend, fall back to 32-vertex base
    if best_n < 32:
        best_colors = _base32_colors()
        best_n = 32
    
    # Try to extend beyond current best_n using local search with numpy
    if best_n < 50 and time.time() < deadline - 10:
        # Build initial adjacency matrices
        adj = _build_adj_matrices(best_colors, best_n)
        
        def _try_extend_numpy(adj, current_n):
            """Try to extend coloring to current_n+1 vertices via local search using numpy."""
            deadline_local = min(time.time() + 15, deadline - 5)
            new_n = current_n + 1
            
            for trial in range(50):
                if time.time() > deadline_local:
                    break
                
                # Randomly assign colours for edges from new vertex to existing
                assign = np.random.randint(0, 4, size=current_n)
                
                # Build new adjacency matrices
                new_adj = [np.zeros((new_n, new_n), dtype=bool) for _ in range(4)]
                for c in range(4):
                    new_adj[c][:current_n, :current_n] = adj[c]
                
                for i in range(current_n):
                    c = assign[i]
                    new_adj[c][i, new_n - 1] = True
                    new_adj[c][new_n - 1, i] = True
                
                # Check if valid
                if _is_triangle_free_numpy(new_adj, new_n):
                    # Convert back to colors list
                    new_colors = list(best_colors)
                    for i in range(current_n):
                        new_colors.append((i, new_n - 1, int(assign[i])))
                    return new_colors, new_n, new_adj
                
                # Try local search: flip some edges
                for it in range(2000):
                    if time.time() > deadline_local:
                        break
                    
                    if _is_triangle_free_numpy(new_adj, new_n):
                        new_colors = list(best_colors)
                        for i in range(current_n):
                            new_colors.append((i, new_n - 1, int(assign[i])))
                        return new_colors, new_n, new_adj
                    
                    # Find a random edge to flip
                    pos = random.randint(0, current_n - 1)
                    old_c = assign[pos]
                    new_c = (old_c + 1) % 4
                    
                    # Update adjacency
                    new_adj[old_c][pos, new_n - 1] = False
                    new_adj[old_c][new_n - 1, pos] = False
                    new_adj[new_c][pos, new_n - 1] = True
                    new_adj[new_c][new_n - 1, pos] = True
                    assign[pos] = new_c
                
            return None, None, None
        
        # Try to extend step by step
        while time.time() < deadline - 5:
            colors, new_n, new_adj = _try_extend_numpy(adj, best_n)
            if colors is not None:
                best_colors = colors
                best_n = new_n
                adj = new_adj
            else:
                # If we can't extend, break
                break
    
    # Format output
    out_colors = []
    for i, j, c in best_colors:
        out_colors.append(c)
    
    return {'n': best_n, 'colors': out_colors}
