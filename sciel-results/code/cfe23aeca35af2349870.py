"""Construct a 5-colouring of K_n without monochromatic C4, maximizing n.

Strategy:
  1. Build a deterministic 5-colouring of K_25 using the affine plane AG(2,5).
     Vertices are points (x, y) in F_5^2.  For each unordered pair {P, Q},
     compute the slope of the line PQ (with a special value for vertical lines).
     Map the 6 possible slopes to 5 colours via a bijection that preserves
     the C4-free property (in AG(2,5), two distinct lines meet in at most one
     point, so any two vertices share at most one common neighbour per slope-class).
     Actually, in AG(2,5), for a fixed slope, the graph where edges connect
     pairs with that slope is a union of cycles (each line of that slope is a
     K_5).  Wait – that's not C4-free because K_5 contains C4.

     Let me reconsider. The correct construction: colour by the *direction*
     of the vector Q - P in F_5^2, normalised.  There are (5^2 - 1)/(5 - 1) = 6
     directions.  We need 5 colours, so we merge two directions.  But merging
     two directions may create C4s.

     Alternative: Use the known Lazebnik-Woldar construction which gives n=26.
     A simpler approach: start with K_11 (cyclic), then use a randomised
     local-search / simulated annealing to extend to larger n, verifying C4-freeness.

     Given time constraints, I'll implement:
       - A verified K_25 construction via finite field (if it works).
       - A randomised hill-climbing extension starting from a valid small graph
         to try to reach n=26, 27, 28.

     For the finite field construction, let me use a different approach:
     Colour (i, j) based on (j - i) mod 25 in a structured way.  The cyclic
     group Z_25 doesn't directly give 5 C4-free classes easily.

     Actually, the cleanest known construction for R_5(C4) uses the affine
     plane AG(2,5).  Label the 25 points.  The 6 parallel classes of lines
     give 6 sets of disjoint K_5's.  Each K_5 contains many C4s, so we can't
     just use one slope class as one colour.

     Let me use a different strategy: 
     - Use the known result that the affine plane AG(2,5) yields a 5-colouring
       where colour c connects points whose difference vector has a specific
       property.  One standard approach: for each of the 5 non-vertical slopes
       plus vertical, assign to 5 colours by grouping.

     Given the complexity, let me implement a robust randomised search:
       1. Start with a valid colouring of K_11 (cyclic).
       2. Use simulated annealing / random local search to add vertices one
          at a time, or to optimise the colouring for a fixed n, trying to
          find valid colourings for n = 25, 26, 27, 28.

     For speed and reliability, I'll:
       a) First try a deterministic construction for n=25.
       b) Then use randomised search with a deadline to try to extend to 26+.

     For the deterministic n=25: I'll use the following construction based on
     the finite field F_25.  Label vertices 0..24.  For edge (i,j) with i<j,
     compute d = (j - i) mod 25.  Map d to a colour via a function that
     ensures C4-freeness.

     Actually, let me just use a well-known valid construction:
     The cyclic group Z_25 with differences mapped via a specific 5-fold
     partition.  The set of quadratic residues mod 25 (nonzero) has size 10,
     split into 5 pairs... this is getting complicated.

     Let me use a simpler, verified approach: randomised local search with
     a tight deadline, starting from random initial colourings and trying to
     fix C4 violations.  I'll aim for n=25 first (known achievable), then
     attempt 26.

     Implementation plan:
       - For a given n, generate a random 5-colouring.
       - Iteratively find a monochromatic C4 and recolour one of its edges
         to a different colour.  Repeat until no C4s remain or time runs out.
       - Use a deadline guard.
       - Try n = 28, 27, 26, 25 in order, returning the first valid one found
         within the time budget.
"""

import time
import random
import sys

# Deadline: 140 seconds (10s safety margin before 150s kill)
_DEADLINE = 140.0


def _time_left():
    return _DEADLINE - time.time()


def _has_c4(colors, n):
    """Check if the 5-colouring of K_n contains a monochromatic C4.
    
    colors is a list of length n*(n-1)//2, in lexicographic order.
    Returns True if a C4 exists, False otherwise.
    """
    # Build adjacency lists per colour
    adj = [[] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            idx += 1
            adj[c].append((i, j))
    
    # For each colour, build adjacency sets and check for C4
    for c in range(5):
        # Build set of edges for fast lookup
        edge_set = set(adj[c])
        # Build adjacency list
        nbrs = [[] for _ in range(n)]
        for i, j in adj[c]:
            nbrs[i].append(j)
            nbrs[j].append(i)
        # For each pair of vertices, check if they share 2+ common neighbours
        for i in range(n):
            for j in range(i + 1, n):
                # Count common neighbours
                common = 0
                for k in nbrs[i]:
                    if k != j and k in nbrs[j]:
                        common += 1
                        if common >= 2:
                            return True
                # This is O(n^3) which is fine for n <= 28
    return False


def _has_c4_fast(colors, n):
    """Faster C4 detection using bitsets via Python integers."""
    # For each colour c, build for each vertex i, a bitset of its neighbours
    nbrs = [[0] * 5 for _ in range(n)]  # nbrs[i][c] = bitset of colour-c neighbours of i
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            idx += 1
            nbrs[i][c] |= (1 << j)
            nbrs[j][c] |= (1 << i)
    
    for c in range(5):
        # For each pair (i, j) with i < j, check common neighbours
        for i in range(n):
            for j in range(i + 1, n):
                common = nbrs[i][c] & nbrs[j][c]
                # Count bits in common
                if common & (common - 1):  # at least 2 bits set
                    return True
    return False


def _count_c4s(colors, n):
    """Count total number of monochromatic C4s (each C4 counted once per pair of opposite vertices)."""
    total = 0
    nbrs = [[0] * 5 for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            idx += 1
            nbrs[i][c] |= (1 << j)
            nbrs[j][c] |= (1 << i)
    
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                common = nbrs[i][c] & nbrs[j][c]
                # Number of common neighbours
                cnt = bin(common).count('1')
                if cnt >= 2:
                    total += cnt * (cnt - 1) // 2
    return total


def _local_search(n, time_budget):
    """Run local search to find a valid 5-colouring of K_n.
    
    Returns a valid colouring list or None if not found in time.
    """
    start = time.time()
    deadline = start + time_budget
    
    # Try multiple random restarts
    for attempt in range(1000):
        if time.time() >= deadline:
            break
        
        # Random initial colouring
        colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
        
        # If already valid, return
        if not _has_c4_fast(colors, n):
            return colors
        
        # Local search: iteratively fix C4 violations
        for step in range(50000):
            if time.time() >= deadline:
                break
            
            # Find a C4 violation
            # We'll find one by checking pairs
            found = False
            nbrs = [[0] * 5 for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    idx += 1
                    nbrs[i][c] |= (1 << j)
                    nbrs[j][c] |= (1 << i)
            
            # Find first C4
            c_found = -1
            i_found = -1
            j_found = -1
            for c in range(5):
                for i in range(n):
                    for j in range(i + 1, n):
                        common = nbrs[i][c] & nbrs[j][c]
                        if common & (common - 1):
                            c_found = c
                            i_found = i
                            j_found = j
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                return colors
            
            # Recolour a random edge involved in the C4
            # Get the common neighbours
            common = nbrs[i_found][c_found] & nbrs[j_found][c_found]
            # Pick two common neighbours k1, k2
            bits = []
            tmp = common
            while tmp:
                lsb = tmp & (-tmp)
                bits.append(lsb.bit_length() - 1)
                tmp ^= lsb
            
            k1 = bits[0]
            k2 = bits[1] if len(bits) > 1 else bits[0]
            
            # Edges in the C4: (i,k1), (k1,j), (j,k2), (k2,i)
            # Recolour one of these edges
            edges_to_recolour = []
            for a, b in [(i_found, k1), (k1, j_found), (j_found, k2), (k2, i_found)]:
                if a < b:
                    eidx = a * n - a * (a + 1) // 2 + b - a - 1
                else:
                    eidx = b * n - b * (b + 1) // 2 + a - b - 1
                edges_to_recolour.append(eidx)
            
            eidx = random.choice(edges_to_recolour)
            colors[eidx] = random.randint(0, 4)
    
    return None


def construct():
    """Construct a 5-colouring of K_n without monochromatic C4, maximizing n."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0  # 140s deadline
    
    best_n = 0
    best_colors = None
    
    # Try from largest n down to smallest
    for n in [28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if time.time() >= _DEADLINE - 5:  # 5s safety margin
            break
        
        # Time budget for this n
        remaining = _DEADLINE - 5 - time.time()
        if remaining < 2:
            break
        
        # Allocate time based on n (harder for larger n)
        if n >= 26:
            time_budget = min(remaining * 0.4, 40.0)
        elif n >= 24:
            time_budget = min(remaining * 0.3, 30.0)
        else:
            time_budget = min(remaining * 0.2, 20.0)
        
        # Try local search
        result = _local_search(n, time_budget)
        if result is not None:
            best_n = n
            best_colors = result
            break
        
        # Also try a quick random check (very fast)
        if time.time() < _DEADLINE - 10:
            for _ in range(50):
                if time.time() >= _DEADLINE - 5:
                    break
                colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
                if not _has_c4_fast(colors, n):
                    best_n = n
                    best_colors = colors
                    break
            if best_n == n:
                break
    
    # Fallback: use the known K_11 cyclic construction
    if best_n < 11 or best_colors is None:
        n = 11
        best_n = n
        best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    
    return {'n': best_n, 'colors': best_colors}
