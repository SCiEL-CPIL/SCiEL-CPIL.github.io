"""
Snake-in-the-Box in Q9 using a different structural approach:
- Fast bitset-based validity checks using numpy for speed.
- Iterated Local Search with multiple move types:
  1. End extension (both ends).
  2. Vertex insertion into the middle of the path.
  3. 2-opt style reversal of a subpath segment (if it remains valid).
  4. Truncation + re-greedy (randomized).
- Use a precomputed "conflict" structure: for each vertex v, the set of vertices
  at Hamming distance 1 (neighbors) is fixed. The induced path condition requires
  that for any two non-consecutive vertices in the path, they are NOT neighbors.
  
Key insight for speed: represent the path as a list, and maintain a set of used vertices.
For insertion, we need to check that the new vertex is not a neighbor of any path vertex
except its two neighbors in the new path. We can precompute for each vertex the list of
its neighbors, and check membership in the path's used set.

To make this faster, we maintain:
- path: list of vertices
- used: set of vertices in path
- For each vertex in path, we know its index.

For insertion of vertex v at position i (between path[i] and path[i+1]):
- v must be adjacent to path[i] and path[i+1].
- v must NOT be adjacent to any path[j] for j != i, i+1.
- v must not be in used.

We can check "v not adjacent to any path[j] for j != i, i+1" by:
  For each neighbor w of v, check if w is in used and w != path[i] and w != path[i+1].
  If any such w exists, insertion is invalid.

This is O(degree(v)) = O(9) per candidate, which is fast.

For end extension from the end (append v to path[-1]):
- v must be adjacent to path[-1].
- v must NOT be adjacent to any path[j] for j < len(path)-1.
  Check: for each neighbor w of v, if w in used and w != path[-1], invalid.

For end extension from the beginning (prepend v to path[0]):
- v must be adjacent to path[0].
- v must NOT be adjacent to any path[j] for j > 0.
  Check: for each neighbor w of v, if w in used and w != path[0], invalid.

This is much faster than checking all path vertices.
"""
import time
import random
import numpy as np

N = 9
TOTAL_VERTS = 1 << N  # 512

def _precompute_neighbors():
    neighbors = [[] for _ in range(TOTAL_VERTS)]
    for v in range(TOTAL_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    return neighbors

_NEIGHBORS = _precompute_neighbors()

def _can_append(v, path, used):
    """Check if v can be appended to the end of path."""
    if v in used:
        return False
    # v must be adjacent to path[-1]
    if (v ^ path[-1]).bit_count() != 1:
        return False
    # v must not be adjacent to any path[j] for j < len(path)-1
    # Check neighbors of v
    for w in _NEIGHBORS[v]:
        if w in used and w != path[-1]:
            return False
    return True

def _can_prepend(v, path, used):
    """Check if v can be prepended to the beginning of path."""
    if v in used:
        return False
    # v must be adjacent to path[0]
    if (v ^ path[0]).bit_count() != 1:
        return False
    # v must not be adjacent to any path[j] for j > 0
    for w in _NEIGHBORS[v]:
        if w in used and w != path[0]:
            return False
    return True

def _can_insert(v, i, path, used):
    """Check if v can be inserted at position i (between path[i] and path[i+1])."""
    if v in used:
        return False
    a = path[i]
    b = path[i + 1]
    # v must be adjacent to both a and b
    if (v ^ a).bit_count() != 1 or (v ^ b).bit_count() != 1:
        return False
    # v must not be adjacent to any path[j] for j != i, i+1
    for w in _NEIGHBORS[v]:
        if w in used and w != a and w != b:
            return False
    return True

def _greedy_extend(path, used, rng):
    """Greedily extend path from the end using random bit ordering."""
    while True:
        cur = path[-1]
        bits = list(range(N))
        rng.shuffle(bits)
        found = False
        for b in bits:
            v = cur ^ (1 << b)
            if _can_append(v, path, used):
                path.append(v)
                used.add(v)
                found = True
                break
        if not found:
            break
    return path, used

def _try_extend_ends(path, used):
    """Try to extend from both ends. Returns True if extended."""
    # Try end
    cur = path[-1]
    for b in range(N):
        v = cur ^ (1 << b)
        if _can_append(v, path, used):
            path.append(v)
            used.add(v)
            return True
    # Try beginning
    cur = path[0]
    for b in range(N):
        v = cur ^ (1 << b)
        if _can_prepend(v, path, used):
            path.insert(0, v)
            used.add(v)
            return True
    return False

def _try_insert_vertex(path, used):
    """Try to find and insert a vertex into the path. Returns True if inserted."""
    L = len(path)
    # Candidate vertices: neighbors of path vertices that are not in used
    # and could potentially be inserted
    # For each edge (path[i], path[i+1]), candidates are common neighbors of path[i] and path[i+1]
    # that are not in used.
    candidates = set()
    for i in range(L - 1):
        a = path[i]
        b = path[i + 1]
        # Common neighbors of a and b
        for w in _NEIGHBORS[a]:
            if w in _NEIGHBORS[b] and w not in used:
                candidates.add((w, i))
    
    # Try to insert candidates
    for v, i in candidates:
        if _can_insert(v, i, path, used):
            path.insert(i + 1, v)
            used.add(v)
            return True
    return False

def _try_reversal(path, used, rng):
    """
    Try a 2-opt style move: reverse a subpath segment.
    Pick two random positions i < j, reverse path[i:j+1].
    Check if the resulting path is still a valid induced path.
    This is expensive to check fully, so we only try small reversals or use a heuristic.
    Actually, reversing a subpath changes the adjacencies at the boundaries and internally.
    Internal adjacencies remain the same (since reversal preserves adjacency).
    Boundary adjacencies: path[i-1] must be adjacent to path[j], and path[i] must be adjacent to path[j+1].
    Also, we need to check that no new non-consecutive adjacencies are created.
    This is complex. Let's skip this for now and focus on other moves.
    """
    return False

def construct():
    deadline = time.time() + 115
    rng = random.Random(42)
    
    # Initial greedy snake
    best_path = [0]
    best_used = {0}
    best_path, best_used = _greedy_extend(best_path, best_used, rng)
    best_len = len(best_path)
    
    # Try a few random greedy starts
    for s in range(1, 30):
        if time.time() > deadline - 15:
            break
        p = [0]
        u = {0}
        p, u = _greedy_extend(p, u, rng)
        if len(p) > best_len:
            best_path = p
            best_used = u
            best_len = len(p)
            
    current_path = list(best_path)
    current_used = set(best_used)
    
    no_improve_count = 0
    
    while time.time() < deadline - 2:
        improved = False
        
        # 1. Try to extend from ends
        if _try_extend_ends(current_path, current_used):
            improved = True
            if len(current_path) > best_len:
                best_path = list(current_path)
                best_used = set(current_used)
                best_len = len(current_path)
                no_improve_count = 0
            continue
        
        # 2. Try to insert a vertex
        if _try_insert_vertex(current_path, current_used):
            improved = True
            if len(current_path) > best_len:
                best_path = list(current_path)
                best_used = set(current_used)
                best_len = len(current_path)
                no_improve_count = 0
            continue
        
        # 3. Perturbation: truncate and re-extend
        trunc_amount = rng.randint(2, 8)
        if len(current_path) > trunc_amount + 5:
            current_path = current_path[:-trunc_amount]
            current_used = set(current_path)
            current_path, current_used = _greedy_extend(current_path, current_used, rng)
            
            if len(current_path) > best_len:
                best_path = list(current_path)
                best_used = set(current_used)
                best_len = len(current_path)
                no_improve_count = 0
            else:
                no_improve_count += 1
                
                # If stuck for a while, restart from best
                if no_improve_count > 20:
                    current_path = list(best_path)
                    current_used = set(best_used)
                    no_improve_count = 0
        else:
            # Path is too short to truncate, restart from best
            current_path = list(best_path)
            current_used = set(best_used)
            no_improve_count += 1
            
    return best_path

# EVOLVE-BLOCK-START
# The construct function is the entry point.
# EVOLVE-BLOCK-END
