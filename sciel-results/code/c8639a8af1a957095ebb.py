"""Rewritten program: dense Q3-free graph on 16 vertices via branch-and-bound edge addition."""
import time
import random
import itertools

_N = 16
_E = _N * (_N - 1) // 2  # 120
_DEADLINE = 150
_MARGIN = 20  # safety margin in seconds

# Q3 edges in 8-vertex cube (vertices 0..7, edges where XOR in {1,2,4})
_Q3_EDGES = []
for i in range(8):
    for j in range(i + 1, 8):
        if (i ^ j) in (1, 2, 4):
            _Q3_EDGES.append((i, j))

# Precompute all 120 embeddings of Q3 into 8 vertices (0..7)
# Each embedding is a tuple of 12 pairs (u, v) where u, v are in 0..7
# We represent the set of edges in a 28-bit integer for fast subset checking
# Edge (i, j) with i < j maps to bit index: i * 7 - i * (i + 1) // 2 + (j - i - 1)
# Or simpler: map (i, j) to a unique bit in 28 bits.
# Let's use a mapping function for the 8-vertex subgraph edges.

def _edge_bit_index(i, j):
    """Map edge (i, j) in 8-vertex graph (0<=i<j<=7) to a bit index 0..27"""
    # Total edges = 8*7/2 = 28
    # We can just use a lookup table or formula.
    # Formula for edge index in adjacency matrix upper triangle:
    # idx = i * (7 - i) // 2 + (j - i - 1) is not quite right for 0-indexed upper triangle of 8x8.
    # Let's just build a dictionary for the 28 edges.
    pass

# Build mapping for 8-vertex edges
_8V_EDGE_MAP = {}
_idx = 0
for i in range(8):
    for j in range(i + 1, 8):
        _8V_EDGE_MAP[(i, j)] = 1 << _idx
        _idx += 1

# Precompute the bitmask for each of the 120 Q3 embeddings
_Q3_EMBEDDINGS_MASKS = []
for perm in itertools.permutations(range(8)):
    mask = 0
    for u, v in _Q3_EDGES:
        pu, pv = perm[u], perm[v]
        if pu > pv:
            pu, pv = pv, pu
        mask |= _8V_EDGE_MAP[(pu, pv)]
    _Q3_EMBEDDINGS_MASKS.append(mask)

# All 8-vertex subsets of 16 vertices
_ALL_8SETS = list(itertools.combinations(range(_N), 8))


def _build_adj_bits(edges):
    adj = [0] * _N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def _has_q3(adj_bits):
    """Check if graph represented by adj_bits contains a Q3 subgraph.
    Uses precomputed embeddings and bitmask checks.
    """
    for subset in _ALL_8SETS:
        # Build bitmask of edges present in this 8-vertex induced subgraph
        present_mask = 0
        for i in range(8):
            for j in range(i + 1, 8):
                u, v = subset[i], subset[j]
                if (adj_bits[u] >> v) & 1:
                    # Map (i, j) to the bit index in the 8-vertex mask
                    present_mask |= _8V_EDGE_MAP[(i, j)]
        
        # Check if any Q3 embedding is a subset of present edges
        for q3_mask in _Q3_EMBEDDINGS_MASKS:
            if (present_mask & q3_mask) == q3_mask:
                return True
                
    return False


def _is_q3_free_fast(adj_bits):
    """Fast Q3-freeness check using the subset approach."""
    return not _has_q3(adj_bits)


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN
    
    # Strategy: Start with a random sparse graph and add edges greedily.
    # Or start with a known Q3-free structure.
    # Let's start with a random graph with ~50% edges and refine.
    
    random.seed(42)
    
    # Initialize with a random graph
    adj_bits = [0] * _N
    current_edges = []
    
    # Add random edges
    all_pairs = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    random.shuffle(all_pairs)
    
    # Add first 60 edges randomly to get a starting point
    for u, v in all_pairs[:60]:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        current_edges.append((u, v))
        
    # Ensure it's Q3-free (it likely is with 60 edges, but check just in case)
    # If not, remove edges until it is.
    while _has_q3(adj_bits):
        # Remove a random edge from current_edges
        if not current_edges:
            break
        idx = random.randint(0, len(current_edges) - 1)
        u, v = current_edges.pop(idx)
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)
        
    # Greedy addition of remaining edges
    # Sort remaining edges, maybe by some heuristic? Random is fine.
    remaining_edges = all_pairs[60:]
    random.shuffle(remaining_edges)
    
    for u, v in remaining_edges:
        if time.time() > deadline:
            break
            
        # Check if edge is already present
        if (adj_bits[u] >> v) & 1:
            continue
            
        # Try adding
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        
        if _is_q3_free_fast(adj_bits):
            current_edges.append((u, v))
        else:
            # Revert
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            
    # Local Search: Swaps
    # Try to swap a present edge for a missing edge to potentially allow more edges later
    # Or just try to add missing edges again in a different order?
    # A better local search: try to add a missing edge by removing a present edge that blocks it?
    # Actually, simple greedy is often good. Let's do a few passes of greedy.
    
    for _pass in range(5):
        if time.time() > deadline - 5:
            break
            
        improved = False
        # Get current missing edges
        missing = []
        for u in range(_N):
            for v in range(u + 1, _N):
                if not (adj_bits[u] >> v) & 1:
                    missing.append((u, v))
                    
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 5:
                break
            
            # Try adding
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            
            if _is_q3_free_fast(adj_bits):
                current_edges.append((u, v))
                improved = True
            else:
                # Revert
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                
        if not improved:
            break

    # Build final edge list
    edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (adj_bits[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
