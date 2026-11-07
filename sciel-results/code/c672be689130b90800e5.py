"""Program for 'Few hyperedges forcing three colors'.

Constructs a 3-uniform hypergraph on vertices 0..8 that is exactly
3-chromatic (no proper 2-coloring exists) using the minimum possible
number of edges.  The optimal construction is the Fano plane (the unique
2-(7,3,1) design) embedded on vertices 0..6, which is the smallest known
3-chromatic 3-uniform hypergraph (7 edges).  Vertices 7 and 8 are isolated.

The result is verified at runtime:
  * every edge is a distinct 3-subset of {0..8};
  * the supplied 3-coloring leaves every edge non-monochromatic;
  * no proper 2-coloring exists (exhaustive check over all 2^9 assignments,
    the deterministic 'Min-SAT' infeasibility test).
"""

# EVOLVE-BLOCK-START
import itertools


def _generate_fano_edges():
    """
    Generates the 7 edges of the Fano plane on vertices 0..6.
    Vertices correspond to non-zero vectors in GF(2)^3.
    Vertex i (1..7) maps to binary representation of i.
    We use indices 0..6 for the 7 non-zero vectors.
    Mapping:
    0 -> 001 (1)
    1 -> 010 (2)
    2 -> 011 (3)
    3 -> 100 (4)
    4 -> 101 (5)
    5 -> 110 (6)
    6 -> 111 (7)
    
    A line is a set of 3 vertices {u, v, w} such that u xor v xor w == 0.
    For any pair (u, v), w = u xor v.
    We iterate over all pairs to generate unique lines.
    """
    # Map index to vector value (1 to 7)
    # Index 0 -> 1, 1 -> 2, ..., 6 -> 7
    # Actually, just use the index+1 as the vector representation in GF(2)^3
    # where bit 0 is LSB.
    # 1: 001
    # 2: 010
    # 3: 011
    # 4: 100
    # 5: 101
    # 6: 110
    # 7: 111
    
    edges = set()
    for u in range(7):
        for v in range(u + 1, 7):
            # Convert indices to vector values
            val_u = u + 1
            val_v = v + 1
            val_w = val_u ^ val_v
            
            # val_w must be non-zero and distinct from val_u, val_v
            # Since u != v, val_u != val_v, so val_w != 0.
            # val_w == val_u implies val_v == 0, impossible.
            # val_w == val_v implies val_u == 0, impossible.
            
            # Find the index of val_w
            # val_w is in 1..7
            idx_w = val_w - 1
            
            # Ensure we form a canonical tuple (sorted)
            line = tuple(sorted([u, v, idx_w]))
            edges.add(line)
            
    # Convert to list of lists for return
    return [list(e) for e in sorted(edges)]


def _fano_colors():
    """
    Returns a valid 3-coloring for the Fano plane vertices 0..6.
    Vertices 7, 8 are isolated and colored 0.
    """
    # Coloring found via backtracking/checking:
    # 0:0, 1:1, 2:2
    # 3:0, 4:1, 5:2
    # 6:0
    return [0, 1, 2, 0, 1, 2, 0, 0, 0]


def _edge_is_nonmonochromatic(edge, colors):
    c = {colors[v] for v in edge}
    return len(c) >= 2


def _has_proper_2coloring(edges):
    # Exhaustive deterministic check: does a 2-coloring exist in which no
    # edge is monochromatic?
    n = 9
    for bits in range(1 << n):
        ok = True
        for e in edges:
            b0 = (bits >> e[0]) & 1
            b1 = (bits >> e[1]) & 1
            b2 = (bits >> e[2]) & 1
            if b0 == b1 == b2:
                ok = False
                break
        if ok:
            return True
    return False


def construct():
    edges = _generate_fano_edges()
    colors = list(_fano_colors())

    # --- runtime verification (deterministic, cheap) ---
    # distinct triples on 0..8
    assert len(set(map(tuple, edges))) == len(edges)
    for e in edges:
        assert len(set(e)) == 3
        assert all(isinstance(v, int) and 0 <= v <= 8 for v in e)
    # 3-coloring: nine labels from 0..2
    assert len(colors) == 9
    assert all(isinstance(c, int) and c in (0, 1, 2) for c in colors)
    # every edge non-monochromatic under the 3-coloring
    for e in edges:
        assert _edge_is_nonmonochromatic(e, colors)
    # exactly three-chromatic: no proper 2-coloring exists
    assert not _has_proper_2coloring(edges)

    return {'edges': edges, 'colors': colors}
# EVOLVE-BLOCK-END
