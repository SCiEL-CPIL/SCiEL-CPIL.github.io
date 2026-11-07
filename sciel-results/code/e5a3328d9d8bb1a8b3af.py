"""Deterministic optimal solution: Fano plane (7 edges) on vertices 0-6, isolated 7,8.

The Fano plane is the unique (up to isomorphism) 3-uniform hypergraph on 7 vertices
with chromatic number 3. It has exactly 7 edges. Any 3-uniform hypergraph with fewer
than 7 edges is 2-colorable, so 7 is the minimum possible edge count for a
3-chromatic triple system on any vertex set (isolated vertices don't affect chromatic
number).

This program generates the Fano plane edges from the projective geometry PG(2,2)
and includes an exhaustive verification that no 2-coloring exists.
"""

import time

def construct():
    # Deadline guard: return immediately since this is a constant-time construction
    # but we include the guard per the directive.
    start = time.time()
    
    # PG(2,2) construction:
    # Points are the 7 non-zero vectors in GF(2)^3.
    # We index them 0-6 in the order of their integer values 1-7.
    # v0=1(001), v1=2(010), v2=3(011), v3=4(100), v4=5(101), v5=6(110), v6=7(111)
    # 
    # Lines are triples of distinct points {i,j,k} such that v_i XOR v_j XOR v_k = 0.
    # Generate all such triples:
    edges = []
    for i in range(7):
        for j in range(i + 1, 7):
            for k in range(j + 1, 7):
                # v_i = i+1, v_j = j+1, v_k = k+1
                if ((i + 1) ^ (j + 1) ^ (k + 1)) == 0:
                    edges.append([i, j, k])
    
    # Should have exactly 7 edges
    assert len(edges) == 7, f"Expected 7 edges, got {len(edges)}"
    
    # A valid 3-coloring of the Fano plane.
    # Verified: each edge is non-monochromatic.
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verification: check that the provided 3-coloring is valid (no monochromatic edge)
    for edge in edges:
        c = [colors[v] for v in edge]
        assert len(set(c)) >= 2, f"Edge {edge} is monochromatic with colors {c}"
    
    # Exhaustive verification that no 2-coloring exists.
    # The Fano plane has 7 vertices (0-6). Check all 2^7 = 128 binary colorings.
    # A 2-coloring is valid if no edge is monochromatic.
    # We verify that NO such valid 2-coloring exists.
    found_2coloring = False
    for mask in range(128):
        # Binary coloring of vertices 0-6
        two_colors = [(mask >> i) & 1 for i in range(7)]
        valid = True
        for edge in edges:
            c = [two_colors[v] for v in edge]
            if len(set(c)) == 1:
                valid = False
                break
        if valid:
            found_2coloring = True
            break
    
    assert not found_2coloring, "Fano plane admits a 2-coloring; this is a bug!"
    
    # Deadline check (should be well within budget)
    assert time.time() - start < 30, "Construction took too long"
    
    # Ensure types are exactly Python int and list
    edges = [[int(v) for v in e] for e in edges]
    colors = [int(c) for c in colors]
    
    return {'edges': edges, 'colors': colors}
