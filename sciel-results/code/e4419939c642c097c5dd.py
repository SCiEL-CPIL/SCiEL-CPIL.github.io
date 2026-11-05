"""
Constructs a K4-free graph on 18 vertices that minimizes the size of the largest
triangle-free induced subgraph.

Approach: Start from the complete 3-partite Turán graph T_3(18) (parts of size 6).
This graph is K4-free (any 4 vertices must have two in the same part, hence no edge
between them, so no K4). The largest triangle-free induced subgraph is the union of
two parts, size 12.

To improve, we add a small number of edges within parts. Adding an edge (u,v) within
a part means that a triangle-free set cannot contain both u and v (since they'd form
a K2, but wait – two adjacent vertices don't form a triangle). Actually, a triangle
requires 3 mutually adjacent vertices. In T_3(18), a triangle is one vertex from each
part. Adding an edge within a part doesn't directly break any triangle (triangles in
T_3(18) use vertices from different parts).

However, adding edges within parts can break K4-freeness if we're not careful. Let's
think: if we add edges within part A, can we form a K4? A K4 would need 4 vertices
pairwise adjacent. In T_3(18), vertices in the same part are not adjacent (unless we
add edges). If we add a clique of size k within part A, then combined with vertices
from other parts, we might form larger cliques.

Specifically, if part A has a clique of size c, then c vertices in A plus one vertex
from B and one from C would form a K_{c+2} (since all A-vertices are adjacent to all
B-vertices and all C-vertices, and A-vertices are pairwise adjacent within the clique).
So if c+2 >= 4, i.e., c >= 2, we get a K4. So we can have at most a matching (no two
adjacent) within each part to avoid K4s.

Wait: if we add a single edge (u,v) within part A, then {u, v, b, c} where b in B, c
in C: u~v (added), u~b, u~c, v~b, v~c, b~c. That's a K4! So even a single edge within
a part creates a K4.

So we CANNOT add any edges within parts of T_3(18) without creating K4s.

Alternative: use a non-complete 3-partite structure. Remove some inter-part edges to
break triangles while preserving K4-freeness.

Key insight: In T_3(18), any subset with vertices from all three parts contains a
triangle (one from each part, all mutually adjacent). If we remove some inter-part
edges, we can break certain triangles, allowing larger triangle-free sets... wait, no,
we want to MINIMIZE the largest triangle-free set. Removing edges makes it EASIER to
find triangle-free sets (fewer triangles to block them). So removing edges is bad.

So the complete 3-partite T_3(18) with parts of size 6 gives score 12. Can we do better?

What about a different graph structure? Consider the complement of a perfect matching
or other K4-free graphs.

Actually, let me reconsider: the task says "K4-free graph whose triangle-free induced
subgraphs have at most s vertices certifies f(18) <= s". We want to minimize s.

The Turán graph T_3(18) gives s = 12. Can we get s < 12?

A different approach: use a graph where the independence number (triangle-free sets
are not the same as independent sets, but related) is small.

Actually, let me try: take the complete graph K_18 and remove edges to make it
K4-free, while ensuring every 12-vertex subset contains a triangle.

This is equivalent to: find a K4-free graph on 18 vertices such that every subset of
12 vertices induces a triangle.

The Turán graph T_3(18) works: every 12-vertex subset either uses all three parts
(containing a triangle) or is contained in two parts (which has size at most 12, and
if it's exactly two full parts, it's triangle-free since it's bipartite). Wait, two
full parts = 12 vertices, and that IS triangle-free. So the max triangle-free set is
12.

To get < 12, we need every 11-vertex... no, we need that NO 12-vertex subset is
triangle-free, meaning every 12-vertex subset contains a triangle. But in T_3(18),
the union of two parts (12 vertices) is triangle-free. So T_3(18) has a triangle-free
set of size 12.

Can we construct a K4-free graph where every 12-vertex subset contains a triangle?
That would give score 11.

One idea: make the graph more "dense" in a controlled way. Use a 3-partite graph
but with unequal parts or modified edge structure.

Alternative: use a graph based on the complement of a triangle-free graph, ensuring
K4-freeness.

Let me try a different construction: take the disjoint union of cliques. K3 ∪ K3 ∪ K3 ∪
K3 ∪ K3 ∪ K3 (six triangles). This is K4-free. The largest triangle-free induced
subgraph: from each triangle, we can take at most 2 vertices (3 would form a triangle).
So max triangle-free = 6 * 2 = 12. Same as T_3(18).

What about K3 ∪ K3 ∪ K3 ∪ K3 ∪ K4? No, K4 is not allowed.

How about a different mix: five K3's and one K3... that's the same.

Let me try: a complete 3-partite graph with parts of sizes 5, 6, 7. Max triangle-free
= 6+7 = 13. Worse.

I think 12 is hard to beat with simple constructions. Let me just output T_3(18)
with a cleaner implementation.

"""

def construct():
    n = 18
    part_size = n // 3
    parts = [
        list(range(0, part_size)),
        list(range(part_size, 2 * part_size)),
        list(range(2 * part_size, 3 * part_size))
    ]
    
    edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    edges.append([u, v])
                    
    return edges
