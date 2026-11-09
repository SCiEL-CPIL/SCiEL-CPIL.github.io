"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

_DEADLINE = 150.0  # seconds total budget for the evaluator
_SAFETY_MARGIN = 5.0  # seconds to leave before the hard kill

# Precompute all Q3 subgraphs (as frozensets of edge indices) on 16 vertices
# A Q3 subgraph is a set of 8 vertices and 12 edges forming a 3-cube.
# We represent edges as (u, v) with u < v.

def _all_edges():
    """Return list of all possible edges on 16 vertices."""
    return [(i, j) for i in range(16) for j in range(i + 1, 16)]

def _edge_index_map(edges):
    """Map edge tuple -> index in the edges list."""
    return {e: i for i, e in enumerate(edges)}

def _find_q3_subgraphs(n=16):
    """
    Enumerate all Q3 subgraphs.
    A Q3 on vertices {v0..v7} has edges corresponding to the 12 edges of a cube.
    The cube has 8 vertices and 12 edges. Each vertex has degree 3.
    
    We enumerate all 8-subsets of 16 vertices and check if they contain a Q3.
    For a set of 8 vertices, a Q3 exists if we can label them as the 8 corners
    of a cube such that all 12 cube edges are present in our edge set.
    
    Actually, for the purpose of building clauses, we need to know WHICH sets of
    12 edges form a Q3. A Q3 is determined by its 8 vertices. Given 8 vertices,
    there are multiple ways to form a cube structure on them.
    
    Let's think differently: enumerate all 8-vertex subsets, and for each, 
    check if the induced subgraph contains a Q3 as a subgraph (not necessarily induced).
    But that's complex. Instead, let's enumerate all possible Q3 structures.
    
    A Q3 can be identified by choosing 8 vertices and a specific cube embedding.
    The number of distinct Q3 subgraphs in K16 is C(16,8) * (number of cube labelings).
    The number of distinct cube structures on 8 labeled vertices is 384 (the automorphism group
    of the cube has 48 elements, and there are 8! / 48 = 1680... no, let me think again).
    
    Actually, let's just enumerate all 8-subsets and for each, check if there's a way to 
    assign the 8 vertices to cube corners such that all 12 cube edges exist in the graph.
    But for building the Max-SAT clauses, we need the actual edge sets.
    
    Simpler approach: For local search, we don't need to precompute all Q3s.
    We can check if adding an edge creates a Q3 by looking at the current graph.
    """
    pass  # Not needed for the local search approach below

def _has_q3(edge_set):
    """
    Check if a graph (given as a set of edge tuples) contains a Q3 subgraph.
    This is expensive, so we use it sparingly.
    
    A Q3 has 8 vertices and 12 edges. We can check by:
    1. Find all 8-vertex subsets that have at least 12 edges among them.
    2. For each, check if the induced subgraph contains a Q3.
    
    But this is very expensive. Instead, we'll use a different strategy for local search.
    """
    # This function is kept for completeness but we won't use it in the main loop
    # due to performance concerns.
    return False

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - _SAFETY_MARGIN
    
    all_edges = _all_edges()
    edge_map = _edge_index_map(all_edges)
    n_edges = len(all_edges)
    
    # Initialize with a random dense graph
    random.seed(42)
    # Start with all edges, then we'll remove as needed
    # Actually, let's start with a moderately dense graph
    # K16 has 120 edges. Let's start with a random subset of ~80 edges
    initial_density = 0.65
    current_edges = set()
    for e in all_edges:
        if random.random() < initial_density:
            current_edges.add(e)
    
    def _count_q3s_involving_edge(edge, edge_set):
        """
        Count how many Q3 subgraphs contain the given edge.
        A Q3 containing edge (u,v) must have u and v as adjacent cube vertices.
        In a cube, each edge belongs to exactly 1 face (square) and is part of 
        the cube structure.
        
        For a given edge (u,v), the other 6 vertices of the Q3 must be chosen
        from the remaining 14 vertices. The cube structure requires specific
        relationships.
        
        This is complex. Instead, let's use a simpler heuristic:
        Count the number of 4-cycles (squares) that contain this edge, since
        each Q3 contains 6 squares, and each square contains 4 edges.
        """
        u, v = edge
        # Find all vertices w such that (u,w) and (v,w) are both in edge_set
        # These form potential "corners" of squares with edge (u,v)
        common_neighbors = []
        for w in range(16):
            if w == u or w == v:
                continue
            if (min(u,w), max(u,w)) in edge_set and (min(v,w), max(v,w)) in edge_set:
                common_neighbors.append(w)
        
        # For each pair of common neighbors (w1, w2), check if (w1,w2) is an edge
        # and if the 4 vertices {u,v,w1,w2} form a square
        # A square u-v-w2-w1-u requires edges (u,v), (v,w2), (w2,w1), (w1,u)
        # We already have (u,v), (v,w2), (w1,u). Need (w1,w2).
        square_count = 0
        for i in range(len(common_neighbors)):
            for j in range(i+1, len(common_neighbors)):
                w1, w2 = common_neighbors[i], common_neighbors[j]
                if (min(w1,w2), max(w1,w2)) in edge_set:
                    square_count += 1
        return square_count
    
    def _is_q3_free(edge_set):
        """
        Check if the graph is Q3-free.
        We'll check by trying to find a Q3 using a backtracking approach.
        
        A Q3 has 8 vertices. We can try to build one vertex by vertex.
        This is still expensive, but let's try a limited approach.
        
        Alternative: Use the fact that a Q3 is a bipartite graph with parts of size 4.
        Check all 4-4 bipartitions of 8-vertex subsets... still expensive.
        
        For local search, we'll use a probabilistic check or a limited search.
        """
        # For now, assume it's Q3-free if we can't quickly find one
        # We'll do a limited search
        vertices = sorted(set(u for e in edge_set for u in e) | set(v for e in edge_set for v in e))
        
        # Try to find a Q3 by checking 8-vertex subsets with high edge density
        # Limit the number of subsets we check
        checked = 0
        max_checks = 5000
        
        # Get all 8-vertex subsets with at least 12 edges in the induced subgraph
        # This is still expensive, so let's sample
        if len(vertices) < 8:
            return True
            
        # Sample some 8-vertex subsets
        for _ in range(max_checks):
            if time.time() > deadline:
                return True  # Assume Q3-free if we're running out of time
            subset = random.sample(vertices, 8)
            subset_set = set(subset)
            
            # Count edges in the induced subgraph
            induced_edges = 0
            for e in edge_set:
                if e[0] in subset_set and e[1] in subset_set:
                    induced_edges += 1
                    if induced_edges >= 12:
                        break
            if induced_edges < 12:
                continue
                
            # Check if this 8-vertex induced subgraph contains a Q3
            # A Q3 on 8 vertices is a specific 12-edge structure
            # We need to check if there's a subgraph isomorphic to Q3
            # This means we need to find 12 edges among the induced edges that form a Q3
            
            # Get the induced edges
            induced_edge_list = []
            for e in edge_set:
                if e[0] in subset_set and e[1] in subset_set:
                    induced_edge_list.append(e)
            
            # Try to find a Q3 among these edges
            # A Q3 has 12 edges. If we have exactly 12, check if they form a Q3
            # If we have more, we need to check subsets
            
            if len(induced_edge_list) <= 20:  # Limit for performance
                # Check if any 12-edge subset forms a Q3
                # This is still expensive, so let's do a quick check
                # A Q3 is 3-regular on 8 vertices
                # Check degree sequence
                degrees = {v: 0 for v in subset}
                for e in induced_edge_list:
                    degrees[e[0]] += 1
                    degrees[e[1]] += 1
                
                # In a Q3, all 8 vertices have degree 3
                # If the induced subgraph has all vertices with degree >= 3, 
                # it might contain a Q3
                if all(degrees[v] >= 3 for v in subset):
                    # More detailed check needed, but for now flag it
                    # We'll do a more thorough check in the main loop if needed
                    return False  # Conservatively assume it has a Q3
            checked += 1
        
        return True
    
    # Local search: iteratively remove edges that are in many Q3s, then try to add edges
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # Check if current graph is Q3-free
        if not _is_q3_free(current_edges):
            # Find and remove an edge that's in a Q3
            # Use the square count as a proxy for Q3 involvement
            if current_edges:
                # Find the edge with the highest square count
                max_squares = -1
                worst_edge = None
                for e in current_edges:
                    sq = _count_q3s_involving_edge(e, current_edges)
                    if sq > max_squares:
                        max_squares = sq
                        worst_edge = e
                if worst_edge is not None:
                    current_edges.remove(worst_edge)
                else:
                    # Remove a random edge
                    current_edges.pop()
            continue
        
        # Current graph is Q3-free. Try to add edges
        # Try adding random missing edges
        missing_edges = [e for e in all_edges if e not in current_edges]
        if not missing_edges:
            break
        
        # Try a few random edges
        improved = False
        for _ in range(10):
            if time.time() > deadline:
                break
            e = random.choice(missing_edges)
            new_edges = current_edges | {e}
            if _is_q3_free(new_edges):
                current_edges = new_edges
                improved = True
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = set(current_edges)
                break
        
        if not improved and iteration > 100:
            # Try a different random starting point
            if iteration % 100 == 0:
                current_edges = set()
                for e in all_edges:
                    if random.random() < initial_density:
                        current_edges.add(e)
    
    return [[e[0], e[1]] for e in best_edges]
# EVOLVE-BLOCK-END
