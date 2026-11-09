import time
import random
import sys
from collections import defaultdict

def construct():
    deadline = time.time() + 110  # safety margin
    n = 20
    num_colors = 20
    
    # We'll work with a graph represented by edge->color dict
    # edge is (u,v) with u<v
    edges = {}  # (u,v) -> color
    adj = defaultdict(set)  # vertex -> set of neighbors
    edge_color = {}  # (u,v) -> color
    
    def get_edge_key(u, v):
        return (min(u,v), max(u,v))
    
    def add_edge(u, v, c):
        key = get_edge_key(u, v)
        edges[key] = c
        edge_color[key] = c
        adj[u].add(v)
        adj[v].add(u)
    
    def remove_edge(u, v):
        key = get_edge_key(u, v)
        if key in edges:
            del edges[key]
            del edge_color[key]
            adj[u].discard(v)
            adj[v].discard(u)
    
    def check_rainbow_4cycle(u, v, c):
        """Check if adding edge (u,v) with color c violates rainbow 4-cycle constraint."""
        # For every pair of common neighbors or paths of length 2
        # A 4-cycle containing (u,v): u-v-w-x-u where w,x are vertices
        # Actually, 4-cycle: u-v-a-b-u where a,b are distinct from u,v and from each other
        # The cycle is u-v-a-b-u, edges: (u,v),(v,a),(a,b),(b,u)
        # We need colors of all 4 edges to be distinct.
        # (u,v) has color c.
        # For each a in adj[v] (a != u), for each b in adj[u] (b != v, b != a):
        #   if (a,b) is an edge, then we have 4-cycle u-v-a-b-u
        #   Check colors: c, color(v,a), color(a,b), color(b,u) all distinct
        for a in adj[v]:
            if a == u:
                continue
            ca = edge_color[get_edge_key(v, a)]
            if ca == c:
                return False
            for b in adj[u]:
                if b == v or b == a:
                    continue
                cb = edge_color[get_edge_key(u, b)]
                if cb == c or cb == ca:
                    return False
                key_ab = get_edge_key(a, b)
                if key_ab in edge_color:
                    cab = edge_color[key_ab]
                    if cab == c or cab == ca or cab == cb:
                        return False
        return True
    
    def count_violations():
        """Count number of 4-cycles that are not rainbow."""
        violations = 0
        vertices = list(range(n))
        for u in vertices:
            for v in adj[u]:
                if v <= u:
                    continue
                cu = edge_color[get_edge_key(u, v)]
                for a in adj[v]:
                    if a == u:
                        continue
                    ca = edge_color[get_edge_key(v, a)]
                    if ca == cu:
                        continue
                    for b in adj[u]:
                        if b == v or b == a:
                            continue
                        cb = edge_color[get_edge_key(u, b)]
                        if cb == cu or cb == ca:
                            continue
                        key_ab = get_edge_key(a, b)
                        if key_ab in edge_color:
                            cab = edge_color[key_ab]
                            if cab == cu or cab == ca or cab == cb:
                                violations += 1
        return violations
    
    def try_add_edge(u, v):
        """Try to add edge (u,v) with some valid color."""
        key = get_edge_key(u, v)
        if key in edges:
            return False
        # Try each color
        for c in range(num_colors):
            if check_rainbow_4cycle(u, v, c):
                add_edge(u, v, c)
                return True
        return False
    
    def try_add_edge_with_color(u, v, c):
        key = get_edge_key(u, v)
        if key in edges:
            return False
        if check_rainbow_4cycle(u, v, c):
            add_edge(u, v, c)
            return True
        return False
    
    # Phase 1: Greedy construction
    # Start with a proper edge-colouring approach
    # Use 1-factorization-like structure
    
    # First, try to build a dense graph greedily
    random.seed(42)
    
    # Start with all possible edges in random order
    all_edges = []
    for u in range(n):
        for v in range(u+1, n):
            all_edges.append((u, v))
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline:
            break
        try_add_edge(u, v)
    
    current_edges = len(edges)
    
    # Phase 2: Local search - try to add more edges by recoloring
    # and try to fix conflicts to allow more edges
    
    # Try removing an edge and adding two others
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Try to add a missing edge
        missing = []
        for u in range(n):
            for v in range(u+1, n):
                if get_edge_key(u, v) not in edges:
                    missing.append((u, v))
        
        if not missing:
            break
        
        # Try random missing edge
        u, v = random.choice(missing)
        if try_add_edge(u, v):
            continue
        
        # Try recoloring an existing edge to free up a color
        # Pick a random existing edge, change its color, see if we can add (u,v)
        existing = list(edges.keys())
        if existing:
            eu, ev = random.choice(existing)
            old_color = edge_color[get_edge_key(eu, ev)]
            # Try changing its color
            for new_c in range(num_colors):
                if new_c == old_color:
                    continue
                # Temporarily change
                edge_color[get_edge_key(eu, ev)] = new_c
                # Check if this creates violations
                # For simplicity, just try to add (u,v)
                if check_rainbow_4cycle(u, v, old_color) or check_rainbow_4cycle(u, v, random.randint(0, num_colors-1)):
                    # Try adding (u,v)
                    added = False
                    for c in range(num_colors):
                        if check_rainbow_4cycle(u, v, c):
                            add_edge(u, v, c)
                            added = True
                            break
                    if not added:
                        # Revert color change
                        edge_color[get_edge_key(eu, ev)] = old_color
                else:
                    edge_color[get_edge_key(eu, ev)] = old_color
    
    # Phase 3: Try to add more by removing and re-adding
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        missing = []
        for u in range(n):
            for v in range(u+1, n):
                if get_edge_key(u, v) not in edges:
                    missing.append((u, v))
        
        if not missing:
            break
        
        u, v = random.choice(missing)
        if try_add_edge(u, v):
            continue
        
        # Try removing a random edge to free up structure
        existing = list(edges.keys())
        if len(existing) > 10:
            eu, ev = random.choice(existing)
            old_c = edge_color[get_edge_key(eu, ev)]
            remove_edge(eu, ev)
            
            # Try to add (u,v) and possibly (eu,ev) back with different color
            added_uv = try_add_edge(u, v)
            added_eu = try_add_edge(eu, ev)
            
            if not added_eu:
                # Try to put it back
                for c in range(num_colors):
                    if check_rainbow_4cycle(eu, ev, c):
                        add_edge(eu, ev, c)
                        break
    
    # Build result
    result = []
    for (u, v), c in edges.items():
        result.append([u, v, c])
    
    return result
