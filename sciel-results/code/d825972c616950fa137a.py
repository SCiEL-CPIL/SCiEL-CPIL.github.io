import time
import random
from itertools import combinations

# Deadline guard: must finish well before 150s wall-clock
_START_TIME = time.time()
DEADLINE = 120.0  # safety margin

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Return coloured edges [[u, v, c], ...] of a simple graph on vertices 0..19
    with colours 0..19 such that every 4-cycle is rainbow.
    Maximize the number of edges.
    
    Strategy: Construct a C4-free graph. A graph with no 4-cycles satisfies the 
    rainbow condition vacuously. We use a greedy algorithm with multiple random
    orderings to find a denser C4-free subgraph of K_20.
    """
    n = 20
    all_edges_template = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edge_count = 0
    best_graph = []
    
    # Try multiple random orderings to find a denser C4-free graph
    max_trials = 500  # will be limited by time
    trial = 0
    
    while _time_left() > 5.0 and trial < max_trials:
        trial += 1
        random.seed(trial)
        all_edges = all_edges_template[:]
        random.shuffle(all_edges)
        
        # Use sets for fast membership testing
        neighbors = [set() for _ in range(n)]
        graph_edges = []
        
        for u, v in all_edges:
            if _time_left() < 5.0:
                break
            
            # Check if adding (u, v) creates a C4
            # A C4 is u-x-y-v-u where x != v, y != u, x != y
            # Path of length 3: u-x-y-v
            creates = False
            for x in neighbors[u]:
                if x == v:
                    continue
                for y in neighbors[x]:
                    if y == u or y == v:
                        continue
                    if v in neighbors[y]:
                        creates = True
                        break
                if creates:
                    break
            
            if not creates:
                neighbors[u].add(v)
                neighbors[v].add(u)
                graph_edges.append((u, v))
        
        if len(graph_edges) > best_edge_count:
            best_edge_count = len(graph_edges)
            best_graph = graph_edges[:]
    
    # Color all edges with color 0 (valid since C4-free)
    result = [[u, v, 0] for u, v in best_graph]
    
    return result
