import itertools
import networkx as nx

def construct():
    # Precompute modular inverses of 1..10 modulo 101
    invs = [0] * 11
    for x in range(1, 11):
        invs[x] = pow(x, -1, 101)
        
    # Generate all non-empty subsets of {1..10}
    # Store as (size, residue, subset_tuple)
    subsets = []
    for r in range(1, 11):
        for terms in itertools.combinations(range(1, 11), r):
            residue = sum(invs[x] for x in terms) % 101
            subsets.append((r, residue, terms))
            
    # Group subsets by residue for quick check
    residue_has_subset = {i: [] for i in range(101)}
    for size, res, terms in subsets:
        residue_has_subset[res].append((size, terms))
        
    min_max_size = 10
    best_assignment = [None] * 101
    
    for K in range(1, 11):
        # Check if every residue has at least one subset of size <= K
        possible = True
        for res in range(101):
            if not any(size <= K for size, _ in residue_has_subset[res]):
                possible = False
                break
        
        if not possible:
            continue
            
        G = nx.Graph()
        for res in range(101):
            G.add_node(('L', res))
            
        # Add right nodes and edges
        # We only add subsets of size <= K
        for size, res, terms in subsets:
            if size <= K:
                G.add_node(('R', terms))
                G.add_edge(('L', res), ('R', terms))
                
        # Check for perfect matching covering all L nodes
        # nx.bipartite.maximum_matching requires the graph to be bipartite
        # We pass one of the partitions to help it
        try:
            matching = nx.bipartite.maximum_matching(G, nx.bipartite.sets(G)[0])
        except Exception:
            continue
            
        # A perfect matching covering 101 L-nodes will have 101 edges.
        # The matching dict will have 202 entries (101 L -> R, 101 R -> L).
        if len(matching) == 202:
            assignment = [None] * 101
            for node, match in matching.items():
                if node[0] == 'L':
                    res = node[1]
                    terms = match[1]
                    assignment[res] = list(terms)
                    
            if all(x is not None for x in assignment):
                return assignment
                
    # Fallback
    return best_assignment
