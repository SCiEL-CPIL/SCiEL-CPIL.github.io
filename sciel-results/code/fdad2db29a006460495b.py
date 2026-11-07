<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    adj = build_adj()
    
    # Greedy partition
    bicliques = greedy_biclique_partition(adj, deadline)
    
    # Try to merge
    bicliques = try_merge_bicliques(bicliques, adj, deadline)
    
    # Convert to proper format
    result = []
    for L, R in bicliques:
        result.append([list(L), list(R)])
    
    return result
=======
def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (90s limit)
    
    adj = build_adj()
    
    # Greedy partition with larger bicliques
    bicliques = greedy_biclique_partition(adj, deadline)
    
    # Try to merge
    bicliques = try_merge_bicliques(bicliques, adj, deadline)
    
    # Local search polish
    bicliques = local_search_polish(bicliques, adj, deadline)
    
    # Final verification and conversion to proper format
    result = []
    for L, R in bicliques:
        L_list = [int(x) for x in L]
        R_list = [int(x) for x in R]
        result.append([L_list, R_list])
    
    return result
>>>>>>> REPLACE
