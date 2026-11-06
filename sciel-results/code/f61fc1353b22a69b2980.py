     def reduce_dominated(nodes):
         # nodes is a set
         # If N[u] <= N[v], remove v
         # Note: N[u] includes u? Usually closed neighborhood.
         # Let's use open neighborhood + {u}.
         # To be safe, we check if {u} U N(u) is subset of {v} U N(v).
         # Since u != v, this implies u in N(v) or u=v.
         # If u in N(v), then v in N(u)? Not necessarily.
         # Standard domination: u dominates v if N(u) \subseteq N(v).
         # Then we can remove v.
         # Implementation:
         # Sort by degree ascending? Small degree first?
         # If N(u) is small, it's more likely to be a subset.
         # So iterate u in increasing degree.
         # For each u, check all v in nodes \ {u}.
         # If N(u) <= N(v), mark v for removal.
         # This is O(N^2 * D).
         # For N=200, D=10, this is 200*200*10 = 400k ops. Fast enough.
         nodes_list = list(nodes)
         to_remove = set()
         degs = {u: len(conflict[u] & nodes) for u in nodes_list}
         # Sort by degree
         sorted_nodes = sorted(nodes_list, key=lambda x: degs[x])
         for u in sorted_nodes:
             if u in to_remove:
                 continue
             Nu = conflict[u] | {u}
             for v in sorted_nodes:
                 if v == u or v in to_remove:
                     continue
                 Nv = conflict[v] | {v}
                 # Check if Nu is subset of Nv
                 # Nu <= Nv
                 # Since Nu contains u, Nv must contain u.
                 # So u must be in Nv.
                 if u not in Nv:
                     continue
                 # Check rest
                 is_subset = True
                 for w in Nu:
                     if w not in Nv:
                         is_subset = False
                         break
                 if is_subset:
                     to_remove.add(v)
         return nodes - to_remove
     