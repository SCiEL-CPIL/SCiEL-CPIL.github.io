   def find_longest_cycle(adj, n):
       # adj: dict of sets
       # Returns longest cycle found
       best_cycle = []
       
       # Sort nodes by degree descending
       nodes = sorted(adj.keys(), key=lambda x: -len(adj[x]))
       
       for start in nodes:
           if not adj[start]: continue
           
           # DFS
           stack = [(start, -1, [start])]
           visited = {start}
           
           while stack:
               node, parent, path = stack.pop()
               
               # Sort neighbors: prefer those with high degree, unvisited
               # We want to go deep
               neighbors = sorted(adj[node], key=lambda x: (x in visited, -len(adj[x])))
               
               for nb in neighbors:
                   if nb == parent:
                       continue
                   if nb in visited:
                       # Found a cycle
                       if nb in path:
                           idx = path.index(nb)
                           cycle = path[idx:]
                           if len(cycle) > len(best_cycle):
                               best_cycle = cycle
                   else:
                       visited.add(nb)
                       stack.append((nb, node, path + [nb]))
                       visited.discard(nb) # Backtrack: remove from visited? 
                       # Wait, standard DFS for longest cycle needs to handle visited carefully.
                       # If we mark visited globally, we might miss cycles.
                       # We need path-based visited.
       
       return best_cycle
   