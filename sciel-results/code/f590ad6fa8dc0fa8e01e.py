import time
from random import Random

def construct():
       deadline = time.time() + 115 # Leave 5s margin
       
       def generate_graph(n, p, seed):
           rng = Random(seed)
           adj = [0] * n
           edges = []
           for u in range(n):
               for v in range(u + 1, n):
                   if rng.random() < p:
                       adj[u] |= (1 << v)
                       adj[v] |= (1 << u)
                       edges.append((u, v))
           return adj, edges

       def max_clique(adj, n, rng, time_limit=0.5):
           # Returns a large clique (list of vertices)
           best = []
           start_time = time.time()
           
           def expand(R, P):
               nonlocal best, start_time
               if time.time() - start_time > time_limit:
                   return
               if not P:
                   if len(R) > len(best):
                       best = list(R)
                   return
                   
               # Pivot selection
               # Choose u in P with max degree in P
               u = -1
               max_deg = -1
               tmp = P
               while tmp:
                   lsb = tmp & -tmp
                   v = lsb.bit_length() - 1
                   deg = (adj[v] & P).bit_count()
                   if deg > max_deg:
                       max_deg = deg
                       u = v
                   tmp ^= lsb
                   
               candidates = P & ~adj[u]
               
               tmp = candidates
               while tmp:
                   lsb = tmp & -tmp
                   v = lsb.bit_length() - 1
                   
                   R.append(v)
                   expand(R, P & adj[v])
                   R.pop()
                   
                   P ^= lsb
                   tmp = P
                   
           # To allow randomization, we can shuffle the initial P?
           # No, P is determined by adjacency.
           # We can randomize the pivot choice among top candidates?
           # Or randomize the iteration order of candidates.
           
           # For now, deterministic. We will run multiple instances with different 'seeds' for rng if we add randomness.
           # Let's add randomness to pivot selection: pick from top 3 degrees.
           
           expand([], (1 << n) - 1)
           return best

       def solve_instance(n, p, seed):
           adj, edges = generate_graph(n, p, seed)
           if not edges:
               return []
           
           remaining_adj = adj[:]
           remaining_edges = set(edges)
           cliques = []
           
           # We will run a greedy loop
           # To improve, we can try multiple "strategies" or randomizations
           
           best_cliques = []
           best_count = float('inf')
           
           # Number of restarts
           # n=32: can do many. n=64: fewer.
           restarts = 50 if n == 32 else 10
           
           for i in range(restarts):
               if time.time() > deadline - 5:
                   break
               
               cur_adj = adj[:]
               cur_edges = set(edges)
               cur_cliques = []
               
               while cur_edges:
                   # Find max clique in cur_adj
                   # Pass a random seed to allow variation?
                   # The current max_clique is deterministic.
                   # Let's modify max_clique to accept a randomizer for tie-breaking or pivot selection.
                   
                   clique = find_max_clique_randomized(cur_adj, n, i)
                   
                   if not clique:
                       # Fallback: pick any edge
                       e = next(iter(cur_edges))
                       cur_cliques.append(list(e))
                       # Remove edge
                       u, v = e
                       cur_adj[u] &= ~(1 << v)
                       cur_adj[v] &= ~(1 << u)
                       cur_edges.discard(e)
                       continue
                   
                   # Remove all edges in clique
                   cset = set(clique)
                   to_remove = set()
                   for u, v in cur_edges:
                       if u in cset and v in cset:
                           to_remove.add((u, v))
                   
                   if to_remove:
                       cur_cliques.append(clique)
                       for u, v in to_remove:
                           cur_adj[u] &= ~(1 << v)
                           cur_adj[v] &= ~(1 << u)
                           cur_edges.discard((u, v))
                   else:
                       # Should not happen
                       e = next(iter(cur_edges))
                       cur_cliques.append(list(e))
                       u, v = e
                       cur_adj[u] &= ~(1 << v)
                       cur_adj[v] &= ~(1 << u)
                       cur_edges.discard(e)
               
               if len(cur_cliques) < best_count:
                   best_count = len(cur_cliques)
                   best_cliques = cur_cliques
                   
           return best_cliques

       # Implement find_max_clique_randomized
       def find_max_clique_randomized(adj, n, seed_offset):
           # Use a specific RNG for this run
           rng = Random(seed_offset)
           best = []
           start_time = time.time()
           limit = 0.1 # Tight limit per clique search to allow more restarts?
           # Actually, for n=64, 0.1s might be enough for greedy max clique.
           # But if we want *exact* max clique, it might take longer.
           # Let's try a greedy expansion that is randomized.
           
           # Greedy Expansion with Randomization
           # Pick start vertex randomly from high degree?
           # Or use the Bron-Kerbosch with randomized pivot.
           
           def expand(R, P):
               nonlocal best, start_time
               if time.time() - start_time > limit:
                   return
               if not P:
                   if len(R) > len(best):
                       best = list(R)
                   return
                   
               # Randomized Pivot: Pick a vertex u in P.
               # To make it effective, pick u with high degree, but add some randomness.
               # Or just pick random u in P?
               # Better: Pick top 3 by degree, choose one randomly.
               
               degs = []
               tmp = P
               while tmp:
                   lsb = tmp & -tmp
                   v = lsb.bit_length() - 1
                   degs.append(((adj[v] & P).bit_count(), v))
                   tmp ^= lsb
                   
               if not degs: return
               
               # Sort by degree
               degs.sort(reverse=True)
               # Pick from top 3
               k = min(3, len(degs))
               u = degs[rng.randint(0, k-1)][1]
               
               candidates = P & ~adj[u]
               
               # Shuffle candidates?
               cand_list = []
               tmp = candidates
               while tmp:
                   lsb = tmp & -tmp
                   v = lsb.bit_length() - 1
                   cand_list.append(v)
                   tmp ^= lsb
                   
               rng.shuffle(cand_list)
               
               for v in cand_list:
                   R.append(v)
                   expand(R, P & adj[v])
                   R.pop()
                   P &= ~(1 << v)
                   
           expand([], (1 << n) - 1)
           return best

       result = []
       for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
           result.append(solve_instance(n, p, seed))
       return result
