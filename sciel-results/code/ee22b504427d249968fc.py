   def pebble(dag, fast_memory_budget):
       import sys
       from collections import OrderedDict, defaultdict

       def extract_dag(dag):
           parents = None
           outputs = None
           inputs = None
           vertices = None

           def as_set(x):
               if x is None: return None
               if isinstance(x, (list, tuple, set, frozenset)): return set(x)
               return {x}

           # dict
           if isinstance(dag, dict):
               for key in ['parents', 'predecessors', 'in_edges', 'preds', 'parent', 'in_edges_list']:
                   if key in dag and isinstance(dag[key], dict):
                       parents = {k: list(v) for k, v in dag[key].items()}
                       break
               if parents is None:
                   # maybe dag maps vertex -> list of parents directly
                   if dag and all(isinstance(v, (list, tuple, set)) for v in dag.values()):
                       parents = {k: list(v) for k, v in dag.items()}
               for key in ['outputs', 'sinks', 'output', 'targets', 'out']:
                   if key in dag:
                       outputs = as_set(dag[key])
                       break
               for key in ['inputs', 'sources', 'input', 'roots', 'src']:
                   if key in dag:
                       inputs = as_set(dag[key])
                       break
               for key in ['vertices', 'nodes', 'vertex', 'node', 'verts']:
                   if key in dag and isinstance(dag[key], (list, tuple, set)):
                       vertices = list(dag[key])
                       break
           else:
               # object or list
               if isinstance(dag, (list, tuple)):
                   # list of vertex dicts?
                   # try: each element has 'parents'
                   if dag and isinstance(dag[0], dict):
                       parents = {}
                       vertices = []
                       for item in dag:
                           vid = item.get('id', item.get('v', item.get('vertex')))
                           if vid is None:
                               vid = item.get('name')
                           if vid is None:
                               continue
                           vertices.append(vid)
                           ps = item.get('parents', item.get('predecessors', []))
                           parents[vid] = list(ps)
                       for item in dag:
                           if item.get('is_output') or item.get('output'):
                               if outputs is None: outputs = set()
                               outputs.add(item.get('id', item.get('v')))
                           if item.get('is_input') or item.get('input'):
                               if inputs is None: inputs = set()
                               inputs.add(item.get('id', item.get('v')))
               elif hasattr(dag, 'parents') or hasattr(dag, 'nodes') or hasattr(dag, 'vertices'):
                   if hasattr(dag, 'parents'):
                       p = dag.parents
                       if isinstance(p, dict):
                           parents = {k: list(v) for k, v in p.items()}
                       elif isinstance(p, (list, tuple)):
                           parents = {i: list(x) for i, x in enumerate(p)}
                   if hasattr(dag, 'outputs'):
                       outputs = as_set(dag.outputs)
                   if hasattr(dag, 'inputs'):
                       inputs = as_set(dag.inputs)
                   if hasattr(dag, 'vertices'):
                       vertices = list(dag.vertices)
                   elif hasattr(dag, 'nodes'):
                       vertices = list(dag.nodes)

           if parents is None:
               raise ValueError("Could not extract parents from dag")

           if vertices is None:
               vs = set(parents.keys())
               for ps in parents.values():
                   vs.update(ps)
               vertices = list(vs)
           else:
               vs = set(vertices)
               vs.update(parents.keys())
               for ps in parents.values():
                   vs.update(ps)
               vertices = list(vs)

           # ensure parents dict has all vertices
           for v in vertices:
               if v not in parents:
                   parents[v] = []

           if inputs is None:
               inputs = set(v for v in vertices if not parents[v])

           if outputs is None:
               children = defaultdict(set)
               for v, ps in parents.items():
                   for p in ps:
                       children[p].add(v)
               outputs = set(v for v in vertices if not children[v])

           return vertices, parents, outputs, inputs

       vertices, parents, outputs, inputs = extract_dag(dag)

       # topological sort (Kahn)
       indeg = {v: len(parents[v]) for v in vertices}
       children = defaultdict(list)
       for v, ps in parents.items():
           for p in ps:
               children[p].append(v)
       from collections import deque
       q = deque(v for v in vertices if indeg[v] == 0)
       topo = []
       indeg2 = dict(indeg)
       while q:
           v = q.popleft()
           topo.append(v)
           for c in children[v]:
               indeg2[c] -= 1
               if indeg2[c] == 0:
                   q.append(c)
       if len(topo) != len(vertices):
           # cycle; append rest
           topo += [v for v in vertices if v not in set(topo)]

       # remaining children count
       rem_children = {v: len(children[v]) for v in vertices}

       available = set(inputs)
       in_fast = set()
       ops = []
       lru = OrderedDict()

       def touch(v):
           if v in lru:
               lru.move_to_end(v)
           else:
               lru[v] = None

       def evict_one():
           # pick victim: not output, lowest rem_children, then oldest in lru
           victim = None
           best = None
           idx = 0
           for cand in lru:
               if cand in outputs:
                   idx += 1
                   continue
               key = (rem_children.get(cand, 0), idx)
               if best is None or key < best:
                   best = key
                   victim = cand
               idx += 1
           if victim is None:
               # fallback: oldest non-output, or any
               for cand in lru:
                   if cand not in outputs:
                       victim = cand
                       break
           if victim is None:
               victim = next(iter(lru))
           ops.append(['store', victim])
           in_fast.discard(victim)
           lru.pop(victim, None)

       budget = max(0, int(fast_memory_budget))

       for v in topo:
           if v in inputs:
               # already available; nothing to compute
               # but we may still need to account for it in lru if loaded later
               continue
           # load parents
           for p in parents[v]:
               if p not in in_fast:
                   if p not in available:
                       # should not happen
                       continue
                   while budget > 0 and len(in_fast) >= budget:
                       evict_one()
                   ops.append(['load', p])
                   in_fast.add(p)
                   touch(p)
           # compute v
           while budget > 0 and len(in_fast) >= budget:
               evict_one()
           ops.append(['compute', v])
           available.add(v)
           in_fast.add(v)
           touch(v)
           # update rem_children of parents
           for p in parents[v]:
               rem_children[p] -= 1

       # Ensure outputs are available (they should be)
       # If an output was evicted and not in available, recompute? 
       # In our scheme, available is only added on compute and never removed (except delete which we don't use).
       # So all computed vertices remain available. Good.

       # Trim ops if too many (should not happen)
       if len(ops) > 20000:
           ops = ops[:20000]

       return ops
   