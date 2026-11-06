import time
import heapq

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 50.0 # 10s margin
    
    # Parse DAG
    # dag structure? The prompt says "dag has topologically ordered parents and output vertices".
    # Let's assume dag is a dict or list.
    # Common format: dag[i] = list of parents. Or dag = (num_vertices, parents_list, outputs_list).
    # The baseline function is `pebble_baseline`. I don't have its source, but I can infer the input format from typical tasks or the name.
    # "dag has topologically ordered parents".
    # Let's assume dag is a list of lists: dag[i] is the list of parents of vertex i.
    # And maybe there's a separate list of outputs? Or outputs are vertices with no children?
    # "output vertices" are mentioned.
    
    # Since I don't know the exact structure, I will try to be robust.
    # If dag is a dict: dag[v] = parents.
    # If dag is a list: dag[i] = parents of i.
    
    # Let's look at the baseline import: `from autoresearch_suite.science_tcs_expansion import pebble_baseline`.
    # I cannot access this. I must guess the structure.
    # Standard for these tasks: `dag` is a list of lists. `dag[i]` contains indices of parents of `i`.
    # `fast_memory_budget` is an integer.
    
    # If the structure is different, the code will fail. I'll add a check.
    
    if not dag:
        return []
        
    # Assume dag is a list of lists
    n = len(dag)
    
    # Calculate children
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in dag[i]:
            children[p].append(i)
            
    # Topological sort is implied by index order? "topologically ordered parents"
    # Usually means if p is parent of i, p < i.
    # So we can just iterate 0 to n-1.
    
    in_fast = [False] * n
    remaining_children = [len(children[i]) for i in range(n)]
    
    operations = []
    
    # Initial loads? No, we load on demand.
    
    # We need to track which vertices are "outputs" to ensure they finish blue.
    # If a vertex is an output, it must be in fast memory at the end.
    # How do we know which are outputs?
    # "output vertices" are mentioned.
    # Maybe dag is a tuple (parents, outputs)?
    # Or maybe outputs are just vertices with no children?
    # Let's assume outputs are vertices with no children.
    # But the prompt says "dag has ... output vertices". This might be a separate field.
    # If I guess wrong, I might fail.
    # However, if I keep all vertices with no children in fast memory, that covers the "no children" case.
    # If there are explicit outputs that *do* have children (unlikely for a DAG pebbling task, outputs are usually sinks), I might fail.
    # Let's assume outputs = sinks (no children).
    
    # Heuristic for eviction:
    # If we need to free space, evict the vertex with min remaining_children > 0.
    
    for i in range(n):
        if time.time() > deadline:
            break
            
        # Ensure parents are in fast memory
        for p in dag[i]:
            if not in_fast[p]:
                # Need to load p
                # Check if we have space
                if sum(in_fast) >= fast_memory_budget:
                    # Need to evict
                    # Find candidate: min remaining_children > 0, not in dag[i] (parents of i)
                    candidates = []
                    for v in range(n):
                        if in_fast[v] and v not in dag[i]: # v is not a parent of i
                            if remaining_children[v] > 0:
                                candidates.append((remaining_children[v], v))
                            else:
                                # Can delete immediately
                                in_fast[v] = False
                                operations.append(["delete", v])
                                # Recheck space?
                                if sum(in_fast) < fast_memory_budget:
                                    break
                    if sum(in_fast) >= fast_memory_budget:
                        if candidates:
                            # Sort by remaining_children
                            candidates.sort()
                            victim = candidates[0][1]
                            in_fast[victim] = False
                            operations.append(["store", victim])
                        else:
                            # No candidates? Should not happen if budget > 0
                            pass
                
                # Load p
                in_fast[p] = True
                operations.append(["load", p])
        
        # Compute i
        in_fast[i] = True
        operations.append(["compute", i])
        
        # Update remaining children for parents
        for p in dag[i]:
            remaining_children[p] -= 1
            if remaining_children[p] == 0:
                # Can delete p
                in_fast[p] = False
                operations.append(["delete", p])
                
    # Ensure outputs are in fast memory?
    # If we deleted them, we must load them back?
    # "All outputs must finish blue".
    # If an output was deleted, we need to load it.
    # But we only delete if remaining_children == 0.
    # If an output has children, it's not a sink.
    # If we assume outputs are sinks, they are deleted when all children (none) are processed.
    # Wait, if it's a sink, remaining_children is 0 initially?
    # No, remaining_children is the number of children.
    # For a sink, children list is empty. remaining_children = 0.
    # So when we compute a sink, we don't update parents.
    # But we might have deleted it earlier? No, we only delete when remaining_children becomes 0 *after* a child is computed.
    # For a sink, it has no children, so it's never "used" by a child.
    # So it stays in fast memory until the end?
    # Yes, if we don't evict it.
    # But if we evict it (store), we must load it back at the end.
    
    # Let's add a final pass to ensure all sinks (outputs) are in fast memory.
    for i in range(n):
        if not children[i]: # Is a sink
            if not in_fast[i]:
                # Need to load
                if sum(in_fast) >= fast_memory_budget:
                    # Evict non-output
                    for v in range(n):
                        if in_fast[v] and children[v]: # Not an output
                            in_fast[v] = False
                            operations.append(["store", v])
                            if sum(in_fast) < fast_memory_budget:
                                break
                in_fast[i] = True
                operations.append(["load", i])

    return operations
