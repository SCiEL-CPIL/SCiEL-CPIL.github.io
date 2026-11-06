<<<<<<< SEARCH
        # Generate successor states
        # 1. COMPUTE: for each vertex not yet computed, if all parents are red, we can compute it
        for v in range(n_vertices):
            if v in red or v in blue:
                continue  # already computed
            # Check if all parents are red
            if all(p in red for p in parents[v]):
                new_red = red | frozenset([v])
                new_blue = blue
                new_state = (new_red, new_blue)
                new_cost = cost  # compute is free
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent_map[new_state] = (state, 'compute', v)
                    step_counter += 1
                    heapq.heappush(heap, (new_cost, step_counter, new_red, new_blue))
        
        # 2. LOAD: bring a vertex from slow to fast memory
        # A vertex can be loaded if it's blue (computed) and not already red
        for v in blue:
            if v not in red:
                new_red = red | frozenset([v])
                new_blue = blue
                new_state = (new_red, new_blue)
                new_cost = cost + 1
                if new_cost < dist.get(new_state, float('inf')):
                    dist[new_state] = new_cost
                    parent_map[new_state] = (state, 'load', v)
                    step_counter += 1
                    heapq.heappush(heap, (new_cost, step_counter, new_red, new_blue))
        
        # 3. STORE: move a vertex from fast to slow memory
        for v in red:
            new_red = red - frozenset([v])
            new_blue = blue | frozenset([v])
            new_state = (new_red, new_blue)
            new_cost = cost + 1
            if new_cost < dist.get(new_state, float('inf')):
                dist[new_state] = new_cost
                parent_map[new_state] = (state, 'store', v)
                step_counter += 1
                heapq.heappush(heap, (new_cost, step_counter, new_red, new_blue))
        
        # 4. DELETE: remove a red pebble (without storing)
        for v in red:
            new_red = red - frozenset([v])
            new_blue = blue
            new_state = (new_red, new_blue)
            new_cost = cost  # delete is free
            if new_cost < dist.get(new_state, float('inf')):
                dist[new_state] = new_cost
                parent_map[new_state] = (state, 'delete', v)
                step_counter += 1
                heapq.heappush(heap, (new_cost, step_counter, new_red, new_blue))
=======
        # Generate successor states
        # 1. COMPUTE: for each vertex not yet computed, if all parents are red, we can compute it
        for v in range(n_vertices):
            if v in red or v in blue:
                continue  # already computed
            # Check if all parents are red
            if all(p in red for p in parents[v]):
                if len(red) < fast_memory_budget: # Check budget before adding
                    new_red = red | frozenset([v])
                    new_blue = blue
                    new_state = (new_red, new_blue)
                    new_cost = cost  # compute is free
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        parent_map[new_state] = (state, 'compute', v)
                        step_counter += 1
                        heapq.heappush(heap, (new_cost, step_counter, new_red, new_blue))
        
        # 2. LOAD: bring a vertex from slow to fast memory
        # A vertex can be loaded if it's blue (computed) and not already red
        for v in blue:
            if v not in red:
                if len(red) < fast_memory_budget: # Check budget before adding
                    new_red = red | frozenset([v])
                    new_blue = blue
                    new_state = (new_red, new_blue)
                    new_cost = cost + 1
                    if new_cost < dist.get(new_state, float('inf')):
                        dist[new_state] = new_cost
                        parent_map[new_state] = (state, 'load', v)
                        step_counter += 1
                        heapq.heappush(heap, (new_cost, step_counter, new_red, new_blue))
        
        # 3. STORE: move a vertex from fast to slow memory
        for v in red:
            new_red = red - frozenset([v])
            new_blue = blue | frozenset([v])
            new_state = (new_red, new_blue)
            new_cost = cost + 1
            if new_cost < dist.get(new_state, float('inf')):
                dist[new_state] = new_cost
                parent_map[new_state] = (state, 'store', v)
                step_counter += 1
                heapq.heappush(heap, (new_cost, step_counter, new_red, new_blue))
        
        # 4. DELETE: remove a red pebble (without storing)
        for v in red:
            new_red = red - frozenset([v])
            new_blue = blue
            new_state = (new_red, new_blue)
            new_cost = cost  # delete is free
            if new_cost < dist.get(new_state, float('inf')):
                dist[new_state] = new_cost
                parent_map[new_state] = (state, 'delete', v)
                step_counter += 1
                heapq.heappush(heap, (new_cost, step_counter, new_red, new_blue))
>>>>>>> REPLACE

