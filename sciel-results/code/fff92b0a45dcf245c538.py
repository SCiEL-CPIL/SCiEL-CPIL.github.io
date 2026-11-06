<<<<<<< SEARCH
    def _grow_initial(rng):
        """Generate an initial self-avoiding walk."""
        chain = [[0, 0]]
        pos_map = {(0, 0): 0}
        occupied = {(0, 0)}
        
        for i in range(1, n):
            last = tuple(chain[-1])
            options = []
            for axis in range(2):
                for step in (-1, 1):
                    q = list(last)
                    q[axis] += step
                    q = tuple(q)
                    if q not in occupied:
                        options.append(q)
            
            if not options:
                return None
            
            site = rng.choice(options)
            chain.append(list(site))
            pos_map[site] = i
            occupied.add(site)
        
        return chain, pos_map
=======
    def _grow_initial(rng):
        """Generate a compact initial self-avoiding walk (spiral-like)."""
        chain = []
        pos_map = {}
        occupied = set()
        
        # Simple spiral growth to create a compact start
        # Directions: Right, Down, Left, Up, then repeat with larger steps
        # We'll just use a greedy compact growth: prefer turns that keep us close to origin
        x, y = 0, 0
        chain.append([x, y])
        pos_map[(x, y)] = 0
        occupied.add((x, y))
        
        # Directions: 0:Right, 1:Down, 2:Left, 3:Up
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        current_dir = 0
        
        for i in range(1, n):
            # Try to turn right first, then go straight, then turn left
            # This creates a spiral
            found = False
            for offset in [1, 0, 3]:
                next_dir = (current_dir + offset) % 4
                dx, dy = dirs[next_dir]
                nx, ny = x + dx, y + dy
                if (nx, ny) not in occupied:
                    x, y = nx, ny
                    current_dir = next_dir
                    found = True
                    break
            
            if not found:
                # If stuck, try any direction
                for offset in range(4):
                    next_dir = (current_dir + offset) % 4
                    dx, dy = dirs[next_dir]
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in occupied:
                        x, y = nx, ny
                        current_dir = next_dir
                        found = True
                        break
            
            if not found:
                return None
            
            chain.append([x, y])
            pos_map[(x, y)] = i
            occupied.add((x, y))
        
        return chain, pos_map
>>>>>>> REPLACE

