        # ... inside loop ...
        else:
            # Deletion failed
            action = random.random()
            if action < 0.3:
                # Modify indices
                i, j = current_network[idx]
                # Random new pair
                new_i, new_j = random.sample(range(N), 2)
                if new_i > new_j: new_i, new_j = new_j, new_i
                test_network = current_network[:]
                test_network[idx] = (new_i, new_j)
                if is_valid_network(test_network):
                    current_network = test_network
            elif action < 0.6:
                # Move to random position
                new_idx = random.randint(0, len(current_network) - 1)
                if new_idx != idx:
                    comp = current_network.pop(idx)
                    current_network.insert(new_idx, comp)
                    if is_valid_network(current_network):
                        pass # keep it
                    else:
                        # revert
                        current_network.pop(new_idx)
                        current_network.insert(idx, comp)
            # else: do nothing, just try next deletion
