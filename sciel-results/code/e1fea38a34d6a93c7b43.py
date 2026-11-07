<<<<<<< SEARCH
                host_map[tv] = h
                used_hosts.add(h)
                if backtrack(pos + 1):
                    return True
                del host_map[tv]
                used_hosts.remove(h)
            
            return False
=======
                host_map[tv] = h
                used_mask |= (1 << h)
                if backtrack(pos + 1):
                    return True
                del host_map[tv]
                used_mask &= ~(1 << h)
            
            return False
>>>>>>> REPLACE
