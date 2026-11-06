                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) in edge_set:
                    continue
                if not creates_c6(u, v):
                    add_edge(u, v)
                    success = True
                    break
