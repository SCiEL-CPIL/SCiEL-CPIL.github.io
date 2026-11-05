            # Initialization
            if attempt % 2 == 0:
                # Structured init
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        d = j - i
                        d = min(d, n - d)
                        colors.append(d % 5)
            else:
                colors = [random.randint(0, 4) for _ in range(num_edges)]
