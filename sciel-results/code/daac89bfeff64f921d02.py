def try_remove(adj):
    # pick random edge
    # maintain edge list? or pick random u, random neighbor
    u = random.randrange(n)
    if not adj[u]:
        return False
    v = random.choice(list(adj[u]))
    remove_edge(adj, u, v)
    return True
