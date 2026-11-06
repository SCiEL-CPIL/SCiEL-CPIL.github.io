                    if len(fast_mem) >= fast_memory_budget:
                        evicted = _evict(fast_mem, slow_mem, remaining_uses, outputs, v, p, children, topo_order)
                        if evicted is not None:
                            fast_mem.discard(evicted)
                            # Check if we need to store or delete
                            if (remaining_uses.get(evicted, 0) > 0) or (evicted in outputs):
                                # Store to slow memory
                                if evicted not in slow_mem:
                                    ops.append(["store", evicted])
                                    slow_mem.add(evicted)
                            else:
                                # Delete completely
                                ops.append(["delete", evicted])
                                if evicted in slow_mem:
                                    slow_mem.discard(evicted)
