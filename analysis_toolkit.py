import networkx as nx
def res_extractor(res, n, m):
    # given res as the measurements of QA, n,m return d,y,r and a DiGraph G, which could be plotted by draw_graph
    # is_cons, is_dag, is_legal are boolean variables.

    if len(res) != int(n*(n-1)*3/2 + 2*n):
        return None
    
    # Notice that '0' in res means positive flip +1 and is correspond to 1 as variables.

    res = list(res)[::-1]
    res = [1 - int(tmp) for tmp in res]

    d = res[:int(n*(n-1))]
    r = res[int(n*(n-1)): int(n*(n-1)*3/2)]
    y = res[int(n*(n-1)*3/2):]

    is_cons = True 
        # no two nodes are mutual connected and r_ij = 0 iff x_i > x_j in topo. order
    is_dag = True
        # check r_ij s.t. x_i > x_k > x_j > x_i
    is_legal = True 
        # having at most m indgree and y_i + d_i = m

    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, desc='v'+str(i))
    edge_list = []
    
    for i in range(n-1):
        for j in range(i+1,n):
            r_ij = int(i*(n-1) - i*(i+1)/2 + j - 1)
            d_ij = (n-1)*i + j - 1
            d_ji = i + (n-1)*j
            if d[d_ij] == 1 and d[d_ji] == 1:
                is_cons = False
            elif d[d_ij] == 1:
                if r[r_ij] == 1:
                    is_cons = False
                edge_list.append((i,j))
            elif d[d_ji] == 1:
                if r[r_ij] == 0:
                    is_cons = False
                edge_list.append((j,i))

    G.add_edges_from(edge_list)

    for i in range(n-2):
        for j in range(i+1, n-1):
            idx_ij = int(i*(n-1) - i*(i+1)/2 + j - 1)
            for k in range(j+1, n):
                idx_ik = int(i*(n-1) - i*(i+1)/2 + k - 1)
                idx_jk = int(j*(n-1) - j*(j+1)/2 + k - 1)
                
                circ = r[idx_ik] + r[idx_ij] * r[idx_jk] - r[idx_ij]*r[idx_ik] - r[idx_jk]*r[idx_ik]
                
                if circ > 0:
                    is_dag = False
                    break
    
    for i in range(n):
        i_parent_num = 0
        for x in edge_list:
            if x[1] == i:
                i_parent_num += 1
        slack = 2*y[2*i + 1] + y[2*i]
        if i_parent_num + slack != m:
            is_legal = False
            break
    
    return d, r, y, G, \
        is_cons, is_dag, is_legal

def draw_graph(G):
    pos = nx.spring_layout(G)
    node_labels = nx.get_node_attributes(G, 'desc')
    nx.draw_networkx(G, pos = pos, with_labels=None)
    nx.draw_networkx_labels(G, pos, node_labels)