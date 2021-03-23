from nltk.tree import Tree

def nltk_tree2edge_info(tree):
    '''
    将一棵树转化为边的信息
    :param tree: NLTK-tree
    :return: 边的起点列表，边的终点列表，边的深度值列表，边在当前层全局位置列表，边在当前子数中位置列表
    '''
    edge_starts=[]
    edge_ends=[]
    depths = []
    global_positions=[]
    local_positions=[]
    tree_queue = [tree] #树的队列，为了广度优先遍历
    depth_queue=[0]*len(tree)   #边的深度的队列
    # nodes=[]
    while tree_queue:
        tree=tree_queue.pop(0)
        if isinstance(tree,Tree):
            tree_label=tree.label()
            # nodes.append(tree_label)
            for i,subtree in enumerate(tree):
                depths.append(depth_queue.pop(0))   #弹出当前边深度值加入边的深度列表
                edge_starts.append(tree_label)
                if isinstance(subtree,Tree):
                    edge_ends.append(subtree.label())
                    depth_queue.extend([depths[-1]+1]*len(subtree)) #如果当前点为树，则将其下所有边的深度值加入深度值队列
                elif isinstance(subtree,str):
                    edge_ends.append(subtree)
                if len(depths)==1 or (len(depths)>1 and depths[-1]>depths[-2]):
                    global_position=0
                    global_positions.append(global_position)
                else:
                    global_position+=1
                    global_positions.append(global_position)
                local_positions.append(i)
                tree_queue.append(subtree)
        # elif isinstance(tree,str):
        #     nodes.append(tree)

    # print(edge_starts)
    # print(edge_ends)
    # print(depths)
    # print(global_positions)
    # print(local_positions)
    # print()
    return edge_starts,edge_ends,depths,global_positions,local_positions