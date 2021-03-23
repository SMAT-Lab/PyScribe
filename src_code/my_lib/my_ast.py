import ast
from nltk import WordPunctTokenizer
from my_tokenizer import tokenize_python
def code2edges(code):
    edges=[]
    expr_ast=ast.parse(code)
    for node in ast.walk(expr_ast):
        str_node = ast.dump(node)
        str_node=str_node[:str_node.index('(')]
        for child in ast.iter_child_nodes(node):
            str_child = ast.dump(child)
            str_child=str_child[:str_child.index('(')]
            # print(str_node[:str_node.index('(')], str_child[:str_child.index('(')])
            edges.append((str_node,str_child))
    return edges

def code2ast_info(code,attribute=None):
    edge_starts = []
    edge_ends = []
    depths = []
    subtree_poses = []
    subling_poses = []
    # attribute_poses=[]
    expr_ast=ast.parse(code)
    # depth=0
    edge_depth_queue = [1]*len(list(ast.iter_child_nodes(expr_ast)))  # 边的深度的队列
    node_depth_queue=[0]*1
    subtree_pos=-1
    attr_depth=-1
    for node in ast.walk(expr_ast):
        str_node = ast.dump(node)
        str_node=str_node[:str_node.index('(')]

        subtree_pos+=1
        if node_depth_queue[0] > attr_depth:
            subtree_pos = 0
        attr_depth = node_depth_queue.pop(0)
        child_node_num = len(list(ast.iter_child_nodes(node)))

        node_depth_queue.extend([attr_depth + 1] * max(child_node_num, 1))

        if attribute is not None:
            for i,(_, attr_value) in enumerate(ast.iter_fields(node)):
                if not (isinstance(attr_value, ast.AST) or isinstance(attr_value, list) or attr_value is None):
                    if (attribute=='str' and isinstance(attr_value,str)) or attribute=='all':
                        if not repr(attr_value).isdigit():
                            tokens = tokenize_python(repr(attr_value).replace('_',' ').lower(),keep_puc=False)
                            for j,token in enumerate(tokens):
                                edge_starts.append(str_node)
                                edge_ends.append(token)
                                depths.append(attr_depth + 1)
                                subtree_poses.append(subtree_pos)
                                subling_poses.append(-(i + 1+j))
                        else:
                            edge_starts.append(str_node)
                            edge_ends.append(repr(attr_value))
                            depths.append(attr_depth+1)
                            subtree_poses.append(subtree_pos)
                            subling_poses.append(-(i+1))
                        # pass

        # elif attribute=='name':
        #     for i,(_, attribute) in enumerate(ast.iter_fields(node)):
        #         if isinstance(attribute,str):
        #             pass

        for i,child in enumerate(ast.iter_child_nodes(node)):
            # if i==1:
            #     print(ast.dump(child))
            edge_starts.append(str_node)
            str_child = ast.dump(child)
            str_child=str_child[:str_child.index('(')]
            edge_ends.append(str_child)
            depths.append(edge_depth_queue.pop(0))
            edge_depth_queue.extend([depths[-1] + 1] * len(list(ast.iter_child_nodes(child))))  # 如果当前点为树，则将其下所有边的深度值加入深度值队列
            # if len(depths) == 1 or (len(depths) > 1 and depths[-1] > depths[-2]):
            #     subtree_pos = 0
                # subtree_poses.append(subtree_pos)
            # else:
            #     # global_position += 1
            #     subtree_poses.append(subtree_pos)
            subtree_poses.append(subtree_pos)
            subling_poses.append(i)
            # print(str_node[:str_node.index('(')], str_child[:str_child.index('(')])
            # edges.append((str_node,str_child))
        # print(str_node,subtree_pos)
        # print(subtree_pos,sep=',')
    return edge_starts,edge_ends,depths,subtree_poses,subling_poses

# def code2ast_info(code):
#     edge_starts = []
#     edge_ends = []
#     depths = []
#     global_positions = []
#     local_positions = []
#     expr_ast=ast.parse(code)
#     # depth=0
#     depth_queue = [1]*len(list(ast.iter_child_nodes(expr_ast)))  # 边的深度的队列
#     for node in ast.walk(expr_ast):
#         str_node = ast.dump(node)
#         str_node=str_node[:str_node.index('(')]
#
#         for i,child in enumerate(ast.iter_child_nodes(node)):
#             edge_starts.append(str_node)
#             str_child = ast.dump(child)
#             str_child=str_child[:str_child.index('(')]
#             edge_ends.append(str_child)
#             depths.append(depth_queue.pop(0))
#             depth_queue.extend([depths[-1] + 1] * len(list(ast.iter_child_nodes(child))))  # 如果当前点为树，则将其下所有边的深度值加入深度值队列
#             if len(depths) == 1 or (len(depths) > 1 and depths[-1] > depths[-2]):
#                 global_position = 0
#                 global_positions.append(global_position)
#             else:
#                 global_position += 1
#                 global_positions.append(global_position)
#             local_positions.append(i)
#             # print(str_node[:str_node.index('(')], str_child[:str_child.index('(')])
#             # edges.append((str_node,str_child))
#     return edge_starts,edge_ends,depths,global_positions,local_positions

if __name__=='__main__':
    import astunparse
    code="a=b*c"
    expr_ast = ast.parse(code)
    # print(ast.size)
    print(len(list(ast.iter_child_nodes(expr_ast))))
    # print(astunparse.dump(expr_ast))
    edge_starts,edge_ends,depths,subtree_poses,subling_poses=code2ast_info(code,attribute='all')
    print(edge_starts)
    print(edge_ends)
    print(depths)
    print(subtree_poses)
    print(subling_poses)

