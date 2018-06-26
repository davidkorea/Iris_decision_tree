from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
# from IPython.display import Image
import pydotplus

def visualization(dt_model, depth):
    dot_data = StringIO()
    export_graphviz(dt_model, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # Image(graph.create_png())
    graph.write_png('./plot/StringIO_tree_depth_{}.png'.format(depth))