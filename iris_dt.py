import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus

DATA_FILE = './Iris.csv'
iris_data = pd.read_csv(DATA_FILE)
FEAT_COLS = list(iris_data.columns)[1:-1]
CATEGRORY_LABEL_DICT = {
        'Iris-setosa':      0,  # 山鸢尾
        'Iris-versicolor':  1,  # 变色鸢尾
        'Iris-virginica':   2   # 维吉尼亚鸢尾
    }
iris_data['label'] = iris_data['Species'].apply(lambda x : CATEGRORY_LABEL_DICT[x])

X = iris_data[FEAT_COLS].values
y = iris_data['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=10)

def decision_tree_plot(dt_model,depth):
    tem_dot_file = 'decision_tree_tmp.dot'
    export_graphviz(dt_model, out_file=tem_dot_file, feature_names=FEAT_COLS,
                    class_names=list(CATEGRORY_LABEL_DICT.keys()),
                    filled=True, impurity=False)
    with open(tem_dot_file) as f:
        dot_graph = f.read()
    graph = pydotplus.graph_from_dot_data(dot_graph)
    graph.write_png('Tree_depth_{}.png'.format(depth))


def feature_importance(dt_model, depth):
    plt.figure()
    plt.barh(range(len(FEAT_COLS)), dt_model.feature_importances_)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.yticks(np.arange(len(FEAT_COLS)), FEAT_COLS)
    plt.title('Feature Importance_depth_{}'.format(depth))
    plt.savefig('./Feature Importance_depth_{}'.format(depth))
    plt.show()


def main():
    depth_list = [2,3,4]
    for d in depth_list:
        dt_model = DecisionTreeClassifier(max_depth=d)
        dt_model.fit(X_train, y_train)

        train_acc = dt_model.score(X_train, y_train)
        test_acc = dt_model.score(X_test, y_test)

        print('Accuracy on Train Set with depth {} = {:.2f}%'.format(d, train_acc*100))
        print('Accuracy on Test Set with depth {} = {:.2f}%'.format(d, test_acc*100))
        print()

        decision_tree_plot(dt_model, d)
        feature_importance(dt_model, d)

main()
