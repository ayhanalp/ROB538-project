import argparse
import pickle
from sklearn import tree
from sklearn.tree import export_graphviz
import numpy as np
import graphviz
import pydotplus
import io
import imageio
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


def show_tree(tree, features, path) :
    f = io.StringIO()
    export_graphviz(tree, out_file=f, feature_names=features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img = imageio.imread(path)
    plt.rcParams["figure.figsize"] = (20, 20)
    plt.imshow(img)


if __name__ == "__main__":

    import pyximport
    pyximport.install(language_level=3)

    parser = argparse.ArgumentParser("Create decision trees")

    parser.add_argument("--data_file", type=str)

    args = parser.parse_args()
    data_file = args.data_file

    with open(data_file, "rb") as f:
        states, history, nets = pickle.load(f)

    num_actions = 4
    num_agents = len(states[0])

    state_data_for_action = [[] for _ in range(num_actions)]
    agent_data_for_action = [[] for _ in range(num_actions)]

    state_data_for_agent = [[] for _ in range(num_agents)]
    action_data_for_agent = [[] for _ in range(num_agents)]

    while True:

        try:
            timestep = states.pop()
        except IndexError:
            print("All time steps parsed")
            break

        for rover_i, rover_state in enumerate(timestep):

            action = nets[rover_i].get_action(rover_state)

            state_data_for_action[action].append(rover_state)
            agent_data_for_action[action].append(rover_i)

            state_data_for_agent[rover_i].append(rover_state)
            action_data_for_agent[rover_i].append(action)

    for i in range(num_actions):

        print("Action number:", i, ", Samples:", len(state_data_for_action[i]))

        x = np.array(state_data_for_action[i])
        y = np.array(agent_data_for_action[i])

        d_tree = tree.DecisionTreeClassifier()
        d_tree = d_tree.fit(x, y)

        dot_data = tree.export_graphviz(d_tree, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render("action_%d_tree" % i)
        features = ["a", "b", "c"]
        show_tree(d_tree, features, 'dec_tree_act' + str(i) + '.png')

    for i in range(num_agents):

        print("Agent number:", i, ", Samples:", len(state_data_for_agent[i]))

        x = np.array(state_data_for_agent[i])
        y = np.array(action_data_for_agent[i])

        d_tree = tree.DecisionTreeClassifier()
        d_tree = d_tree.fit(x, y)

        dot_data = tree.export_graphviz(d_tree, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render("agent_%d_tree" % i)

        features = ["a", "b", "c"]
        show_tree(d_tree, features, 'dec_tree_agnt' + str(i) + '.png')

