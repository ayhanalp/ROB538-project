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
import pandas as pd
from sklearn.tree import export_text


def show_tree(tree, features, c_names,  path) :
    f = io.StringIO()
    export_graphviz(tree, out_file=f, feature_names=features, class_names=c_names)
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

        d_tree = tree.DecisionTreeClassifier(max_depth=4)
        d_tree = d_tree.fit(x, y)


        '''
        # To export DT as text
        r = export_text(d_tree)
        print(r)
        '''

        '''
        sensors are divided into 90 deg segments with directions NE, NW, SW, SE
        we have the following:
        agent sensor (4)
        poi type A sensor(4)
        poi type B sensor(4)
        poi type C sensor(4)
        poi type D sensor(4)
        List of types that must be observed(4)
        the types that have been observed so far(4)
        so the input should be size 28
        '''

        dot_data = tree.export_graphviz(d_tree, precision=2, proportion=True, out_file=None)
        graph = graphviz.Source(dot_data)
        # graph.render("action_%d_tree" % i)

        features = ["Agent Sensor - NE", "Agent Sensor - NW", "Agent Sensor - SW", "Agent Sensor - SE",
                    "POI Type A Sensor - NE", "POI Type A Sensor - NW", "POI Type A Sensor - SW",
                    "POI Type A Sensor - SE",
                    "POI Type B Sensor - NE", "POI Type B Sensor - NW", "POI Type B Sensor - SW",
                    "POI Type B Sensor - SE",
                    "POI Type C Sensor - NE", "POI Type C Sensor - NW", "POI Type C Sensor - SW",
                    "POI Type C Sensor - SE",
                    "POI Type D Sensor - NE", "POI Type D Sensor - NW", "POI Type D Sensor - SW",
                    "POI Type D Sensor - SE",
                    "Type A (t.b.o.)", "Type B (t.b.o.)", "Type C (t.b.o.)", "Type D (t.b.o.)",
                    "Type A (h.b.o.)", "Type A (h.b.o.)", "Type A (h.b.o.)", "Type A (h.b.o.)"]

        class_names = ["Rover_0", "Rover_1", "Rover_2", "Rover_3", "Rover_4", "Rover_5", "Rover_6", "Rover_7"]
        agents_seen = np.unique(y)
        classes = [class_names[agent] for agent in agents_seen]
        show_tree(d_tree, features, classes, 'DT_Figures/dec_tree_act' + str(i) + '.png')

    for i in range(num_agents):

        print("Agent number:", i, ", Samples:", len(state_data_for_agent[i]))

        x = np.array(state_data_for_agent[i])
        y = np.array(action_data_for_agent[i])

        d_tree = tree.DecisionTreeClassifier()
        d_tree = d_tree.fit(x, y)

        '''
        # To export DT as text
        r = export_text(d_tree)
        print(r)
        '''

        dot_data = tree.export_graphviz(d_tree, out_file=None)
        graph = graphviz.Source(dot_data)
        # graph.render("agent_%d_tree" % i)

        features = ["Agent Sensor - NE", "Agent Sensor - NW", "Agent Sensor - SW", "Agent Sensor - SE",
                    "POI Type A Sensor - NE", "POI Type A Sensor - NW", "POI Type A Sensor - SW",
                    "POI Type A Sensor - SE",
                    "POI Type B Sensor - NE", "POI Type B Sensor - NW", "POI Type B Sensor - SW",
                    "POI Type B Sensor - SE",
                    "POI Type C Sensor - NE", "POI Type C Sensor - NW", "POI Type C Sensor - SW",
                    "POI Type C Sensor - SE",
                    "POI Type D Sensor - NE", "POI Type D Sensor - NW", "POI Type D Sensor - SW",
                    "POI Type D Sensor - SE",
                    "Type A (t.b.o.)", "Type B (t.b.o.)", "Type C (t.b.o.)", "Type D (t.b.o.)",
                    "Type A (h.b.o.)", "Type A (h.b.o.)", "Type A (h.b.o.)", "Type A (h.b.o.)"]

        class_names = ["POI_A", "POI_B", "POI_C", "POI_D"]
        actions_seen = np.unique(y)
        classes = [class_names[action] for action in actions_seen]
        show_tree(d_tree, features, classes, 'DT_Figures/dec_tree_agnt' + str(i) + '.png')

