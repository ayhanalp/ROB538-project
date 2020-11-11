
import pickle 
import numpy as np

with open("DATA.pkl",'rb') as f:
    states,history,nets=pickle.load(f)
    print(states[0])
    print(nets[0].hiddenToOutMat)
    a=nets[0].get_action(states[0][0])
    print(np.array(a))
