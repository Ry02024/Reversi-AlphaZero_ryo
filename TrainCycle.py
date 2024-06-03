from ResidualNetwork import dual_network
from SelfPlay import self_play
from TrainNetwork import train_network
from EvaluateNetwork import evaluate_network


dual_network()

for i in range(5):
    print('Train',i+1,'====================')
    self_play()
    train_network()
    evaluate_network()
