from model import make_pillar_feature_net
import numpy as np

P = 2
C = 64
N = 100
D = 9

x = np.random.random((P, N, D))
y = np.random.random((P, 1, C))

pillar_feature_net = make_pillar_feature_net()

pillar_feature_net.compile('sgd', 'mse')
pillar_feature_net.fit(x, y)

pillar_feature_net.save('Model/pointpillars' + '_C' + str(C) + '_N' + str(N) + '.h5')
print('Done!')



