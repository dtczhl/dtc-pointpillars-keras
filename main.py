from model import make_pillar_feature_net, make_backbone
import numpy as np

P = 2
C = 64
N = 100
D = 9

# x = np.random.random((P, N, D))
# y = np.random.random((P, 1, C))

pillar_feature_net = make_pillar_feature_net()

pillar_feature_net.compile('sgd', 'mse')
pillar_feature_net.summary()
pillar_feature_net.save('Model/pointpillars_pfn' + '_C' + str(C) + '_N' + str(N) + '.h5')
print('Pillar PFN Done!')


backbone_model = make_backbone(80, 64, 64)
backbone_model.compile('sgd', 'mse')
backbone_model.summary()
backbone_model.save('Model/pointpillars_backbone' + '_W' + str(80) + '_H' + str(64) + '_C' + str(64) + '.h5')
print('Pillar Backbone Done!')





