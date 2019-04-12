from sklearn import datasets, linear_model, cross_validation, preprocessing, decomposition, manifold
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np
from models.linear_models import get_features, sample_train
import torch
from torch.autograd import Variable
import random
import os
from utils.AverageMeter import AverageMeter
import utils.loc as loc
import utils.get_data as gd
import numpy

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''Parameters'''
way = 20
Model = loc.resnet_model_dir

def es(preX, prey, feats, targets, test_x, test_y, shot=1, way=5):
    model = linear_model.LogisticRegression(random_state=1)
    model.fit(preX, prey)
    W = model.coef_.T
    preX = preX.numpy()
    V = np.linalg.lstsq(preX, W.T)[0].T
    W_new = np.dot(test_x, V.T).T
    correct = 0
    for i, (ys, x) in enumerate(zip(targets, feats)):
        x = x.numpy()
        ys = ys.numpy()
        if np.argmax(np.dot(x.T, W_new)) == ys:
            correct += 1
    acc = correct / float(i)
    print("es, shot[{}], way[{}], acc[{}]".format(shot, way, acc))
    return acc

def pca_es(preX, prey, feats, targets, test_x, test_y, shot=1, way=5, compo = 0):
    model = linear_model.LogisticRegression(random_state=1)
    model.fit(preX, prey)
    W = model.coef_.T
    preX = preX.numpy()
    pca = decomposition.PCA(n_components=compo)
    #lle = manifold.LocallyLinearEmbedding(n_components=50, random_state=1)
    preX = pca.fit_transform(preX)
    # preX_lle = lle.fit_transform(preX)
    V = np.linalg.lstsq(preX, W.T)[0].T
    test_x = pca.transform(test_x)
    # test_x = lle.transform(test_x)
    W_new = np.dot(test_x, V.T).T
    correct = 0
    for i, (ys, x) in enumerate(zip(targets, feats)):
        x = x.numpy()
        ys = ys.numpy()
        if np.argmax(np.dot(x.T, W_new)) == ys:
            correct += 1
    acc = correct / float(i)
    print("es, shot[{}], way[{}], acc[{}]".format(shot, way, acc))
    return acc


def lle_es(preX, prey, feats, targets, test_x, test_y, shot=1, way=5, compo = 0):
    model = linear_model.LogisticRegression(random_state=1)
    model.fit(preX, prey)
    W = model.coef_.T
    preX = preX.numpy()
    lle = manifold.LocallyLinearEmbedding(n_components=50, random_state=1)
    preX = lle.fit_transform(preX)
    V = np.linalg.lstsq(preX, W.T)[0].T
    test_x = lle.transform(test_x)
    W_new = np.dot(test_x, V.T).T
    correct = 0
    for i, (ys, x) in enumerate(zip(targets, feats)):
        x = x.numpy()
        ys = ys.numpy()
        if np.argmax(np.dot(x.T, W_new)) == ys:
            correct += 1
    acc = correct / float(i)
    print("es, shot[{}], way[{}], acc[{}]".format(shot, way, acc))
    return acc


def get_features(sample_size = 10, way = 5, Model = loc.inception_model_dir):
    targets = []
    for i in range(way):
        targets += [i] * sample_size
    targets = torch.FloatTensor(targets)
    feats = (gd.get_image_featrues(loc.test_dir, sample_size=sample_size, way=way, Model = Model))
    return feats, targets

def get_train_features(sample_size = 10, way = 80, Model = loc.inception_model_dir):
    targets = []
    for i in range(way):
        targets += [i]
    targets = torch.FloatTensor(targets)
    feats = (gd.get_image_featrues_es(loc.train_dir, sample_size=sample_size, way=way, Model = Model))
    return feats, targets

def sample_train(feats, shot=1, way = 5):
    X = []
    y = []
    n = feats.shape[0]
    for i in range(way):
        y += [i]
        x = []
        x.append(feats[random.sample(range(int(n / way * i), int(n/way * (i+1))), shot)])
        x = torch.cat(x)
        x_mean = torch.mean(x, dim=0, keepdim=True)
        X.append(x_mean)

    y = torch.FloatTensor(y)
    X = torch.cat(X)
    assert X.shape[0] == y.shape[0]
    return X.numpy(), y.numpy()


def run(feats = [0], targets = [0], preX = [0], prey=[0], way=5, components=1, decay = 'pca'):
    acc_es_f_5s = AverageMeter()
    acc_es_f_1s = AverageMeter()
    X_f_1_size = 0
    X_f_5_size = 0
    feat_size = feats.shape[0]

    for i in range(20):
        X, y = sample_train(feats=feats, shot=5, way=way)
        if decay == 'pca':
            acc_es_f_5 = pca_es(preX, prey, feats, targets, test_x=X, test_y=y, shot=5, way=way, compo=components)
        elif decay == 'lle':
            acc_es_f_5 = lle_es(preX, prey, feats, targets, test_x=X, test_y=y, shot=5, way=way, compo=components)
        else:
            acc_es_f_5 = es(preX, prey, feats, targets, test_x=X, test_y=y, shot=5, way=way)
        acc_es_f_5s.update(acc_es_f_5)
        X_f_5_size = X.shape[0]

        X, y = sample_train(feats=feats, shot=1, way=way)
        if decay == 'pca':
            acc_es_f_1 = pca_es(preX, prey, feats, targets, test_x=X, test_y=y, shot=1, way=way, compo=components)
        elif decay == 'lle':
            acc_es_f_1 = lle_es(preX, prey, feats, targets, test_x=X, test_y=y, shot=1, way=way, compo=components)
        else:
            acc_es_f_1 = es(preX, prey, feats, targets, test_x=X, test_y=y, shot=1, way=way)
        acc_es_f_1s.update(acc_es_f_1)
        X_f_1_size = X.shape[0]

    print(way, "way   model:", Model)

    print("acc_es_f_1: ", acc_es_f_1s.avg)
    print()

    print("acc_es_f_5: ", acc_es_f_5s.avg)
    print()

    return acc_es_f_1s.avg, acc_es_f_5s.avg


if __name__ == '__main__':
    feats, targets = get_features(sample_size=600, way=way, Model=Model)
    preX, prey = get_train_features(sample_size=550, way=80, Model=Model)
    print(preX.shape[0])
    print(feats.shape[0])

    # compo_range = [1, 5, 10, 20, 30, 40, 50, 60, 70]
    # a1_pca = []
    # a5_pca = []
    # a1_lle = []
    # a5_lle = []
    # for compo in compo_range:
    #     apca1, apca5 = run(feats, targets, preX, prey, way, compo, 'pca')
    #     alle1, alle5 = run(feats, targets, preX, prey, way, compo, 'lle')
    #     a1_pca.append([apca1])
    #     a5_pca.append([apca5])
    #     a1_lle.append([alle1])
    #     a5_lle.append([alle5])
    # ac1 , ac5 = run(feats, targets, preX, prey, way, compo, 'nodecay')
    # a1_pca.append([ac1])
    # a1_lle.append([ac1])
    # a5_pca.append([ac5])
    # a5_lle.append([ac5])
    # print('1shot  pca', a1_pca)
    # print('5shot  pca', a5_pca)
    # print('1shot  lle', a1_lle)
    # print('5shot  lle', a5_lle)

    ac1, ac5 = run(feats, targets, preX, prey, way, components=10, decay='pca')

    print("1shot pca10 resnet", ac1, '  way:', way)
    print("5shot pca10 resnet", ac5, '  way:', way)






