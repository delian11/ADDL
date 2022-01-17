import os
import itertools
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_features(features, labels, num_classes, epoch, plot_dir, dirname, prefix):
    """Plot features on 2D plane.
    Args:
        features: (num_instances, feature_dim). 3589*256(tuple)
        labels: (num_instances). 
    """
    prev_time = datetime.now()
    feat_pca = PCA(n_components=50).fit_transform(features)
    feat_pca_tsne = TSNE(n_components=2).fit_transform(feat_pca)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    for label_idx in range(num_classes):
        plt.scatter(
            feat_pca_tsne[labels==label_idx, 0],
            feat_pca_tsne[labels==label_idx, 1],
            c=colors[label_idx],
            s=4,
        )
    font1 = {
        'family': 'Times New Roman',
        'weight': 'normal'
    }
    plt.yticks(fontproperties = 'Times New Roman')
    plt.xticks(fontproperties = 'Times New Roman')

    if dirname == 'CK+':
        plt.legend(['Angry', 'Surprise', 'Disgust', 'Fear', 'Happy', 'Sad', 'Contempt'], loc='best', markerscale=1.5, prop=font1)
        # plt.legend(['Angry', 'Surprise', 'Disgust', 'Fear', 'Happy', 'Sad', 'Contempt'], loc='upper right', markerscale=1.5, prop=font1)
    elif dirname in ['MMI', 'OULU']:
        plt.legend(['Angry', 'Surprise', 'Disgust', 'Fear', 'Happy', 'Sadness'], loc='best', markerscale=1.5, prop=font1)
        # plt.legend(['Angry', 'Surprise', 'Disgust', 'Fear', 'Happy', 'Sadness'], loc='upper right', markerscale=1.5, prop=font1)
    elif dirname == 'RAF':
        plt.legend(['Surprise', 'Fear', 'Digust', 'Happy', 'Sad', 'Angry', 'Neutral'], loc='best', markerscale=1.5, prop=font1)
        # plt.legend(['Surprise', 'Fear', 'Digust', 'Happy', 'Sad', 'Angry', 'Neutral'], loc='upper right', markerscale=1.5, prop=font1)
    elif dirname == 'SFEW':
        plt.legend(['Angry', 'Digust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'], loc='best', markerscale=1.5, prop=font1)
        # plt.legend(['Angry', 'Digust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'], loc='upper right', markerscale=1.5, prop=font1)

    save_path = os.path.join(plot_dir, dirname)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        

    save_name = os.path.join(save_path, prefix + '_best.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
    print('{} saved.\t{}'.format(save_name, time_str))


def plot_confusion_matrix(cm,class_names,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

