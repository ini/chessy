import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def get_data_frame(X, y):
	data = pd.DataFrame(X)
	data['y'] = y
	data['y'] = data['y'].astype(str)
	return data


def get_data_frame_subset(X, y, num_points):
	rndperm = np.random.permutation(X.shape[0])[:num_points]
	X = X[rndperm, :]
	y = y[rndperm]
	return get_data_frame(X, y)


def plot(data, results, palette='hls', alpha=0.3):
	num_labels = len(set(data['y']))
	plt.figure(figsize=(16, 10))
	sns.scatterplot(
	    x=results[:, 0], y=results[:, 1],
	    hue='y',
	    palette=sns.color_palette(palette, num_labels),
	    data=data,
	    legend='full',
	    alpha=alpha,
	)
	plt.show()


def pca(X, y, n_components=2):
	data = get_data_frame(X, y)
	pca_results = PCA(n_components=n_components).fit_transform(X)
	plot(data, pca_results)


def tsne(X, y, n_tsne_points=None, n_components=2, verbose=False, perplexity=40, n_iter=300):
	data = get_data_frame_subset(X, y, n_tsne_points)
	tsne_results = TSNE(
		n_components=n_components, verbose=verbose, 
		perplexity=perplexity, n_iter=n_iter).fit_transform(X)
	plot(data, tsne_results)



X = np.load('embeddings.npy')
X = X.reshape(X.shape[0], -1)
y = [name[:-len('.pgn')] for name in sorted(os.listdir('./players/')) if name.endswith('.pgn')]
assert len(X) == len(y)

tsne(X, y, verbose=True)