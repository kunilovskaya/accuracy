from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import itertools
import numpy as np

## this is an ANOVA-alternative method of feature selection to try and find best features for HTQ empirically, by eliminating features one-by-one
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier  ## returns feature_importances_ attribute requires for RFE
from sklearn import preprocessing


def recursive_elimination(X, y, feats=15):
	print(set(y))
	le = preprocessing.LabelEncoder()
	for i in range(len(set(y))):
		y = le.fit_transform(y)
	print(set(y))
	clf = RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced')
	clf.fit(X, y)
	selector = RFE(clf, feats, step=1)
	selector = selector.fit(X, y)
	top_idx = [x[0] for x in enumerate(selector.support_) if x[1] == True]
	
	return top_idx


from sklearn.feature_selection import SelectKBest
### this function goes inside the get_xy function, but can be run just to get the feel of the data
def ANOVAselection(x, labels, features=None):
	# Feature selection
	ff = SelectKBest(k=features).fit(x, labels)
	newdata = SelectKBest(k=features).fit_transform(x, labels)
	top_ranked_features = sorted(enumerate(ff.scores_), key=lambda y: y[1], reverse=True)[:features]
	# print(top_ranked_features) ## it is a list of tuples (11, 3930.587580730744)
	top_ranked_features_indices = [x[0] for x in top_ranked_features]
	
	return newdata, top_ranked_features_indices


## unlike the function above this one finds best features of the input classes, not just some of them
def HTQ_get_xy(df, class_col='astatus', features=15, scaling=1, select_mode='RFE'):
	df0 = df.copy()
	Y0 = df0.loc[:, class_col].values  # returns an array of text_types for coloring
	print('set of Y values:', set(Y0))
	
	training_set = df0.drop(['afile', 'akorp', 'alang', 'astatus'], 1)
	
	if features:
		
		if select_mode == 'ANOVA':
			x0, top_feat = ANOVAselection(training_set, Y0,
			                              features)  # subsetting the data yet again deleting columns with less informative features
		if select_mode == 'RFE':
			top_feat = recursive_elimination(training_set, Y0, features)
			x0 = training_set.iloc[:, top_feat].values
		# x0, top_feat = featureselection(training_set, Y0, features=features)
		
		print('Data (X for SVM) after feature selection:', x0.shape)
		print('We use %s best features (ranked by their importance by %s):' % (features, select_mode),
		      [training_set.keys()[i] for i in top_feat], file=sys.stderr)
		metadata_df = df.loc[:, ['afile', 'akorp', 'alang', 'astatus']]
		new_df0 = training_set.iloc[:,
		          top_feat]  ## this is where I lose meta-columns that I want to keep for subsequent use and where I had an error with feats selection for PCA
		new_df = pd.concat([metadata_df, new_df0], axis=1)
	else:
		x0 = training_set.values
		print('Data (X for SVM) after feature selection:', x0.shape)
		new_df = df.copy()
	
	if scaling:
		# throw away columns not to be used
		
		## transform your data such that its distribution will have a mean value 0 and standard deviation of 1.
		## each value in the dataset will have the sample mean value subtracted, and then divided by the standard deviation of the whole dataset.
		sc = StandardScaler()
		X0 = sc.fit_transform(x0)
	
	else:
		print('Using raw unscaled X data')
		X0 = x0.values
	
	return X0, Y0, new_df

# expects an array, the original matrix for feature names (with or without metadata columns);
# the 3d numeric argument is the number of PCA components to return
## the final argument is the name of the column by which to sort the loadings, ex. 'Dim1'
import pandas as pd
from sklearn.decomposition import PCA


def HTQ_pca_transform(x, df, dims=2, best='Dim1', print_best=0):
	pca = PCA(n_components=dims)
	pca_reduced = pca.fit_transform(x)
	
	try:
		df0 = df.drop(['afile', 'akorp', 'alang', 'astatus'], 1)
	except:
		df0 = df
	if dims == 2:
		# Dump components with features to learn about loadings of each feature:
		res0 = pd.DataFrame(pca.components_, columns=df0.columns, index=['Dim1', 'Dim2'])
		res = res0.transpose()  # to enable sorting
		# get the feature that contribute most to the distinction between the corpora
		res.sort_values(best, ascending=False, inplace=True)
		if print_best:
			print(res.head(print_best))
	# print(res.head())
	if dims == 3:
		# Dump components with features to learn about loadings of each feature:
		res0 = pd.DataFrame(pca.components_, columns=df0.columns, index=['Dim1', 'Dim2', 'Dim3'])
		res = res0.transpose()  # to enable sorting
		# get the feature that contribute most to the distinction between the corpora
		res.sort_values(best, ascending=False, inplace=True)
		if print_best:
			print(res.head(print_best))
	# print(res.head())
	if dims == 4:
		# Dump components with features to learn about loadings of each feature:
		res0 = pd.DataFrame(pca.components_, columns=df0.columns, index=['Dim1', 'Dim2', 'Dim3', 'Dim4'])
		res = res0.transpose()  # to enable sorting
		# get the feature that contribute most to the distinction between the corpora
		res.sort_values(best, ascending=False, inplace=True)
		if print_best:
			print(res.head(print_best))
	
	explained_variance = pca.explained_variance_ratio_
	tot_variance = sum(explained_variance)
	print(tot_variance)
	percent = str(tot_variance).split(".")[1][:2] + '%'
	#     percent = str(explained_variance[0]).split(".")[1][:2]+'%'
	print('Variance explained: ', explained_variance)
	
	return pca_reduced, percent, res


# def plot_confusion_matrix(cm, classes, normalize=False, title='Normalized Confusion Matrix', cmap=plt.cm.Blues):
# 	print('This function is known to swap names of classes! Double-check', file=sys.stderr)
# 	if normalize:
# 		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# 	# print("Normalized confusion matrix")
# 	else:
# 		print('Confusion matrix, without normalization')
#
# 	# print(cm)
#
# 	plt.imshow(cm, interpolation='nearest', cmap=cmap)
# 	plt.title(title)
# 	plt.colorbar()
# 	tick_marks = np.arange(len(classes))
# 	plt.xticks(tick_marks, classes, rotation=45)
# 	plt.yticks(tick_marks, classes)
#
# 	fmt = '.2f' if normalize else 'd'
# 	thresh = cm.max() / 2.
# 	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
# 		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
# 		         color="white" if cm[i, j] > thresh else "black")
#
# 	plt.ylabel('True label')
# 	plt.xlabel('Predicted label')
# 	plt.tight_layout()


from sklearn.svm import SVC
import sys
from sklearn.dummy import DummyClassifier


## 'RF', 'dummy'
def crossvalidate(X, y, algo='SVM', grid=0, cv=10, class_weight='balanced'):
	skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
	
	if algo == 'SVM':
		clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', random_state=0, class_weight='balanced')
	if algo == 'RF' and grid == True:
		refit_score = 'f1_score'
		scoring = 'f1_macro'
		print('Running %s-CV gridsearch for RandomForests optimized for %s' % (cv, refit_score))
		best_par = gridsearch_RF(X, y, refit_score=refit_score, scoring=scoring)
		# max_features = best_par['max_features'],
		# max_depth = best_par['max_depth'],
		# n_estimators = best_par['n_estimators'],
		# min_samples_split = best_par['min_samples_split']
		
		clf = RandomForestClassifier(n_estimators=best_par['n_estimators'], max_depth=best_par['max_depth'],
		                             max_features=best_par['max_features'],
		                             min_samples_split=best_par['min_samples_split'],
		                             random_state=0, class_weight=class_weight)
	if algo == 'dummy':
		strategy = 'most_frequent'
		print('\n====DUMMY (%s)====' % strategy)
		clf = DummyClassifier(strategy=strategy, random_state=42) ##strategy='most_frequent' (Zero Rule Algorithm) 'uniform'
	
	scores_f = cross_val_score(clf, X, y, cv=skf, scoring='f1_macro')  # cv = loo
	scores_acc = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
	'''
    this function is simpler than cross_validate, it returns a list of scores on just one specified metric,
    rather than a dict and tons of other info in that dict
    '''
	print("F1 over 10folds: ", scores_f.mean())
	print("Accuracy over 10folds: ", scores_acc.mean())
	
	preds = cross_val_predict(clf, X, y, cv=skf)
	'''
    returns, for each element in the input, the prediction that was obtained for that element
    '''
	cfmat = confusion_matrix(y, preds)
	print('Raw confusion matrix:\n', cfmat)
	
	report = classification_report(y, preds, output_dict=True)
	print(classification_report(y, preds))
	df = pd.DataFrame(report).transpose()
	idx = np.argmin(df['support'].tolist())
	val = np.amin(df['support'].tolist())
	df = df.reset_index()
	minority = df.iloc[idx, 0]
	
	# get measures on the minority class
	minorf1 = report[minority]['f1-score']
	print('Average F1 on the minority class (%s/%s): %s' % (minority, val, minorf1))  # np.average(minorf1).round(3)


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def gridsearch_RF(x, y, refit_score='f1_score', scoring='f1_macro'):
	clf = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=0)
	
	param_grid = {
		'min_samples_split': [3, 5, 10],
		'n_estimators': [100, 300],
		'max_depth': [3, 5, 15, 25],
		'max_features': [3, 5, 10, 15]
	}
	
	"""
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
	skf = StratifiedKFold(n_splits=5)
	grid = GridSearchCV(clf, param_grid, refit=refit_score,
	                    cv=skf, scoring=scoring, n_jobs=-1)
	grid.fit(x, y)
	
	return grid.best_params_


def nese_visualizePCA(x, y, res, dimx=1, dimy=3, feats=None):
	plt.figure(figsize=(10, 10))
	plt.grid(False)
	
	FONTSIZE = 7  # size of point labels
	COLOR = 'grey'  # this is the color of the text, not of the datapoint
	XYTEXT = (4, 3)
	POINTSIZE = 70
	# english = blue C2, german = orange_C1, russian = green C2
	# any non-translation = default point, any translation = square, stu = cross, pro = circle
	
	for i, _ in enumerate(x):
		
		if y[i] == 'pro' or y[i] == 'transl':
			ruTT = plt.scatter(*x[i], s=POINTSIZE, marker='s', facecolors='none', edgecolors='C2')
		if y[i] == 'ref':
			ruNT = plt.scatter(*x[i], s=POINTSIZE, color='grey')
	
	if 'transl' in y:
		plt.legend((ruTT, ruNT),
		           ("students", "non-translations"),
		           scatterpoints=1,
		           prop={'size': 20},
		           loc="best")
	if 'pro' in y:
		plt.legend((ruTT, ruNT),
		           ("professional", "non-translations"),
		           scatterpoints=1,
		           prop={'size': 20},
		           loc="best")
	
	plt.xlabel('PCA Dimension %s' % dimx, fontsize=16)
	plt.ylabel('PCA Dimension %s' % dimy, fontsize=16)
	# plt.tight_layout()
	plt.tick_params(axis='x', size=10, labelsize=15)  # labelsize is values, size is for ticks
	# plt.yticks([]) # uncomment to get rid of any y-axis labelling
	# plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
	print('PCA reduction of %s-feature space on EN-RU pair (variance explained by D1+D2 = %s )' % (feats, res))
	# plt.title('PCA reduction of %s 23-feature space on EN-DE pair (variance explained by D1+D2 = %s )' % (data, res),
	#           fontsize=20)
	# plt.savefig('/home/masha/02targets_wlv/pics/ref_proPCA_on$s.png' % (focus,n_feats), bbox_inches='tight')
	plt.show()


def HTQ_visualizePCA(x, y, res, dimx=1, dimy=2, feats=10):
	plt.figure(figsize=(5, 5))
	plt.grid(False)
	
	FONTSIZE = 7  # size of point labels
	COLOR = 'grey'  # this is the color of the text, not of the datapoint
	XYTEXT = (4, 3)
	POINTSIZE = 70
	# english = blue C2, german = orange_C1, russian = green C2
	# any non-translation = default point, any translation = square, stu = cross, pro = circle
	if len(set(y)) == 2:
		for i, _ in enumerate(x):
			if y[i] == 'good':
				good = plt.scatter(*x[i], s=POINTSIZE, marker='o', facecolors='none', edgecolors='C2')  ##green
			if y[i] == 'bad':
				bad = plt.scatter(*x[i], s=POINTSIZE, marker='s', facecolors='none', edgecolors='red')
		plt.legend((good, bad),
		           ("good", "bad"),
		           scatterpoints=1,
		           prop={'size': 20},
		           loc="best")
		plt.xlabel('PCA Dimension %s' % dimx, fontsize=16)
		plt.ylabel('PCA Dimension %s' % dimy, fontsize=16)
		plt.tick_params(axis='x', size=10, labelsize=15)  # labelsize is values, size is for ticks
		# plt.yticks([]) # uncomment to get rid of any y-axis labelling
		# plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
		print('PCA reduction of %s-feature space (best-worst, variance explained by D1+D2 = %s )' % (feats, res))
		#         plt.savefig('/home/masha/02targets_wlv/pics/%s_HTQ_%sfeats.png' % (x.shape[0],x.shape[1]), bbox_inches='tight')
		plt.show()
	if len(set(y)) == 3:
		for i, _ in enumerate(x):
			if y[i] == 'good':
				good = plt.scatter(*x[i], s=POINTSIZE, marker='o', facecolors='none', edgecolors='C2')  ##green
			if y[i] == 'bad':
				bad = plt.scatter(*x[i], s=POINTSIZE, marker='s', facecolors='none', edgecolors='red')
			if y[i] == 'source' or y[i] == 'ref' or y[i] == 'pro':
				third = plt.scatter(*x[i], s=POINTSIZE, marker='x', color='grey')
		
		if 'source' in y:
			plt.legend((good, bad, third),
			           ("good", "bad", "souces"),
			           scatterpoints=1,
			           prop={'size': 20},
			           loc="best")
		if 'ref' in y:
			plt.legend((good, bad, third),
			           ("good", "bad", "reference"),
			           scatterpoints=1,
			           prop={'size': 20},
			           loc="best")
		
		if 'pro' in y:
			plt.legend((good, bad, third),
			           ("good", "bad", "professional"),
			           scatterpoints=1,
			           prop={'size': 20},
			           loc="upper")
		plt.xlabel('PCA Dimension %s' % dimx, fontsize=16)
		plt.ylabel('PCA Dimension %s' % dimy, fontsize=16)
		plt.tick_params(axis='x', size=10, labelsize=15)  # labelsize is values, size is for ticks
		# plt.yticks([]) # uncomment to get rid of any y-axis labelling
		# plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
		print('PCA reduction of %s-feature space (best-worst, variance explained by D1+D2 = %s )' % (feats, res))
		#         plt.savefig('/home/masha/02targets_wlv/pics/%s_HTQ_%sfeats.png' % (x.shape[0],x.shape[1]), bbox_inches='tight')
		plt.show()
	
	if len(set(y)) == 4:
		for i, _ in enumerate(x):
			if y[i] == 'good':
				good = plt.scatter(*x[i], s=POINTSIZE, marker='o', facecolors='none', edgecolors='C2')  ##green
			if y[i] == 'bad':
				bad = plt.scatter(*x[i], s=POINTSIZE, marker='s', facecolors='none', edgecolors='red')
			if y[i] == 'source' or y[i] == 'pro':
				sources = plt.scatter(*x[i], s=POINTSIZE, marker='x', color='orange')
			if y[i] == 'ref':
				ref = plt.scatter(*x[i], s=POINTSIZE, marker='x', color='grey')
		if 'source' in y:
			plt.legend((good, bad, sources, ref),
			           ("good", "bad", "sources", "non-translations"),
			           scatterpoints=1,
			           prop={'size': 20},
			           loc="best")
		
		if 'pro' in y:
			plt.legend((good, bad, sources, ref),
			           ("good", "bad", "professional", "non-translations"),
			           scatterpoints=1,
			           prop={'size': 20},
			           loc="upper")
		
		plt.xlabel('PCA Dimension %s' % dimx, fontsize=16)
		plt.ylabel('PCA Dimension %s' % dimy, fontsize=16)
		plt.tick_params(axis='x', size=10, labelsize=15)  # labelsize is values, size is for ticks
		# plt.yticks([]) # uncomment to get rid of any y-axis labelling
		# plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
		print('PCA reduction of %s-feature space (best-worst, variance explained by D1+D2 = %s )' % (feats, res))
		# plt.title('PCA reduction of %s 23-feature space on EN-DE pair (variance explained by D1+D2 = %s )' % (data, res),
		#           fontsize=20)
		#         plt.savefig('/home/masha/02targets_wlv/pics/%s_HTQ_%sfeats.png' % (x.shape[0],feats), bbox_inches='tight')
		plt.show()


import seaborn as sns


def HTQ_textsdensity(x, y, dimx=1, feats=10):
	sns.set_style("whitegrid")
	sns.set_context('talk')
	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot(121)
	
	if len(set(y)) == 2:
		d = {k: [] for k in set(y)}
		for i in range(len(y)):
			if y[i] in list(set(y)):
				d[y[i]].append(x[i])
		
		sns.distplot(d['good'], color="C2", hist=False, rug=True, ax=ax, label='good')
		sns.distplot(d['bad'], color="red", hist=False, rug=True, ax=ax, label='bad')
		
		ax.set(title="Distance between bad and good translations and their sources (on %s original features)" % feats,
		       xlabel='text scores on PCA Dim%s' % dimx,
		       ylabel="density")
		plt.legend()
	#         plt.savefig('/home/masha/02targets_wlv/pics/%s_HTQ_%sfeats.png' % (x.shape[0],feats), bbox_inches='tight')
	
	if len(set(y)) == 3:
		d = {k: [] for k in set(y)}
		for i in range(len(y)):
			if y[i] in list(set(y)):
				d[y[i]].append(x[i])
		
		sns.distplot(d['source'], color="grey", hist=False, rug=True, ax=ax, label='source',
		             kde_kws={'linestyle': '--'})
		sns.distplot(d['good'], color="C2", hist=False, rug=True, ax=ax, label='good')
		sns.distplot(d['bad'], color="red", hist=False, rug=True, ax=ax, label='bad')
		
		# ax.set(title="Distance between bad and good translations and their sources (on %s original features)" % feats,
		#        xlabel='text scores on PCA Dim%s' % dimx,
		#        ylabel="density")
		plt.legend()
	#         plt.savefig('/home/masha/02targets_wlv/pics/%s_HTQ_%sfeats.png' % (x.shape[0],feats), bbox_inches='tight')
	
	if len(set(y)) == 4:
		d = {k: [] for k in set(y)}
		for i in range(len(y)):
			if y[i] in list(set(y)):
				d[y[i]].append(x[i])
		
		sns.distplot(d['source'], color="orange", hist=False, rug=True, ax=ax, label='source',
		             kde_kws={'linestyle': '--'})
		sns.distplot(d['good'], color="C2", hist=False, rug=True, ax=ax, label='good')
		sns.distplot(d['bad'], color="red", hist=False, rug=True, ax=ax, label='bad')
		sns.distplot(d['ref'], color="grey", hist=False, rug=True, ax=ax, label='reference',
		             kde_kws={'linestyle': '-.'})
		
		ax.set(xlabel='text scores on PCA Dim%s' % dimx,
		       # title="Distance between bad and good translations and their sources (on %s original features)" % feats,
		       ylabel="density")
		plt.grid(False)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		plt.legend()


from numpy import std, mean, sqrt


# correct if the population SD is expected to be equal for the two groups and for nx !=ny
def cohen_d(x, y):
	nx = len(x)
	ny = len(y)
	dof = nx + ny - 2
	return (mean(x) - mean(y)) / sqrt(((nx - 1) * std(x, ddof=1) ** 2 + (ny - 1) * std(y, ddof=1) ** 2) / dof)


def quantify_diffs(df, feat='deverbals', corpus1='ref', corpus2='pro'):
	feat_corpus1 = df.loc[df.astatus == corpus1][feat].tolist()
	feat_corpus2 = df.loc[df.astatus == corpus2][feat].tolist()
	print('two-tails t-test for independent variables with unequal variance for\n', feat.upper())
	print('\t', corpus1.upper(), corpus2.upper())
	print('Means:\t %.4f %.4f' % (np.average(feat_corpus1), np.average(feat_corpus2)))
	print('Variance: %.4f %.4f' % (np.var(feat_corpus1), np.var(feat_corpus2)))
	print('STD:\t %.4f %.4f' % (np.std(feat_corpus1), np.std(feat_corpus2)))
	print('\n%s vs %s' % (corpus1.upper(), corpus2.upper()))
	from scipy.stats import ttest_ind
	t, p = ttest_ind(feat_corpus1, feat_corpus2, equal_var=False)
	df = len(feat_corpus1) + len(feat_corpus2) - 2
	print("t = %g  p = %g df = %g" % (t, p, df))
	print("Effect size:", str(cohen_d(feat_corpus1, feat_corpus2)), '\n')
