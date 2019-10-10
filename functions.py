# this file contains the language-independent functions to analyse the big table of text features in 2targets research:
# GOAL of the research: given vectors of sharable LG features for English and two target languages (German, Russian),
# measure distances between the sources and their targets,
# and detect the amount of shining through in each language pair

## this file has universal functions for experiments 1 and 3, functions for experiment 2 are in language specific help-files
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from numpy import std, mean, sqrt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

## create your raw X values and choose/create your Y
## mode = ['lang_status','lang_status_corpus', 'lang']; scaling = [0,1]; df with raw or z-scored freqs
## it seems intuitive that scaling is required for raw freqs; i.e. with z-scores no scaling is needed
from sklearn.feature_selection import SelectKBest

### this function goes inside the get_xy function, but can be run just to get the feel of the data
def featureselection(x, labels, features=None):
	# Feature selection
	ff = SelectKBest(k=features).fit(x, labels)
	newdata = SelectKBest(k=features).fit_transform(x, labels)
	top_ranked_features = sorted(enumerate(ff.scores_), key=lambda y: y[1], reverse=True)[:features]
	# print(top_ranked_features) ## it is a list of tuples (11, 3930.587580730744)
	top_ranked_features_indices = [x[0] for x in top_ranked_features]
	return newdata, top_ranked_features_indices


from sklearn.preprocessing import StandardScaler

def get_xy_topfeats_4contrast(df, mode, features=25, scaling=1):
	if mode == 'lang':  # en ru! but this includes translations, tooooo!
		
		Y0 = df.loc[:, 'alang'].values  # returns an array of text_types for coloring
		print('set of Y values:', set(Y0), file=sys.stderr)
	
	elif mode == 'lang_status':  ## en_source, ru_ref, ru_stu, ru_pro
		df0 = df.copy()
		df0['astatus'] = df0.apply(lambda row: row.alang + '_' + row.astatus, axis=1)
		# replace en_transl in column 'astatus' with en_source
		df0.loc[df0.astatus == 'en_transl', 'astatus'] = 'en_source'
		Y0 = df0.loc[:, 'astatus'].values  # returns an array of text_types for coloring
		print('set of Y values:', set(Y0))
	
	elif mode == 'lang_status_corpus':  ## useful only for distinguisging en from RusLTC and CroCo, or maybe en_stu, from en_pro
		df0 = df.copy()
		df0['astatus'] = df0.apply(lambda row: row.alang + '_' + row.astatus + '_' + row.akorp, axis=1)
		Y0 = df0.loc[:, 'astatus'].values  # returns an array of text_types for coloring
		print('set of Y values:', set(Y0))
	
	## what do you want best features for?
	## we are going to determine KBest for the contrast between non-translations only, right?
	## this requires subsetting by ROW
	df1 = df.copy()
	## this is a shorter df to be used for feature selection
	df1 = df1.loc[((df1.alang == 'ru') & (df1.astatus == 'ref')) | (df1.alang == 'en')]
	svm_y = df1.alang.tolist()
	print('set of Y-values for feature selection and SVM to verify it:', set(svm_y))
	
	training_set = df1.drop(['afile', 'akorp', 'alang', 'astatus'], 1)
	
	
	if features:
		x0, top_feat = featureselection(training_set, svm_y, features=features)
		print('Original input data length:', df.shape)
		print('Data (X) for SVM after feature selection:', x0.shape)
		print('We use %s best features (ranked by importance):\n' % features,
		      [training_set.keys()[i] for i in top_feat])
		metadata_df = df.loc[:, ['afile', 'akorp', 'alang', 'astatus']]
		## use the full-length df to get the selected rows
		no_meta = df.drop(['afile', 'akorp', 'alang', 'astatus'], 1)
		new_df0 = no_meta.iloc[:, top_feat]  ## this is where I lose meta-columns that I want to keep for subsequent use and where I had an error with feats selection for PCA
		new_df = pd.concat([metadata_df, new_df0], axis=1)
	if scaling:
		# throw away columns not to be used
		
		## transform your data such that its distribution will have a mean value 0 and standard deviation of 1.
		## each value in the dataset will have the sample mean value subtracted, and then divided by the standard deviation of the whole dataset.
		sc = StandardScaler()
		svm_x = sc.fit_transform(x0)
		x1 = new_df0.values
		X0 = sc.fit_transform(x1)
	
	else:
		print('Using raw unscaled X data')
		svm_x = x0
		X0 = new_df0.values
	
	return X0, Y0, svm_x, svm_y, new_df


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


## unlike the function above this one finds best features of the input classes, not just some of them
def get_xy_topfeats_straight(df, class_col='astatus', features=25, scaling=1, select_mode='RFE'):
	df0 = df.copy()
	Y0 = df0.loc[:, class_col].values  # returns an array of text_types for coloring
	print('set of Y values:', set(Y0))
	
	training_set = df0.drop(['afile', 'akorp', 'alang', 'astatus'], 1)
	
	if features:
		
		if select_mode == 'ANOVA':
			x0, top_feat = featureselection(training_set, Y0, features)  # subsetting the data yet again deleting columns with less informative features
		if select_mode == 'RFE':
			top_feat = recursive_elimination(training_set, Y0, features)
			x0 = training_set.iloc[:, top_feat].values
		# x0, top_feat = featureselection(training_set, Y0, features=features)
		
		print('Data (X for SVM) after feature selection:', x0.shape)
		print('We use %s best features (ranked by their importance by %s):' % (features, select_mode),
		      [training_set.keys()[i] for i in top_feat], file=sys.stderr)
		metadata_df = df.loc[:, ['afile', 'akorp', 'alang', 'astatus']]
		new_df0 = training_set.iloc[:,top_feat]  ## this is where I lose meta-columns that I want to keep for subsequent use and where I had an error with feats selection for PCA
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


def get_xy(df, mode, scaling):
	if mode == 'lang':  # en ru
		Y0 = df.loc[:, 'alang'].values  # returns an array of text_types for coloring
		print(set(Y0))
	
	elif mode == 'lang_status':  ## en_source, ru_ref, ru_stu, ru_pro
		df0 = df.copy()
		df0['astatus'] = df0.apply(lambda row: row.alang + '_' + row.astatus, axis=1)
		# replace en_transl in column 'astatus' with en_source
		df0.loc[df0.astatus == 'en_transl', 'astatus'] = 'en_source'
		Y0 = df0.loc[:, 'astatus'].values  # returns an array of text_types for coloring
		print(set(Y0))
	
	elif mode == 'lang_status_corpus':  ## useful only for distinguisging en from RusLTC and CroCo, or maybe en_stu, from en_pro
		df0 = df.copy()
		df0['astatus'] = df0.apply(lambda row: row.alang + '_' + row.astatus + '_' + row.akorp, axis=1)
		Y0 = df0.loc[:, 'astatus'].values  # returns an array of text_types for coloring
		print(set(Y0))
	# elif mode == 'ref_pro':
	# 	df0 = df.copy()
	# 	df0['astatus'] = df0.apply(lambda row: row.alang + '_' + row.astatus + '_' + row.akorp, axis=1)
	# 	Y0 = df0.loc[:, 'astatus'].values  # returns an array of text_types for coloring
	# 	print(set(Y0))
	
	if scaling:
		# throw away columns not to be used
		
		## transform your data such that its distribution will have a mean value 0 and standard deviation of 1.
		## each value in the dataset will have the sample mean value subtracted, and then divided by the standard deviation of the whole dataset.
		sc = StandardScaler()
		try:
			X0 = df.drop(['afile', 'akorp', 'alang', 'astatus'], 1)
		except KeyError:
			X0 = df.drop(['afile', 'alang', 'astatus'], 1)
		X0 = sc.fit_transform(X0)
	
	else:
		print('Using raw unscaled X data')
		try:
			X0 = df.drop(['afile', 'akorp', 'alang', 'astatus'], 1).values
		except KeyError:
			X0 = df.drop(['afile', 'alang', 'astatus'], 1).values
	
	return X0, Y0


# modified function for building X,Y in experiment3 (English_ST excluded)
def target_only_get_xy(df, scaling):
	df0 = df.copy()
	Y0 = df0.loc[:, 'astatus'].values  # returns an array of text_types for coloring
	print(set(Y0))
	
	if scaling:
		# throw away columns not to be used
		
		## transform your data such that its distribution will have a mean value 0 and standard deviation of 1.
		## each value in the dataset will have the sample mean value subtracted, and then divided by the standard deviation of the whole dataset.
		sc = StandardScaler()
		X0 = df.drop(['afile', 'akorp', 'alang', 'astatus'], 1)
		X0 = sc.fit_transform(X0)
	
	else:
		print('Using raw data')
		X0 = df.drop(['afile', 'akorp', 'alang', 'astatus'], 1).values
	return X0, Y0


# expects an array, the original matrix for featurenames;
# the 3d numeric argument is the number of PCA components to return
## the final argument is the name of the column by which to sort the loadings, ex. 'Dim1'
def pca_transform(x, df, dims=2, best='Dim1', print_best=0):
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


## produce X and Y to train LDA and to transform all data
## the last parameter is only necessary if eks=None, i.e. we are using all PCA dims
def make_ldaX(mat_less, mat_all, scaling=0, eks=None, PCAdims=None):
	if eks == None:
		## we assume that PCA has more than 1D
		Xtr = mat_less.iloc[:, -PCAdims:]
		Xte = mat_all.iloc[:, -PCAdims:]
		Xtr = Xtr.values
		Xte = Xte.values
	elif len(eks) > 1:
		Xtr = mat_less.iloc[:, -len(eks):]
		Xte = mat_all.iloc[:, -len(eks):]
		Xtr = Xtr.values
		Xte = Xte.values
	else:
		Xtr = mat_less.iloc[:, -1]
		Xte = mat_all.iloc[:, -1]
		Xtr = Xtr.values.reshape(-1, 1)
		Xte = Xte.values.reshape(-1, 1)
	
	if scaling:
		sc = StandardScaler()
		Xtr = sc.fit_transform(Xtr)
		Xte = sc.fit_transform(Xte)
	else:
		#         print('Mystery to solve: what is the effect of scaling of PCA output before LDA')
		pass
	print('Corresponding X shapes:\nTrain: %s\nTest: %s' % (Xtr.shape, Xte.shape))
	
	return Xtr, Xte


## arg1: use the original lang pair limited input (i.e. matrix);
## arg2: supply the best PCA result (for EN-RU 3D make sense)
## arg3: choose type of experiment (which rows to use, Y) 'langs' (LDA trained for En/ru non-translations) or 'nese' (LDA for RU ref/transl diffs)
## arg4: choose column-wise data (X) to be used: eks=None(all PCAcomponents by default), [0] or [0,2]-- to be passed to Xpcaed[:,0][:,None]
## arg5: surprisingly scaling(bool value here) affects the res on 4class and visuals, but not pro/stu SVM res
def LDA_data_construction(input_mat, PCAed, experiment='langs', eks=None, lang=None, on_=None):
	mat = input_mat.copy().reset_index()
	print(lang)
	## add PCA transformed data as new cols to the original df for further subsetting
	if eks != None:
		if len(eks) == 1:
			mat0 = pd.concat([mat, pd.DataFrame(PCAed[:, eks[0]][:, None])], axis=1)
			print('== Using PCA dim%s ==' % eks[0])
		elif len(eks) > 1:
			temp_arr = np.concatenate((PCAed[:, eks[0]][:, None], PCAed[:, eks[1]][:, None]), axis=1)
			mat0 = pd.concat([mat, pd.DataFrame(temp_arr)], axis=1)
			print('== Using PCA dim%s and dim%s ==' % (eks[0], eks[1]))
	else:
		mat0 = pd.concat([mat, pd.DataFrame(PCAed)], axis=1)
		print('== Using all (PCA) dims ==')
	
	## further subsetting by row
	if experiment == 'langs':
		## limit mat to ru + ref and any en
		mat1 = mat0.copy()
		mat1 = mat1.loc[((mat1.alang == lang) & (mat1.astatus == 'ref')) | (mat1.alang == 'en')]
		## we are going to train LDA on non-translations only
		Ytrain = mat1.alang.tolist()
		print('Train LDA for lang contrast on (Y):', set(Ytrain), len(Ytrain))
		## we are going to apply the discriminant to all the data (1671)
		mat0['index'] = mat0.apply(lambda row: row.alang + '_' + row.astatus, axis=1)
		Ytest = mat0['index'].tolist()  ## keep two EN collections separate for now; deal with it later
		print('Apply lang-contrast LDA to (Y):', set(Ytest), len(Ytest))
		## the last parameter is only necessary if eks=None, i.e. we are using all PCA dims
		Xtrain, Xtest = make_ldaX(mat1, mat0, scaling=0, eks=eks, PCAdims=PCAed.shape[1])
	
	elif experiment == 'nese':
		if on_ == 'pro':
			## limit to ref and pro varieties only, dispose of en
			mat1 = mat0.copy()
			mat1 = mat1.loc[(mat1.alang == lang) & ((mat1.astatus == 'ref') | (mat1.astatus == 'pro'))]
			Ytrain = mat1.astatus.tolist()
			print('Train LDA for translationese on (Y):', set(Ytrain), len(Ytrain))
			mat0 = mat0.loc[mat0.alang == lang]
			Ytest = mat0.astatus.tolist()
			print('Apply translationese LDA to (Y):', set(Ytest), len(Ytest))
			## the last parameter is only necessary if eks=None, i.e. we are using all PCA dims
			Xtrain, Xtest = make_ldaX(mat1, mat0, scaling=0, eks=eks, PCAdims=PCAed.shape[1])
		elif on_ == 'stu':
			## limit to ref and pro varieties only, dispose of en
			mat1 = mat0.copy()
			mat1 = mat1.loc[(mat1.alang == lang) & ((mat1.astatus == 'ref') | (mat1.astatus == 'stu'))]
			Ytrain = mat1.astatus.tolist()
			print('Train LDA for translationese on (Y):', set(Ytrain), len(Ytrain))
			mat0 = mat0.loc[mat0.alang == lang]
			Ytest = mat0.astatus.tolist()
			print('Apply translationese LDA to (Y):', set(Ytest), len(Ytest))
			## the last parameter is only necessary if eks=None, i.e. we are using all PCA dims
			Xtrain, Xtest = make_ldaX(mat1, mat0, scaling=0, eks=eks, PCAdims=PCAed.shape[1])
		elif on_ == 'all':
			## limit to ru varieties only, dispose of en
			mat1 = mat0.copy()
			mat1 = mat1.loc[mat1.alang == lang]
			Ytrain = ['transl' if x == 'stu' or x == 'pro' else x for x in mat1.astatus.tolist()]
			print('Train LDA for translationese on (Y):', set(Ytrain), len(Ytrain))
			Ytest = mat1.astatus.tolist()
			print('Apply translationese LDA to (Y):', set(Ytest), len(Ytest))
			## the last parameter is only necessary if eks=None, i.e. we are using all PCA dims
			Xtrain, Xtest = make_ldaX(mat1, mat1, scaling=0, eks=eks, PCAdims=PCAed.shape[1])
	return Xtrain, Xtest, Ytrain, Ytest


## x,y == data to train discriminant on, X,Y == full dataset
def train_LDA(x, X, y, feats=None, weights=0, solver='eigen', shrinkage='auto',
              N=1):  # df=None, scalings=False, solver='eigen'
	### if solver='svd'(Singular value decomposition) use store_covariance=True,
	lda = LDA(solver=solver, shrinkage=shrinkage, n_components=N)  # number of linear discriminants
	lda.fit(x, y)
	lda_tr = lda.transform(
		x)  ## this is required for NT mode in the ru_plotLDA function and for cross-validation with SVM!
	
	lda_all = lda.transform(X)
	lda_explained = lda.explained_variance_ratio_.sum()
	print('why on earth it is greater than 0?', lda_explained)
	# print('var_explained is the sum of a score for each feature; the score is calculated as a ratio of the first component of each eigenvector and eigen_vals sum')
	
	# percent = str(lda_explained[0]).split(".")[1][:2] + '%'
	# print(percent)
	# print(lda) # parameters of the model
	# print(lda.scalings_) ## n_features x n_features
	# # the weight of each attribute on discriminant (en-ru/nt-tt)
	if weights:
		print(set(y))
		# print(lda.coef_)
		loadings = lda.coef_

		# # the weight of each attribute for each class (en-ru/nt-tt)
		# # df0 = df.drop(['afile', 'akorp', 'alang', 'astatus'], 1)
		# # ## Scalings_ returns a matrix : n_features, n_components.
		# # ## The n_components are the eigenvectors of the (Swithin)-1 * Sbetween so the number of the components is equal to the number of features.
		# # ## to project the data to the first LD (https://github.com/scikit-learn/scikit-learn/issues/8956):
		# # ##### Select the first 2 columns of the scalings_ matrix and then do: X_new = np.dot(X, scalings_[:, 0:2]) or X_new = lda.fit_transform(X,y)
		# # X_new = np.dot(X, lda.scalings_[:, 0:1])
		# print(X_new.shape)
		# # print(lda.scalings_) ## returns 23 x 23 matrix
		res0 = pd.DataFrame(loadings, columns=feats, index=['LDA_discr'])
		res = res0.transpose()  # to enable sorting
		# get the feature that contribute most to the distinction between the corpora
		res.sort_values('LDA_discr', ascending=False, inplace=True)
		print(res)
		
		return lda_tr, lda_all, lda  # the last output is the model that can be used to access stats for plotting (lda.scalings_ matrix after getting values for all instances)
	
	else:
		print('Done. One thing I dont understand is var explained here wrt PCA result')
		
		return lda_tr, lda_all, lda  # the last output is the model


### x,y need to contain the classes that were used to LEARN LDA discriminant (XOlang, YOlang, or X_TT_NT, Y_TT_NT)
def validateLDA_LDAclassifier(x, y):
	lda = LDA(n_components=1)  # number of linear discriminants
	lda.fit(x, y)
	## if no PCA transformation
	# lda.fit(X_4LDA, Y_4LDA)
	
	skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
	
	preds = cross_val_predict(lda, x, y, cv=skf)
	print('Cross-validated estimates for data points')
	print(classification_report(y, preds))
	scoring = ['accuracy', 'f1_macro']  ## , 'precision_macro', 'recall_macro'
	cv_scores = cross_validate(lda, x, y, cv=skf, scoring=scoring, n_jobs=2)
	acc1 = cv_scores['test_accuracy'].mean()
	print(acc1)
	
	from sklearn.metrics import confusion_matrix
	## Compute confusion matrix
	cm = confusion_matrix(y, preds)
	print(cm)
	
	print('High accuracy of cross-classification with LDA confirms validity of the LDA discriminant')


# validate with a classifyer: it gets the LDAtransformed vectors (of size 1) as input and attempts the classification on it
# the classes (Y) are langs
# evert resorts to 10 fold cv
# dont forget to balance classes or else the classifier get lazy and ignores  de as the minority class
def cv_SVM(x, y, class_weight):
	## defaults radial basis function (RBF): used when there is no prior knowledge about the data
	clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', random_state=0, class_weight=class_weight)
	skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
	scores = cross_val_score(clf, X=x, y=y, cv=skf, scoring='f1_macro')  # cv = loo
	# print("F1macro on each fold:", scores)
	print("F1 over 10folds: ", scores.mean())
	
	scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
	cv_scores = cross_validate(clf, x, y, cv=skf, scoring=scoring, n_jobs=2)
	print('Accuracy over 10folds', cv_scores['test_accuracy'].mean())
	
	preds = cross_val_predict(clf, x, y, cv=skf)
	# print(classification_report(y, preds))
	
	report = classification_report(y, preds, output_dict=True)
	df = pd.DataFrame(report).transpose()
	idx = np.argmin(df['support'].tolist())
	val = np.amin(df['support'].tolist())
	df = df.reset_index()
	minority = df.iloc[idx, 0]
	print(confusion_matrix(y, preds))
	
	# get measures on the minority class
	my_dict = classification_report(y, preds, output_dict=True)
	minorf1 = my_dict[minority]['f1-score']
	print('Average F1 on the minority class (%s/%s): %s' % (minority, val, np.average(minorf1).round(3)))


# correct if the population SD is expected to be equal for the two groups and for nx !=ny
def cohen_d(x, y):
	nx = len(x)
	ny = len(y)
	dof = nx + ny - 2
	return (mean(x) - mean(y)) / sqrt(((nx - 1) * std(x, ddof=1) ** 2 + (ny - 1) * std(y, ddof=1) ** 2) / dof)


from scipy.stats import ttest_ind


def quantify_LDA(var1, var2, name1=None, name2=None):
	print('Means:', np.average(var1), np.average(var2))
	print('Variance:', np.var(var1), np.var(var2))
	print('STD:', np.std(var1), np.std(var2))
	t, p = ttest_ind(var1, var2, equal_var=False)
	df = len(var1) + len(var2) - 2
	print("%s VS %s:  t = %g  p = %g df = %g" % (name1, name2, t, p, df))
	print("Effect size:", str(cohen_d(var1, var2)), '\n')


def validate_LDA(x, y, Xlda_train, Xlda_all, Yall, exp=None, lang=None,
                 on_=None):  # exp='SL-independent translationese'
	print('=====step5: cross-classification with LDA')
	validateLDA_LDAclassifier(x,
	                          y)  ## unlike SVM it consumes the LDA input: it is a cross-classification scenario, rather than cross-validation with SVM
	print()
	print('==SVM CV==')
	print('== step5a (SVM CV on train classes): how well LDA captures the distinctions it is trained for?')
	cv_SVM(Xlda_train, y, 'balanced')
	data = Xlda_train[:, 0][:, None]
	df = pd.DataFrame(data, columns=['LDAscores'])
	df['astatus'] = y
	if exp == 'lang contrast':
		en = df.loc[df.astatus == 'en'].LDAscores.tolist()
		tl = df.loc[df.astatus == lang].LDAscores.tolist()
		quantify_LDA(en, tl, name1='en', name2=lang)
	elif exp == 'SL-independent translationese':
		if on_ == 'pro':
			ref = df.loc[df.astatus == 'ref'].LDAscores.tolist()
			pro = df.loc[df.astatus == 'pro'].LDAscores.tolist()
			quantify_LDA(ref, pro, name1='ref', name2='pro')
		elif on_ == 'stu':
			ref = df.loc[df.astatus == 'ref'].LDAscores.tolist()
			stu = df.loc[df.astatus == 'stu'].LDAscores.tolist()
			quantify_LDA(ref, stu, name1='ref', name2='stu')
		elif on_ == 'all':
			ref = df.loc[df.astatus == 'ref'].LDAscores.tolist()
			transl = df.loc[df.astatus == 'transl'].LDAscores.tolist()
			quantify_LDA(ref, transl, name1='ref', name2='transl')
	print()
	
	print('== step5b (ref/transl): does LDA trained for %s help to distinquish varieties of TL?' % exp)
	
	## creating data for ttest and cohens d for step 3default
	if exp == 'lang contrast':
		data = Xlda_all[:, 0][:, None]
		df = pd.DataFrame(data, columns=['LDAscores'])
		Yall = [x[3:] if 'ru_' in x or 'de_' in x else x for x in Yall]
		df['astatus'] = Yall
		vars_df = df.loc[(df.astatus == 'stu') | (df.astatus == 'pro') | (df.astatus == 'ref')]
		Y_vars = vars_df.astatus.tolist()
		Yall = ['transl' if x == 'pro' or x == 'stu' else x for x in Y_vars]
		X_vars = np.array(vars_df.LDAscores.tolist()).reshape(-1, 1)
		#         print(len(Y_vars))
		#         print(X_vars.shape)
		cv_SVM(X_vars, Yall, 'balanced')
		
		print()
		ref = vars_df.loc[vars_df.astatus == 'ref'].LDAscores.tolist()
		transl = vars_df.loc[(vars_df.astatus == 'pro') | (vars_df.astatus == 'stu')].LDAscores.tolist()
		quantify_LDA(ref, transl, name1='ref', name2='transl')
	
	else:
		data = Xlda_all[:, 0][:, None]
		df = pd.DataFrame(data, columns=['LDAscores'])
		df['astatus'] = Yall
		vars_df = df
		print('In this setting the res for this test coinsides with step5a above\n')
	
	print('== step5c (ref/pro): are the distinctions captured by %s LDA relevant for translation proficiency?' % exp)
	df_5c = vars_df.loc[(vars_df.astatus == 'ref') | (vars_df.astatus == 'pro')]
	Xc = np.array(df_5c.LDAscores.tolist()).reshape(-1, 1)
	Yc = df_5c.astatus.tolist()
	cv_SVM(Xc, Yc, 'balanced')
	ref = vars_df.loc[vars_df.astatus == 'ref'].LDAscores.tolist()
	pro = vars_df.loc[vars_df.astatus == 'pro'].LDAscores.tolist()
	quantify_LDA(ref, pro, name1='ref', name2='pro')
	
	print('== step5d (ref/stu), LDA trained for %s ' % exp)
	df_5d = vars_df.loc[(vars_df.astatus == 'ref') | (vars_df.astatus == 'stu')]
	Xd = np.array(df_5d.LDAscores.tolist()).reshape(-1, 1)
	Yd = df_5d.astatus.tolist()
	cv_SVM(Xd, Yd, 'balanced')
	ref = vars_df.loc[vars_df.astatus == 'ref'].LDAscores.tolist()
	stu = vars_df.loc[vars_df.astatus == 'stu'].LDAscores.tolist()
	quantify_LDA(ref, stu, name1='ref', name2='stu')
	
	print('== step5e (pro/stu)')
	df_5e = vars_df.loc[(vars_df.astatus == 'pro') | (vars_df.astatus == 'stu')]
	Xe = np.array(df_5e.LDAscores.tolist()).reshape(-1, 1)
	Ye = df_5e.astatus.tolist()
	cv_SVM(Xe, Ye, 'balanced')
	pro = vars_df.loc[vars_df.astatus == 'pro'].LDAscores.tolist()
	stu = vars_df.loc[vars_df.astatus == 'stu'].LDAscores.tolist()
	quantify_LDA(pro, stu, name1='pro', name2='stu')
