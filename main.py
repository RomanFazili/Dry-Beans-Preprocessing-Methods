import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import BorderlineSMOTE as BLSMOTE

from dataset import Dataset
from gridsearch_analyzer import GridSearchAnalyzer
from zscore_capper import ZScoreCapper

SEED = 42

def main(grid_search: GridSearchCV, X: np.ndarray, y: np.ndarray, model_name: str) -> None:

    # Create analyzer and run
    analyzer = GridSearchAnalyzer(grid_search)
    analyzer.run(X, y)
    
    analyzer.print_summary_grid_search()

    # Generate plots
    analyzer.plot_comprehensive_roc_curves(X, y, title=f"ROC Curves for {model_name}")
    analyzer.plot_confusion_matrix(X, y, title=f"Confusion Matrix for {model_name}")
    analyzer.plot_class_distribution_after_sampling(X, y, title="Class Distribution Before And After BLSMOTE Sampling")
    
    analyzer.print_detailed_accuracy_by_class(X, y)
    
    # Get detailed evaluation
    evaluation = analyzer.evaluate_best_model(X, y, SEED, 10)

    print(f"\nFinal Accuracy: {evaluation['aggregate']['accuracy_mean']:.4f} ± {evaluation['aggregate']['accuracy_std']:.4f}")
    print(f"Final F1 Macro: {evaluation['aggregate']['f1_macro_mean']:.4f} ± {evaluation['aggregate']['f1_macro_std']:.4f}")

def grid_search_knn_optimal() -> GridSearchCV:
    
    pipeline = ImbPipeline([
        ('power', PowerTransformer(method='box-cox', standardize=False)),
        ('scaler', StandardScaler()),
        ('zscore_cap', ZScoreCapper(threshold=3.0)),
        ('lda', LinearDiscriminantAnalysis(n_components=None)), # automatically uses #classes - 1 components
        ('blsmote', BLSMOTE(sampling_strategy={'BOMBAY': 1000}, random_state=SEED)),
        ('knn', KNeighborsClassifier(n_neighbors=70)),
    ])

    param_grid = {
        'knn__n_neighbors': [70], # 60 - 80
        'blsmote__sampling_strategy': [{'BOMBAY': 900}], # < 1000
        'lda__solver': ['svd'], # confirmed
        'zscore_cap__threshold': [3.0], # 2 - 4

        # all pipeline steps are significant, tested like this:
        # 'zscore_cap': [ZScoreCapper(threshold=3.0), FunctionTransformer()], 
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    return grid_search

def grid_search_dt_optimal() -> GridSearchCV:
    
    pipeline = ImbPipeline([
        ('power', PowerTransformer(method='box-cox', standardize=False)),
        ('scaler', StandardScaler()),
        ('zscore_cap', ZScoreCapper(threshold=3.0)),
        ('lda', LinearDiscriminantAnalysis(n_components=None)), # automatically uses #classes - 1 components
        ('blsmote', BLSMOTE(sampling_strategy={'BOMBAY': 1000}, random_state=SEED)),
        ('decision_tree', DecisionTreeClassifier(min_samples_leaf=8, random_state=SEED)),
        # ('gaussian_nb', GaussianNB())
    ])

    param_grid = {
        'decision_tree__criterion': ['entropy'], #[gini, entropy, log_loss],
        'decision_tree__splitter': ['random'], #[best, random],
        # 'decision_tree__max_depth': [None, 10, 20, 30, 40, 50],
        'decision_tree__min_samples_split': [2], # < 5
        'decision_tree__min_samples_leaf': [8], # < 10
        'decision_tree__min_weight_fraction_leaf': [0.0], # < 0.5
        'decision_tree__max_features': [None], #[None, 'sqrt', 'log2', 0.5, 0.75, 1.0],
        'decision_tree__max_leaf_nodes': [None], #[None, 10, 20, 30, 40, 50],
        # 'decision_tree__min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'decision_tree__class_weight': [None], #[None, 'balanced'],
        # 'decision_tree__ccp_alpha': [0.0, 0.1, 0.2],
        # 'decision_tree__monotonic_cst': [None, [1, 0, -1]],
        'blsmote__sampling_strategy': [{'BOMBAY': 850}], # < 900
        'lda__solver': ['svd'],
        'zscore_cap__threshold': [3.0], # 2.5 - 3.5
        'power__method': ['box-cox'], # ['box-cox', 'yeo-johnson'],
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    return grid_search

def grid_search_nb_optimal() -> GridSearchCV:

    from sklearn.feature_selection import SelectKBest, f_classif
    
    pipeline = ImbPipeline([
        ('power', PowerTransformer(method='box-cox', standardize=False)),
        ('scaler', StandardScaler()),
        ('zscore_cap', ZScoreCapper(threshold=3.0)),
        ('select_k', SelectKBest(score_func=f_classif, k=29)),
        ('lda', LinearDiscriminantAnalysis(n_components=None)), # automatically uses #classes - 1 components
        ('blsmote', BLSMOTE(sampling_strategy={'BOMBAY': 1000}, random_state=SEED)),
        ('gaussian_nb', GaussianNB())
    ])

    param_grid = {
        'gaussian_nb__var_smoothing': [1e-9],
        'blsmote__sampling_strategy': [{'BOMBAY': 900}],
        'select_k__k': [28],
        'lda__solver': ['svd'],
        'blsmote__k_neighbors': [3],
        'blsmote__m_neighbors': [10],
        'zscore_cap__threshold': [3.0],
    }
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    return grid_search
    
def grid_search_knn_baseline() -> GridSearchCV:
    
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=70)),
    ])

    param_grid = {
        'knn__n_neighbors': [30],
    }

    # Create GridSearchCV
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    return grid_search

def grid_search_dt_baseline() -> GridSearchCV:
    
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('decision_tree', DecisionTreeClassifier(min_samples_leaf=8, random_state=SEED)),
    ])

    param_grid = {
        'decision_tree__criterion': ['entropy'], #[gini, entropy, log_loss],
        'decision_tree__splitter': ['random'], #[best, random],
        # 'decision_tree__max_depth': [None, 10, 20, 30, 40, 50],
        'decision_tree__min_samples_split': [2], # < 5
        'decision_tree__min_samples_leaf': [8], # < 10
        'decision_tree__min_weight_fraction_leaf': [0.0], # < 0.5
        'decision_tree__max_features': [None], #[None, 'sqrt', 'log2', 0.5, 0.75, 1.0],
        'decision_tree__max_leaf_nodes': [None], #[None, 10, 20, 30, 40, 50],
        # 'decision_tree__min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'decision_tree__class_weight': [None], #[None, 'balanced'],
        # 'decision_tree__ccp_alpha': [0.0, 0.1, 0.2],
        # 'decision_tree__monotonic_cst': [None, [1, 0, -1]],
    }

    # Create GridSearchCV
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    return grid_search

def grid_search_nb_baseline() -> GridSearchCV:

    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('gaussian_nb', GaussianNB())
    ])

    param_grid = {
        'gaussian_nb__var_smoothing': [1e-9],
    }
    
    # Create GridSearchCV
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    return grid_search
    
def grid_search_svm_baseline() -> GridSearchCV:
    
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=SEED)),
    ])

    param_grid = {
        'svm__kernel': ['rbf'], #[rbf, linear],
        'svm__C': [10], #[5 - 15],
        'svm__gamma': ['scale'], #['scale', 0.01, 0.1, 1],
        'scaler': [StandardScaler()],
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    return grid_search

def grid_search_rf_baseline() -> GridSearchCV:

    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=SEED))
    ])

    param_grid = {
        'rf__max_depth': [20], # > 10
        'rf__min_samples_split': [2],
        'rf__min_samples_leaf': [1],
        'rf__max_features': [None],
        'rf__bootstrap': [True],
        'rf__class_weight': [None],
        'scaler': [StandardScaler()],
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=4,
        return_train_score=True
    )

    return grid_search

def grid_search_mlp_baseline() -> GridSearchCV:

    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(random_state=SEED))
    ])

    param_grid = {
        # 'mlp__hidden_layer_sizes': [(100,), (100, 50), (200,)],
        # 'mlp__activation': ['relu', 'tanh'],
        # 'mlp__solver': ['adam'],
        # 'mlp__alpha': [1e-5, 1e-4, 1e-3],
        # 'mlp__learning_rate_init': [1e-3, 1e-2],
        'scaler': [StandardScaler()],
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=4,
        return_train_score=True
    )

    return grid_search

if __name__ == "__main__":

    # Get model selection from user
    input_path = input("Enter the input path: ")
    model_name = input("Enter the model name: ").upper()

    # Load dataset
    dataset = Dataset.load_dataset(input_path=input_path)
    dataset.add_derived_features()

    X = dataset.X
    y = dataset.y

    if model_name == 'KNN-OPTIMAL':
        grid_search = grid_search_knn_optimal()
    elif model_name == 'DT-OPTIMAL':
        grid_search = grid_search_dt_optimal()
    elif model_name == 'NB-OPTIMAL':
        grid_search = grid_search_nb_optimal()
    elif model_name == 'SVM-BASELINE':
        grid_search = grid_search_svm_baseline()
    elif model_name == 'RF-BASELINE':
        grid_search = grid_search_rf_baseline()
    elif model_name == 'MLP-BASELINE':
        grid_search = grid_search_mlp_baseline()
    elif model_name == 'KNN-BASELINE':
        grid_search = grid_search_knn_baseline()
    elif model_name == 'DT-BASELINE':
        grid_search = grid_search_dt_baseline()
    elif model_name == 'NB-BASELINE':
        grid_search = grid_search_nb_baseline()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    main(grid_search, X, y, model_name)
