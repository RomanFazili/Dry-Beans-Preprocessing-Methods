from typing import Dict, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score, roc_curve, auc
)
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from zscore_capper import ZScoreCapper


class GridSearchAnalyzer:
    """Analyze and visualize GridSearchCV results."""
    
    def __init__(self, grid_search: GridSearchCV) -> None:
        self.grid_search = grid_search
        
    def run(self, X: np.ndarray, y: np.ndarray) -> None:
        """Run the grid search and store the best estimator, parameters, score, and cv results."""

        self.grid_search.fit(X, y)
        self.best_estimator_ = self.grid_search.best_estimator_
        self.best_params_ = self.grid_search.best_params_
        self.best_score_ = self.grid_search.best_score_
        self.cv_results_ = self.grid_search.cv_results_

    def evaluate_best_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        seed: int, 
        cv_folds: int,
    ) -> Dict[str, Any]:
        """Evaluate best model with detailed cross-validation."""
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        fold_metrics = []
        y_true_all = []
        y_pred_all = []
        y_proba_all = []
        
        labels = sorted(list(set(y.tolist())))
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            self.best_estimator_.fit(X_train, y_train)
            y_pred = self.best_estimator_.predict(X_test)
            y_proba = self.best_estimator_.predict_proba(X_test)
            
            acc = float(accuracy_score(y_test, y_pred))
            # micro is equivalent to accuracy in this case, only need macro and weighted
            prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
            prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
            
            fold_result = {
                'fold': fold_idx,
                'accuracy': acc,
                'precision_macro': float(prec_macro),
                'recall_macro': float(rec_macro),
                'f1_macro': float(f1_macro),
                'precision_weighted': float(prec_w),
                'recall_weighted': float(rec_w),
                'f1_weighted': float(f1_w),
                'confusion_matrix': cm,
            }
            
            fold_metrics.append(fold_result)
            y_true_all.append(y_test)
            y_pred_all.append(y_pred)
            y_proba_all.append(y_proba)
        
        # Aggregate metrics
        y_true_concat = np.concatenate(y_true_all)
        y_pred_concat = np.concatenate(y_pred_all)
        
        acc_mean = float(np.mean([m['accuracy'] for m in fold_metrics]))
        acc_std = float(np.std([m['accuracy'] for m in fold_metrics], ddof=0))
        f1_macro_mean = float(np.mean([m['f1_macro'] for m in fold_metrics]))
        f1_macro_std = float(np.std([m['f1_macro'] for m in fold_metrics], ddof=0))
        f1_weighted_mean = float(np.mean([m['f1_weighted'] for m in fold_metrics]))
        f1_weighted_std = float(np.std([m['f1_weighted'] for m in fold_metrics], ddof=0))
        
        overall_report = classification_report(y_true_concat, y_pred_concat, labels=labels, output_dict=True, zero_division=0)
        overall_cm = confusion_matrix(y_true_concat, y_pred_concat, labels=labels).tolist()
        
        # Calculate ROC AUC if probabilities available
        roc_auc_macro_ovr = None
        roc_auc_macro_ovo = None
        average_precision_macro = None
        per_class_average_precision = None
        
        if len(y_proba_all) == len(y_true_all):
            y_proba_concat = np.vstack(y_proba_all)
            Y_true_bin = label_binarize(y_true_concat, classes=labels)
            roc_auc_macro_ovr = float(roc_auc_score(Y_true_bin, y_proba_concat, multi_class='ovr', average='macro'))
            roc_auc_macro_ovo = float(roc_auc_score(Y_true_bin, y_proba_concat, multi_class='ovo', average='macro'))
            average_precision_macro = float(average_precision_score(Y_true_bin, y_proba_concat, average='macro'))
            per_class_average_precision = {
                str(lbl): float(average_precision_score(Y_true_bin[:, idx], y_proba_concat[:, idx]))
                for idx, lbl in enumerate(labels)
            }
        
        return {
            'cv': {'n_splits': cv_folds, 'labels': [str(label_val) for label_val in labels]},
            'per_fold': fold_metrics,
            'aggregate': {
                'accuracy_mean': acc_mean,
                'accuracy_std': acc_std,
                'f1_macro_mean': f1_macro_mean,
                'f1_macro_std': f1_macro_std,
                'f1_weighted_mean': f1_weighted_mean,
                'f1_weighted_std': f1_weighted_std,
                'confusion_matrix_overall': overall_cm,
                'classification_report_overall': overall_report,
                'roc_auc_macro_ovr': roc_auc_macro_ovr,
                'roc_auc_macro_ovo': roc_auc_macro_ovo,
                'average_precision_macro': average_precision_macro,
                'per_class_average_precision': per_class_average_precision,
            }
        }

    def plot_confusion_matrix(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        title: str = "Confusion Matrix (CV)", 
        cv_folds: int = 10, 
        seed: int = 42,
    ) -> None:
        """Plot confusion matrix for the best model using cross-validation."""
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        y_pred_all = []
        y_true_all = []
        
        labels = sorted(list(set(y.tolist())))
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            self.best_estimator_.fit(X_train, y_train)
            y_pred = self.best_estimator_.predict(X_test)
            
            y_pred_all.extend(y_pred)
            y_true_all.extend(y_test)
        
        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.show()
    
    def print_detailed_accuracy_by_class(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        cv_folds: int = 10, 
        seed: int = 42
    ) -> None:
        """Generate detailed accuracy metrics by class using cross-validation."""

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        
        labels = sorted(list(set(y.tolist())))
        n_classes = len(labels)
        
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            self.best_estimator_.fit(X_train, y_train)
            y_pred = self.best_estimator_.predict(X_test)
            y_proba = self.best_estimator_.predict_proba(X_test)
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(y_proba)
        
        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)
        y_proba = np.array(all_y_proba)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        tp_rates = []
        fp_rates = []
        mcc_scores = []
        roc_areas = []
        prc_areas = []
        
        y_true_bin = label_binarize(y_true, classes=labels)
        
        for i, label in enumerate(labels):
            tp_rate = recall[i]
            tp_rates.append(tp_rate)
            
            tp = cm[i, i]  # True positives for this class
            fp = cm[:, i].sum() - tp  # False positives for this class
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - tp)  # True negatives
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            fp_rates.append(fp_rate)
            
            fn = cm[i, :].sum() - tp  # False negatives
            mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
            mcc_scores.append(mcc)
            
            roc_auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i], average='macro')
            roc_areas.append(roc_auc)
            
            prc_auc = average_precision_score(y_true_bin[:, i], y_proba[:, i])
            prc_areas.append(prc_auc)
        
        # Calculate weighted averages
        weights = support / support.sum()
        weighted_tp_rate = np.average(tp_rates, weights=weights)
        weighted_fp_rate = np.average(fp_rates, weights=weights)
        weighted_precision = np.average(precision, weights=weights)
        weighted_recall = np.average(recall, weights=weights)
        weighted_f1 = np.average(f1, weights=weights)
        weighted_mcc = np.average(mcc_scores, weights=weights)
        weighted_roc = np.average(roc_areas, weights=weights)
        weighted_prc = np.average(prc_areas, weights=weights)
        
        # Create DataFrame
        data = []
        for i, label in enumerate(labels):
            data.append({
                'Class': label,
                'TP Rate': tp_rates[i],
                'FP Rate': fp_rates[i],
                'Precision': precision[i],
                'Recall': recall[i],
                'F-Measure': f1[i],
                'MCC': mcc_scores[i],
                'ROC Area': roc_areas[i],
                'PRC Area': prc_areas[i]
            })
        
        # Add weighted average row
        data.append({
            'Class': 'Weighted Avg.',
            'TP Rate': weighted_tp_rate,
            'FP Rate': weighted_fp_rate,
            'Precision': weighted_precision,
            'Recall': weighted_recall,
            'F-Measure': weighted_f1,
            'MCC': weighted_mcc,
            'ROC Area': weighted_roc,
            'PRC Area': weighted_prc
        })
        
        df = pd.DataFrame(data)
        
        # Print formatted table
        print("\n=== Detailed Accuracy By Class ===")
        print(df.to_string(index=False, float_format='%.3f', justify='right'))

        return df
    
    def plot_class_distribution_after_sampling(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        title: str = "Class Distribution After BLSMOTESampling",
    ) -> None:
        """Plot class distribution after applying the pipeline's BLSMOTE sampling step."""

        if 'blsmote' not in self.best_estimator_.named_steps:
            return None
        
        X_sampled, y_sampled = self.best_estimator_.named_steps['blsmote'].fit_resample(X, y)
        
        unique, counts = np.unique(y_sampled, return_counts=True)
        labels = [str(label) for label in unique]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original distribution
        unique_orig, counts_orig = np.unique(y, return_counts=True)
        labels_orig = [str(label) for label in unique_orig]
        
        ax1.bar(labels_orig, counts_orig, color='lightcoral', alpha=0.7)
        ax1.set_title("Original Class Distribution")
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of Instances')
        ax1.tick_params(axis='x', rotation=45)
        
        for i, count in enumerate(counts_orig):
            ax1.text(i, count + max(counts_orig) * 0.01, str(count), ha='center', va='bottom')
        
        # After sampling distribution
        ax2.bar(labels, counts, color='lightgreen', alpha=0.7)
        ax2.set_title(f"After BLSMOTE Sampling")
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Number of Instances')
        ax2.tick_params(axis='x', rotation=45)
        
        for i, count in enumerate(counts):
            ax2.text(i, count + max(counts) * 0.01, str(count), ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()


    def plot_comprehensive_roc_curves(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        title: str = "Comprehensive ROC Curves (CV)", 
        save_path: Optional[str] = None, 
        cv_folds: int = 10, 
        seed: int = 42
    ) -> Dict[str, Any]:
        """Plot comprehensive ROC curves with Macro, Micro, and Per-class AUC using cross-validation."""
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        y_proba_all = []
        y_true_all = []
        
        labels = sorted(list(set(y.tolist())))
        n_classes = len(labels)
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            self.best_estimator_.fit(X_train, y_train)
            y_proba = self.best_estimator_.predict_proba(X_test)
            
            y_proba_all.append(y_proba)
            y_true_all.extend(y_test)
        
        y_proba_concat = np.vstack(y_proba_all)
        y_true_concat = np.array(y_true_all)
        
        y_bin = label_binarize(y_true_concat, classes=labels)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba_concat[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_proba_concat.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Plot per-class ROC curves, micro-average, macro-average & diagonal line
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,label=f'{labels[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
                 label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
        
        plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', linewidth=4,
                 label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
        
        # Formatting
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add summary statistics
        textstr = f'Summary:\nMicro AUC: {roc_auc["micro"]:.3f}\nMacro AUC: {roc_auc["macro"]:.3f}\nClasses: {n_classes}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.show()
        
        # Print AUC summary
        print("\n" + "="*70)
        print("COMPREHENSIVE ROC AUC SUMMARY (Cross-Validation)")
        print("="*70)

        # Per-class AUCs
        for i, label in enumerate(labels):
            print(f"{str(label):<15} {roc_auc[i]:.4f}")

        print("-"*70)
        print(f"{'Micro-average':<15} {roc_auc['micro']:.4f}")
        print(f"{'Macro-average':<15} {roc_auc['macro']:.4f}")
        print("="*70)

        return {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'labels': labels,
            'n_classes': n_classes
        }

    def print_summary_grid_search(self):
        """Print a summary of the grid search results."""
        print("="*60)
        print(f"Best Score: {self.best_score_:.4f}")
        print(f"Best Parameters: {self.best_params_}")
        print("="*60)
