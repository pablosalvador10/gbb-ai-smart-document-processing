import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Tuple
from utils.ml_logging import get_logger

logger = get_logger()

def evaluate_model(
    y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray, show_visualization: bool = False
) -> Tuple[Dict[str, float], np.ndarray, Dict]:
    """
    Evaluate the performance of a classification model.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        labels (np.ndarray): List of label names.
        show_visualization (bool): Whether to show visualizations or not.

    Returns:
        Tuple[Dict[str, float], np.ndarray, Dict]: Dictionary containing evaluation metrics, confusion matrix, and classification report.
    """
    try:
        logger.info("Evaluating model performance...")

        valid_labels = set(labels)
        original_y_pred = y_pred.copy()
        y_pred = np.array([
            label if label in valid_labels else 'hallucination' for label in y_pred
        ])

        hallucinations = [original_y_pred[i] for i, label in enumerate(y_pred) if label == 'hallucination']
        if hallucinations:
            hallucination_counts = pd.Series(hallucinations).value_counts()
            hallucination_table = pd.DataFrame({
                'Hallucination': hallucination_counts.index,
                'Count': hallucination_counts.values
            })
            logger.info(f"Invalid predictions detected and marked as 'hallucination':\n{hallucination_table}")

        hallucination_indices = [i for i, label in enumerate(y_pred) if label == 'hallucination']

        true_labels_for_hallucinations = y_true[hallucination_indices]
        if true_labels_for_hallucinations.size > 0:
            true_labels_counts = pd.Series(true_labels_for_hallucinations).value_counts()
            true_labels_table = pd.DataFrame({
                'True Label': true_labels_counts.index,
                'Count': true_labels_counts.values
            })
            logger.info(f"True labels corresponding to hallucinations:\n{true_labels_table}")

        valid_indices = [i for i in range(len(y_pred)) if i not in hallucination_indices]
        y_true_filtered = y_true[valid_indices]
        y_pred_filtered = y_pred[valid_indices]

        logger.info(f"Length of y_true_filtered: {len(y_true_filtered)}")
        logger.info(f"Length of y_pred_filtered: {len(y_pred_filtered)}")

        if len(y_true_filtered) != len(y_pred_filtered):
            logger.error("The lengths of y_true_filtered and y_pred_filtered do not match.")
            raise ValueError("The lengths of y_true_filtered and y_pred_filtered do not match.")

        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        precision = precision_score(y_true_filtered, y_pred_filtered, labels=labels, average="weighted", zero_division=0)
        recall = recall_score(y_true_filtered, y_pred_filtered, labels=labels, average="weighted", zero_division=0)
        f1 = f1_score(y_true_filtered, y_pred_filtered, labels=labels, average="weighted", zero_division=0)
        conf_matrix = confusion_matrix(y_true_filtered, y_pred_filtered, labels=labels)
        class_report_dict = classification_report(
            y_true_filtered, y_pred_filtered, labels=labels, target_names=labels, output_dict=True, zero_division=0
        )
        class_report_df = pd.DataFrame(class_report_dict).transpose()

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

        if show_visualization:
            plot_evaluation_metrics(metrics)
            plot_confusion_matrix(conf_matrix, labels)
            plot_classification_report(class_report_df, labels)

        return metrics, conf_matrix, class_report_dict

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def plot_confusion_matrix(conf_matrix: np.ndarray, labels: np.ndarray) -> None:
    """
    Plot the confusion matrix using Plotly.

    Args:
        conf_matrix (np.ndarray): Confusion matrix.
        labels (np.ndarray): List of label names.
    """
    try:
        fig = px.imshow(
            conf_matrix,
            labels=dict(x="Predicted Labels", y="True Labels", color="Count"),
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
        )
        fig.show()
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
        raise

def plot_classification_report(class_report: pd.DataFrame, labels: np.ndarray) -> None:
    """
    Plot the classification report using Plotly.

    Args:
        class_report (pd.DataFrame): Classification report as a DataFrame.
        labels (np.ndarray): List of label names.
    """
    try:
        report_df = class_report.iloc[:-3, :]  # Exclude 'accuracy', 'macro avg', and 'weighted avg'

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=report_df.iloc[:, :-1].values,
                    x=report_df.columns[:-1],
                    y=report_df.index,
                    colorscale="Blues",
                    showscale=True,
                )
            ]
        )
        fig.update_layout(
            title="Classification Report Heatmap",
            xaxis_title="Metrics",
            yaxis_title="Document Types",
        )
        fig.show()
    except Exception as e:
        logger.error(f"Error plotting classification report: {e}")
        raise

def plot_evaluation_metrics(metrics: Dict[str, float]) -> None:
    """
    Plot the evaluation metrics using Plotly.

    Args:
        metrics (Dict[str, float]): Dictionary containing evaluation metrics.
    """
    try:
        fig = px.bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            labels={"x": "Metrics", "y": "Values"},
            text_auto=True,
        )
        fig.update_layout(
            title="Evaluation Metrics",
            yaxis=dict(range=[0, 1]),
            xaxis_title="Metrics",
            yaxis_title="Values",
        )
        fig.show()
    except Exception as e:
        logger.error(f"Error plotting evaluation metrics: {e}")
        raise