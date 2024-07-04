import logging
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

import torch
from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

torch.manual_seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class E2EDatasetLoader(Dataset):
    """
    DataLoader compatible with both dense (pandas DataFrame) and sparse (scipy CSR matrix) data formats.
    Automatically handles conversion to PyTorch tensors for model compatibility.

    Parameters:
    features (pd.DataFrame or scipy.sparse.csr_matrix): Input features.
    targets (np.ndarray or None): Target labels, optional for scenarios like unsupervised learning or inference.

    Returns:
    A PyTorch Dataset with features and optionally targets.
    """

    def __init__(self, features, targets=None):
        # Handle pandas DataFrame
        if isinstance(features, pd.DataFrame):
            # TODO: in future we need to excplicitly catch the errors from the apply
            features = features.apply(pd.to_numeric)
            self.features = torch.tensor(features.values, dtype=torch.float32)
        # Handle numpy arrays
        elif isinstance(features, np.ndarray):
            self.features = torch.tensor(features, dtype=torch.float32)
        # Handle scipy sparse CSR matrix
        elif isinstance(features, csr_matrix):
            self.features = torch.tensor(features.toarray(), dtype=torch.float32)
        else:
            raise TypeError("Unsupported feature type. Features should be either a pandas DataFrame, numpy array, "
                            "or scipy sparse CSR matrix.")

        if targets is not None:
            if isinstance(targets, np.ndarray):
                # Convert Numpy targets to tensor, assuming targets are numeric
                self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
            else:
                # Handle csr matrix
                self.targets = targets.tocsr()
        else:
            self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        if isinstance(self.features, torch.Tensor):
            instance = self.features[index]
        else:  # For csr matrix
            instance = torch.from_numpy(self.features[index, :].todense()).float()

        if self.targets is not None:
            if isinstance(self.targets, torch.Tensor):
                target = self.targets[index]
            else:  # For csr matrix
                target = torch.from_numpy(self.targets[index].todense()).float()
        else:
            target = None

        return instance, target


class DRMArchitecture(nn.Module):
    """
        A simple neural network architecture for binary classification tasks.

        Parameters:
        input_size (int): Number of input features.
        dropout (float): Dropout rate to use between layers for regularization.
        hidden_layer_size (int): Number of neurons in each hidden layer.
        output_neurons (int): Number of output neurons, typically 1 for binary classification.

        Returns:
        A PyTorch neural network model.
        """

    def __init__(self, input_size, dropout=0.1, hidden_layer_size=10, output_neurons=1):
        super(DRMArchitecture, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.Dropout(dropout),
            nn.ELU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.Dropout(dropout),
            nn.ELU(),
            nn.Linear(hidden_layer_size, 16),
            nn.Dropout(dropout),
            nn.ELU(),
            nn.Linear(16, output_neurons),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class PropDRM(BaseEstimator, ClassifierMixin):
    """
       A PyTorch-based estimator compatible with scikit-learn GridSearchCV for hyperparameter tuning.

       Parameters:
       batch_size (int): Batch size for training.
       num_epochs (int): Number of epochs to train for.
       learning_rate (float): Learning rate for the optimizer.
       stopping_crit (int): Stopping criterion for early stopping if improvement is less than this threshold.
       hidden_layer_size (int): Size of hidden layers in the neural network.
       dropout (float): Dropout rate for regularization in the network.

       Attributes:
       device (torch.device): Device (CPU/GPU) to run the training on.
       model (torch.nn.Module): The neural network model.
       optimizer (torch.optim.Optimizer): Optimizer for training the model.
       loss_fn (torch.nn.modules.loss): Loss function used for training.
       """

    def __init__(self, batch_size=8, num_epochs=10, learning_rate=0.0001, patience=5, hidden_layer_size=30,
                 dropout=0.2):
        logging.info("Initializing the model.")
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.hidden_layer_size = hidden_layer_size
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None  # Initialize model later
        self.optimizer = None
        self.loss_fn = torch.nn.BCELoss()

        self.classes_ = None

    def fit(self, X, y):
        logging.info("Starting the training process.")
        self.classes_ = np.unique(y)  # Capture all unique classes
        X, y = check_X_y(X, y, accept_sparse=True, dtype='numeric', force_all_finite='allow-nan')
        # Determine the input size from the features
        input_size = X.shape[1]

        self.model = DRMArchitecture(input_size, dropout=self.dropout, hidden_layer_size=self.hidden_layer_size,
                                     output_neurons=1)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Prepare the data loader
        dataset = E2EDatasetLoader(X, y)

        best_loss = float('inf')
        epochs_no_improve = 0

        # Training loop
        for epoch in range(self.num_epochs):
            total_loss = 0
            for features, labels in dataset:
                features, labels = features.to(self.device), labels.to(self.device)
                self.model.train()
                outputs = self.model(features).view(-1, 1)
                labels = labels.view(-1, 1)
                loss = self.loss_fn(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            tolerance = 1e-4
            mean_loss = total_loss / len(dataset)
            if best_loss - mean_loss >= tolerance:
                best_loss = mean_loss
                epochs_no_improve = 0
                logging.info(f"Epoch {epoch + 1}, Loss improved to {mean_loss:.6f}")
            else:
                epochs_no_improve += 1
                logging.info(f"Epoch {epoch + 1}, No improvement in loss for {epochs_no_improve} epochs")

            if epochs_no_improve >= self.patience:
                logging.info("Early stopping triggered due to no improvement")
                break
        return self

    def predict(self, X):
        logging.info("Making predictions.")
        check_is_fitted(self, ['model'])
        X = check_array(X, accept_sparse=True, )
        dataset = E2EDatasetLoader(X)
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for features, _ in dataset:
                features = features.to(self.device)
                outputs = self.model(features)
                predictions.extend(outputs.sigmoid().round().cpu().numpy())
        return np.array(predictions)

    def predict_proba(self, X):
        logging.info("Predicting probabilities.")
        check_is_fitted(self, ['model', 'classes_'])
        dataset = E2EDatasetLoader(X)
        prob_predictions = []
        self.model.eval()
        with torch.no_grad():
            for features, _ in dataset:
                features = features.to(self.device)
                outputs = self.model(features)
                # Calculate probabilities
                probabilities = outputs.sigmoid().cpu().numpy()
                # Convert probabilities to two-column format if binary classification
                if len(self.classes_) == 2:
                    # First column for negative class (1 - probability of positive class)
                    probabilities = np.hstack([1 - probabilities, probabilities])
                prob_predictions.append(probabilities)

        return np.array(prob_predictions)

    def score(self, X, y, **kwargs):
        """
        Scores the model using the Area Under the ROC Curve (AUC-ROC) metric.

        Parameters:
        X (array-like): Feature set.
        y (array-like): True labels.

        Returns:
        float: AUC-ROC score.
        """
        check_is_fitted(self, ['model'])
        y_pred_proba = self.predict_proba(X)
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]
        return roc_auc_score(y, y_pred_proba)
