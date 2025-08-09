from ._utils import focal_loss
from ._model import ScouterModel
from ._datasets import BalancedDataset
from .ScouterData import ScouterData
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class Scouter():
    """
    Scouter model class
        
    Attributes
    ----------
    train_adata: anndata.AnnData
        AnnData object for the train split
    val_adata: anndata.AnnData
        AnnData object for the validation split        
    test_adata: anndata.AnnData
        AnnData object for the test split
    embd_tensor: torch.tensor
        torch.tensor object of the gene embedding matrix
    key_label: str
        The column name of `adata.obs` that corresponds to perturbation conditions
    key_var_genename: str
        The column name of `adata.var` that corresponds to gene names.
    key_embd_index: str
        The column name of `adata.obs` that corresponds to gene index in embedding matrix.
    n_genes: int
        Number of genes in the cell expression 
    network: ScouterModel
        The model achieves minimal validation loss after training
    loss_history: dict
        Dictionary containing the loss history on both train split and validation split
    """
    
    def __init__(
        self,
        pertdata: ScouterData,
        device: str='auto'
    ):
        """
        Parameters
        ----------
        - pertdata:
            An ScouterData Object containing cell expression anndata and gene embedding matrix
        - device:
            Device to run the model on. Default: 'auto'
        """

        if not isinstance(pertdata, ScouterData):
            raise TypeError("`pertdata` must be an ScouterData object")
        
        self.embd_idx_dict = {gene:i for i, gene in enumerate(pertdata.embd.index)}
        self.embd_tensor = torch.tensor(pertdata.embd.values, dtype=torch.float32)
        
        self.key_label = pertdata.key_label
        self.key_embd_index = pertdata.key_embd_index
        self.key_var_genename = pertdata.key_var_genename
        self.n_genes = pertdata.train_adata.shape[1]
        
        self.all_adata = pertdata.adata
        self.train_adata = pertdata.train_adata
        self.val_adata = pertdata.val_adata
        self.test_adata = pertdata.test_adata
        self.ctrl_adata = self.train_adata[self.train_adata.obs[self.key_label] == 'ctrl']
        # Determine the device
        if device == 'auto':
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                self.device = torch.device("cuda:" + str(current_device))
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)


    def model_init(self,
                   n_encoder=(2048, 512),
                   n_out_encoder=64,
                   n_decoder=(2048,),
                   use_batch_norm=True,
                   use_layer_norm=False,
                   dropout_rate=0.):
        """
        Initialize the ScouterModel.

        Parameters:
        ----------
        - n_encoder: 
            Tuple specifying the hidden layer sizes for the cell encoder.
        - n_out_encoder: 
            Size of the output layer for the cell encoder.
        - n_decoder: 
            Tuple specifying the hidden layer sizes for the generator.
        - use_batch_norm: 
            Whether to use batch normalization.
        - use_layer_norm: 
            Whether to use layer normalization.
        - dropout_rate: 
            Dropout rate.
        """
        self.network = ScouterModel(self.n_genes, 
                                    self.embd_tensor,
                                    n_encoder=n_encoder, 
                                    n_out_encoder=n_out_encoder, 
                                    n_decoder=n_decoder,
                                    use_batch_norm=use_batch_norm, 
                                    use_layer_norm=use_layer_norm,
                                    dropout_rate=dropout_rate).to(self.device)
        self.best_val_loss = np.inf
        self.loss_history = {'train_loss': [], 'val_loss': []}


    def train(self,
              batch_size=256,
              class_weights=None,
              lr=0.001,
              sched_gamma=0.9,
              n_epochs=40,
              patience=5,
              alpha=None,
              gamma=2.0
             ):
        """
        Trains the model with the given parameters.
    
        Parameters:
        ----------            
        - nonzero_idx_key: 
            The key name of 'adata.uns' that contains the index of non-zero genes in each perturbation group (needed for loss calculation).
        - batch_size: 
            The number of samples per batch. Defaults to 256.
        - loss_gamma:
            The γ parameter in the loss function. Defaults to 0.
        - loss_lambda: 
            The λ parameter in the loss function. Defaults to 0.5.
        - lr: 
            The learning rate for the optimizer. Defaults to 0.001.
        - sched_gamma: 
            The multiplicative factor of learning rate decay in ExponentialLR. Defaults to 0.9.
        - n_epochs: 
            The maximum number of epochs for training. Defaults to 40.
        - patience: 
            Number of epochs with no improvement after which training will be stopped. Defaults to 5.
        """
        train_dataset = BalancedDataset(self.train_adata, key_label=self.key_label, key_embd_index=self.key_embd_index)
        val_dataset = BalancedDataset(self.val_adata, key_label=self.key_label, key_embd_index=self.key_embd_index)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False) if len(val_dataset) > 0 else None

        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        
        optimizer = optim.Adam(self.network.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=sched_gamma)
        
        epochs_no_improve = 0
        best_model_state_dict = None
        
        for epoch in range(n_epochs):
            # Training phase
            self.network.train()
            train_loss = 0.0
            train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Training Batches", leave=True, unit="batch")
            for ctrl_expr, _, pert_idx, bcode in train_progress:
                ctrl_expr, pert_idx = ctrl_expr.to(self.device), pert_idx.to(self.device)
                group = self.train_adata[bcode,:].obs[self.key_label].values[0]
                true_labels = torch.tensor(self.all_adata.varm['labels'][group].values, dtype=torch.float32).to(self.device)
                true_labels_batch = true_labels.unsqueeze(0).repeat(ctrl_expr.size(0), 1)
                
                optimizer.zero_grad()
                pred_expr = self.network(pert_idx, ctrl_expr)
                loss = focal_loss(pred_expr, true_labels_batch, alpha=alpha, gamma=gamma)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
    
            if val_loader:
                # Validation phase
                self.network.eval()
                val_loss = 0.0
                val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Validation Batches", leave=True, unit="batch")
                with torch.no_grad():
                    for ctrl_expr, _, pert_idx, bcode in val_progress:
                        ctrl_expr, pert_idx = ctrl_expr.to(self.device), pert_idx.to(self.device)
                        group = self.val_adata[bcode,:].obs[self.key_label].values[0]
                        true_labels = torch.tensor(self.all_adata.varm['labels'][group].values, dtype=torch.float32).to(self.device)
                        true_labels_batch = true_labels.unsqueeze(0).repeat(ctrl_expr.size(0), 1)
                        
                        pred_logits = self.network(pert_idx, ctrl_expr)
                        loss = focal_loss(pred_logits, true_labels_batch, alpha=alpha, gamma=gamma)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
            else:
                val_loss = None
    
            # Store the loss in the history and print
            self.loss_history['train_loss'].append(train_loss)
            if val_loss is not None:
                self.loss_history['val_loss'].append(val_loss)
                print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {train_loss:.4f}')
            
            # Step the learning rate scheduler
            scheduler.step()
    
            # Early stopping logic
            if val_loss is not None:
                improvement = self.best_val_loss - val_loss
                if improvement > 0.0001:  # Check if the improvement is greater than 0.001
                    self.best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model_state_dict = self.network.state_dict()
                else:
                    epochs_no_improve += 1
    
                if epochs_no_improve == patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break

        # Load the best model
        if best_model_state_dict is None:
            print("No improvement observed, keeping the original model")
        else:
            self.network.load_state_dict(best_model_state_dict)


    def pred(self, pert_list, n_pred=300, seed=24):
        """
        Transcriptional prediction given the list of perturbations.
    
        Parameters:
        ----------
        - pert_list (list of str):
            A list of perturbations for prediction
        - n_pred (int, optional): 
            Number of control sells to sample, and hence the number of predictions to make for each perturbation. Defaults to 300.
        - seed (int, optional): 
            Random seed for reproducibility. Defaults to 24.
        
        Returns:
        ----------
        - dict:
            A dictionary where keys are the perturbations and values are arrays predicted transcriptional responses. 
            Each array contains `n_pred` predicted transcriptional responses for the corresponding perturbation.
        """       
        
        np.random.seed(seed)
        # Examine if there is any input gene not in the embedding matrix
        unique_inputs = np.unique(sum([p.split('+') for p in pert_list], []))
        unique_not_embd = [p not in self.embd_idx_dict for p in unique_inputs]
        not_found = unique_inputs[unique_not_embd]
        if len(not_found) > 0:
            raise ValueError(f'{len(not_found)} gene(s) are not found in the gene embedding matrix: {not_found}')

        pert_return = {}
        for pert in pert_list:
            pert_idx_list = [[self.embd_idx_dict[g] for g in pert.split('+')]] * n_pred
            pert_idx = torch.tensor(pert_idx_list, dtype=torch.long).to(self.device)
            ctrl_idx = np.random.choice(range(len(self.ctrl_adata)), size=n_pred, replace=True)
            ctrl_expr = torch.tensor(self.ctrl_adata[ctrl_idx].X.toarray(), dtype=torch.float32).to(self.device)

            self.network.eval()
            with torch.no_grad():
                pred_logits = self.network(pert_idx, ctrl_expr)
                pred_probs = torch.softmax(pred_logits.view(-1, self.n_genes, 3), dim=2)
                pert_return[pert] = pred_probs.mean(dim=0).cpu().numpy()

        return pert_return

        
    
    def barplot(self, condition):
        """
        Generates a bar plot showing the predicted class probabilities for a given perturbation condition.

        Parameters:
        ----------
        - condition (str):
            The perturbation condition to generate the bar plot for.
        """
        pred_probs = self.pred([condition])[condition]
        df = pd.DataFrame(pred_probs, columns=['Down', 'Unchanged', 'Up'])
        df['Gene'] = self.all_adata.var_names
        df_melted = df.melt(id_vars='Gene', var_name='Class', value_name='Probability')

        plt.figure(figsize=(15, 8))
        sns.barplot(x='Gene', y='Probability', hue='Class', data=df_melted)
        plt.xticks(rotation=90)
        plt.title(f'Predicted Class Probabilities for {condition}')
        plt.show()

    def evaluate(self, pert_list=None):
        """
        Evaluates the model's performance on a list of perturbations.

        Parameters:
        ----------
        - pert_list (list of str, optional):
            A list of perturbations to evaluate. If None, uses the test set. Defaults to None.

        Returns:
        - pandas.DataFrame:
            A DataFrame containing the evaluation metrics (accuracy, precision, recall, F1-score)
            for each perturbation condition.
        """
        if pert_list is None:
            pert_list = [p for p in self.test_adata.obs[self.key_label].unique() if p != 'ctrl']

        metrics = []
        for pert in pert_list:
            pred_probs = self.pred([pert])[pert]
            pred_labels = np.argmax(pred_probs, axis=1)
            true_labels = self.all_adata.varm['labels'][pert].values + 1
            accuracy = accuracy_score(true_labels, pred_labels)
            conf_matrix = confusion_matrix(true_labels, pred_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro', zero_division=0)
            metrics.append({
                'Perturbation': pert,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-score': f1,
                'Confusion matrix': conf_matrix
            })
        return pd.DataFrame(metrics)
        

"""
    def analyze_20DEGs(self, DEG_sets, pert_list=None):
        if pert_list is None:
            pert_list = [p for p in self.test_adata.obs[self.key_label].unique() if p != 'ctrl']
    
        metrics = []
        
        pred_dict = self.pred(pert_list)
    
        for pert in pert_list:
            # Get the full set of predictions and true labels for the perturbation
            pred_probs = pred_dict[pert]
            true_labels = self.all_adata.varm['labels'][pert].values
    
            for direction, pert_dict in DEG_sets.items():
                if pert in pert_dict:
                    gene_indices = pert_dict[pert]
    
                    if len(gene_indices) > 0:
                        # Filter the predictions and labels to the current gene set
                        pred_probs_subset = pred_probs[gene_indices]
                        true_labels_subset = true_labels[gene_indices]
                        
                        # Get predicted labels (0, 1, 2) from probabilities for the subset
                        pred_labels_subset = np.argmax(pred_probs_subset, axis=1)
    
                        # Shift true labels from (-1, 0, 1) to (0, 1, 2) to match predictions
                        true_labels_subset_shifted = true_labels_subset + 1
                        
                        # Calculate the confusion matrix using a consistent 3x3 format
                        conf_matrix = confusion_matrix(true_labels_subset_shifted, pred_labels_subset, labels=[0, 1, 2])
                        
                        metrics.append({
                            'Perturbation': pert,
                            'Direction': direction,
                            'confusion_matrix': conf_matrix
                        })
    
        return pd.DataFrame(metrics)
"""