import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

#from francesco_Manco.autoencoder import Autoencoder

class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, bottleneck_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_dim)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        bottleneck = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        classification = self.classifier(bottleneck)
        return reconstruction, classification
    

def load_dataset(dataset_path, batch_size=64):
    """Carica il dataset, esegue StratifiedKFold e restituisce i DataLoader per ogni fold."""
    df = pd.read_csv(dataset_path)
    X = df.drop(['Unnamed: 0', 'IndexS', 'IndexN', 'target'], axis=1)
    y = df['target']

    # Convert to numpy arrays
    X_np = X.values
    y_np = y.values

    # StratifiedKFold split
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_loaders = []

    # Iterate over folds
    for train_idx, val_idx in kf.split(X_np, y_np):
        # Split data into train and validation
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        # Create TensorDatasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Append to fold_loaders list
        fold_loaders.append((train_loader, val_loader))

    return fold_loaders

def train_autoencoder(model, train_loader, val_loader, num_epochs=100, device="cuda", use_early_stopping=True, early_stopping_patience=15):
    """Addestra l'autoencoder con PyTorch."""
    alpha = 0.5  # Peso per la ricostruzione
    learning_rate = 0.001

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)

    criterion_reconstruction = nn.MSELoss()
    criterion_classification = nn.BCELoss()

    train_losses, val_losses = [], []
    train_recon_losses, train_class_losses = [], []
    val_recon_losses, val_class_losses = [], []
    accuracies = []

    best_test_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_recon_loss, train_class_loss = 0.0, 0.0, 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            reconstruction, classification = model(inputs)

            loss_reconstruction = criterion_reconstruction(reconstruction, inputs)
            loss_classification = criterion_classification(classification.squeeze().unsqueeze(1), labels.float())

            loss = alpha * loss_reconstruction + (1 - alpha) * loss_classification
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon_loss += loss_reconstruction.item()
            train_class_loss += loss_classification.item()

        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss, val_recon_loss, val_class_loss, correct_predictions, total_samples = 0.0, 0.0, 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                reconstruction, classification = model(inputs)

                loss_reconstruction = criterion_reconstruction(reconstruction, inputs)
                loss_classification = criterion_classification(classification.squeeze().unsqueeze(1), labels.float())
                loss = alpha * loss_reconstruction + (1 - alpha) * loss_classification

                val_loss += loss.item()
                val_recon_loss += loss_reconstruction.item()
                val_class_loss += loss_classification.item()

                predicted = (classification >= 0.5).float()
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = (correct_predictions / total_samples) * 100
        accuracies.append(accuracy)

        scheduler.step(val_loss)

        if val_loss < best_test_loss:
            best_test_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience and use_early_stopping:
            model.load_state_dict(best_model_state)
            break

    return model, best_model_state


class Experimentor:
    def __init__(self ,dataset_path, name):

        self.fold_loaders = load_dataset(dataset_path)
        self.name = name
        self.result_path = os.path.join('./results', name)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    def classify(self):
        #carica il dataset
        df= pd.read_csv(r'C:\Users\Utente\Desktop\SAE-microbiome\francesco_Manco\data\df_MV-ASD_clr.csv')
        self.X = df.drop(['Unnamed: 0', 'IndexS', 'IndexN', 'target'], axis=1)
        self.y = df['target']

        #implementa la CV e decide le metriche da calcolare
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        mean_results = {'macro avg precision': [], 'macro avg recall': [], 'macro avg f1-score': [],
                        'weighted avg precision': [], 'weighted avg recall': [], 'weighted avg f1-score': []}

        #cicla su ogni fold creata dalla CV
        for fold, (train_index, test_index) in enumerate(skf.split(self.X, self.y), 1):
            print(f"Fold {fold}:")
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[test_index]

            #crea e addestra il random forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)

            #calcola le metriche
            report = classification_report(y_val, y_pred, output_dict=True)
            for metric in mean_results.keys():
                mean_results[metric].append(report['macro avg'][metric.split()[-1]])

        for metric in mean_results.keys():
            mean_results[metric] = np.mean(mean_results[metric])

        #converte le metriche in un dataframe e le salva in un file csv
        results_df = pd.DataFrame.from_dict(mean_results, orient='index', columns=['Mean Value (5 fold)']).T
        results_df.to_csv(os.path.join(self.result_path, 'baseline_results.csv'), index=False)
        print(results_df)

    def classify_with_pca(self, n_components):

        df= pd.read_csv(r'C:\Users\Utente\Desktop\SAE-microbiome\francesco_Manco\data\df_MV-ASD_clr.csv')
        self.X = df.drop(['Unnamed: 0', 'IndexS', 'IndexN', 'target'], axis=1)
        self.y = df['target']
        #crea la CV e inizializza un dizionario per le metriche
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        mean_results = {'macro avg precision': [], 'macro avg recall': [], 'macro avg f1-score': [],
                        'weighted avg precision': [], 'weighted avg recall': [], 'weighted avg f1-score': []}

        #cicla su ogni fold della CV
        for fold, (train_index, test_index) in enumerate(skf.split(self.X, self.y), 1):
            print(f"Fold {fold}:")
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[test_index]

            #applica la PCA e addestra il random forest
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_val_pca = pca.transform(X_val)

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_pca, y_train)
            y_pred = rf.predict(X_val_pca)

            #calcola e salva le metriche
            report = classification_report(y_val, y_pred, output_dict=True)
            for metric in mean_results.keys():
                mean_results[metric].append(report['macro avg'][metric.split()[-1]])

        for metric in mean_results.keys():
            mean_results[metric] = np.mean(mean_results[metric])

        results_df = pd.DataFrame.from_dict(mean_results, orient='index', columns=['Mean Value (5 fold)']).T
        results_df.to_csv(os.path.join(self.result_path, 'pca_rf_results.csv'), index=False)
        print(results_df)


    def sae_experiment(self, input_dim, bottleneck_dim, num_epochs=100, device="cuda"):
        results = []

        for fold_idx, (train_loader, val_loader) in enumerate(self.fold_loaders):
            print(f"\nFold {fold_idx + 1}/{len(self.fold_loaders)}")

            model = Autoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim).to(device)
            model, best_state = train_autoencoder(model, train_loader, val_loader, num_epochs=num_epochs, device=device)

            encoder = model.encoder

            X_train_encoded, y_train = [], []
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                X_train_encoded.append(encoder(inputs).cpu().detach().numpy())
                y_train.append(labels.cpu().detach().numpy())

            X_val_encoded, y_val = [], []
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                X_val_encoded.append(encoder(inputs).cpu().detach().numpy())
                y_val.append(labels.cpu().detach().numpy())

            X_train_encoded = np.vstack(X_train_encoded)
            y_train = np.vstack(y_train).ravel()
            X_val_encoded = np.vstack(X_val_encoded)
            y_val = np.vstack(y_val).ravel()

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_encoded, y_train)

            y_pred = rf.predict(X_val_encoded)
            report = classification_report(y_val, y_pred, output_dict=True)
            results.append(report)

            torch.save(best_state, os.path.join(self.result_path, f"fold_{fold_idx + 1}_best_model.pth"))
            print(f"Best model for fold {fold_idx + 1} saved.")

        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.result_path, "sae_rf_results.csv"), index=False)
        print(f"Results saved to {os.path.join(self.result_path, 'sae_rf_results.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='C:\\Users\\Utente\\Desktop\\SAE-microbiome\\francesco_Manco\\data\\df_MV-ASD_clr.csv', required=True, help="Percorso al dataset")
    #parser.add_argument("--name", type=str, required=True, help="Nome dell'esperimento")
    parser.add_argument("--input_dim", type=int, default=1315 ,required=True, help="Dimensione input")
    parser.add_argument("--encoding_dim", type=int, default=16,required=True, help="Dimensione rappresentazione latente")
    parser.add_argument("--num_epochs", type=int, default=100, help="Numero di epoche")
    parser.add_argument("--device", type=str, default="cuda:0", help="Dispositivo per l'esecuzione")
    parser.add_argument("--expname", type=str, required=True, help="Experiment name",
                        choices=["Baseline", "PCA_RF", "SAE"] ,default='Baseline')
    args = parser.parse_args()

    experimentor = Experimentor(args.dataset, args.expname)

    if args.expname == "Baseline":
        experimentor.classify()
    elif args.expname == "PCA_RF":
        experimentor.classify_with_pca(n_components=args.encoding_dim)
    elif args.expname == "SAE":
        experimentor.sae_experiment(args.input_dim, args.encoding_dim, num_epochs=args.num_epochs, device=args.device)
