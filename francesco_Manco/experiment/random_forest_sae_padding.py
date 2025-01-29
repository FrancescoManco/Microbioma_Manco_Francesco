import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataloader(X_padded, y, batch_size=64):
    dataset = TensorDataset(
        torch.tensor(X_padded.values, dtype=torch.float32),
        torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        classification = self.classifier(latent)
        return latent, reconstruction, classification

def train_autoencoder(model, train_loader, val_loader, optimizer, criterion_reconstruction, criterion_classification, alpha, device, original_input_size, patience=10):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    for epoch in range(100):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            latent, reconstruction, classification = model(X_batch)
            loss_reconstruction = criterion_reconstruction(reconstruction[:, :original_input_size], X_batch[:, :original_input_size])
            loss_classification = criterion_classification(classification, y_batch)
            loss = alpha * loss_reconstruction + (1 - alpha) * loss_classification
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                _, reconstruction, classification = model(X_batch)
                loss_reconstruction = criterion_reconstruction(reconstruction[:, :original_input_size], X_batch[:, :original_input_size])
                loss_classification = criterion_classification(classification, y_batch)
                total_val_loss += (alpha * loss_reconstruction + (1 - alpha) * loss_classification).item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    model.load_state_dict(best_model_state)

def sae_cross_validation(X, y, encoding_shape, alpha, original_input_size, num_folds=5, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    set_seed(random_state)
    
    mean_results = {'macro avg precision': [], 'macro avg recall': [], 'macro avg f1-score': [],
                    'weighted avg precision': [], 'weighted avg recall': [], 'weighted avg f1-score': []}
    best_f1_score = -1
    best_model = None
    best_fold = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        set_seed(random_state + fold)
        print(f"Fold {fold}:")
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        train_loader = create_dataloader(X_train_fold, y_train_fold)
        val_loader = create_dataloader(X_val_fold, y_val_fold)

        model = Autoencoder(input_dim=X.shape[1], bottleneck_dim=encoding_shape).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion_reconstruction = nn.MSELoss()
        criterion_classification = nn.BCELoss()
        
        train_autoencoder(model, train_loader, val_loader, optimizer, criterion_reconstruction, criterion_classification, alpha, device, original_input_size)

        encoder = model.encoder
        encoder.eval()
        with torch.no_grad():
            latent_train = encoder(torch.tensor(X_train_fold.values, dtype=torch.float32).to(device)).cpu().numpy()
            latent_val = encoder(torch.tensor(X_val_fold.values, dtype=torch.float32).to(device)).cpu().numpy()

        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        rf.fit(latent_train, y_train_fold)
        y_pred = rf.predict(latent_val)
        report = classification_report(y_val_fold, y_pred, output_dict=True)

        # Collect metrics
        current_f1_score = report['macro avg']['f1-score']
        for metric in mean_results.keys():
            mean_results[metric].append(report['macro avg'][metric.split()[-1]])

        if current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
            best_model = model
            best_fold = fold

    for metric in mean_results.keys():
        mean_results[metric] = np.mean(mean_results[metric])
    
    print("\nCross-validation results:")
    for metric, value in mean_results.items():
        print(f"{metric}: {value:.4f}")
    print(f"\nBest model was from fold {best_fold} with F1-score: {best_f1_score:.4f}")

    save_path = r"C:\Users\Utente\Desktop\SAE-microbiome\francesco_Manco\models"
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'input_dim': X.shape[1],
        'bottleneck_dim': encoding_shape,
        'fold': best_fold,
        'f1_score': best_f1_score
    }, f"{save_path}/best_autoencoder_clr_0_7_padding.pt")
    
    results_df = pd.DataFrame({
        'Metric': ['Best fold', 'Best F1-score'] + list(mean_results.keys()),
        'Value': [best_fold, best_f1_score] + list(mean_results.values())
    })
    results_df.to_csv(r"C:\Users\Utente\Desktop\SAE-microbiome\results\SAE_clr_0_7_padding.csv", index=False)
    return mean_results

if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\Utente\Desktop\SAE-microbiome\francesco_Manco\data\df_MV-ASD_clr.csv')
    X = df.drop(['target'], axis=1)
    y = df['target']
    original_input_size = X.shape[1]
    target_input_size = 5619

    if original_input_size < target_input_size:
        pad_size = target_input_size - original_input_size
        padding = pd.DataFrame(np.zeros((X.shape[0], pad_size)), columns=[f'pad_{i}' for i in range(pad_size)])
        print("padding applicato")
        X = pd.concat([X, padding], axis=1)
    elif original_input_size > target_input_size:
        X = X.iloc[:, :target_input_size]

    SEED = 42
    set_seed(SEED)
    encoding_shape = 64
    alpha = 1.0
    sae_cross_validation(X, y, encoding_shape, alpha, original_input_size, random_state=SEED)