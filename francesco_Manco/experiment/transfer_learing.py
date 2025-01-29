import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import random
from sklearn.model_selection import train_test_split

# funzione per impostare i seed per la riproducibilità
def set_seed(seed):
    """
    Imposta tutti i seed necessari per PyTorch, NumPy e Python random
    """
    # Python random
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Assicura operazioni CUDA deterministiche
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#funzione per creare i dataloader
def create_dataloader(X, y, batch_size=64):
    dataset = TensorDataset(torch.tensor(X.values, dtype=torch.float32),
                            torch.tensor(y.values, dtype=torch.float32).view(-1, 1))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(Autoencoder, self).__init__()
        # Encoder initial layer
        self.encoder_initial = nn.Sequential(
            nn.Linear(input_dim, 1322),
            nn.ReLU(),
            nn.BatchNorm1d(1322),
            nn.Dropout(0.4),
            #nn.Linear(2048, 1322),
            #nn.ReLU(),
            #nn.BatchNorm1d(1322),
            #nn.Dropout(0.4),
        )
        # Pre-trained encoder layers
        self.encoder_pretrained = nn.Sequential(
            nn.Linear(1322, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, bottleneck_dim)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 1322),
            nn.ReLU(),
            nn.BatchNorm1d(1322),
            nn.Dropout(0.4),
            #nn.Linear(1322, 2048),
            #nn.ReLU(),
            #nn.BatchNorm1d(2048),
            #nn.Dropout(0.4),
            nn.Linear(1322, input_dim),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim),
            nn.Dropout(0.4)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_initial(x)
        latent = self.encoder_pretrained(x)
        reconstruction = self.decoder(latent)
        classification = self.classifier(latent)
        return latent, reconstruction, classification

    def encode(self, x):
        x = self.encoder_initial(x)
        latent = self.encoder_pretrained(x)
        return latent

#funzione per caricare il modello preaddestrato
def load_pretrained_encoder(model, pretrained_path):
    # Load the pre-trained model's state dictionary
    pretrained_state = torch.load(pretrained_path)['model_state_dict']
    # Extract the encoder's state dictionary from the pre-trained model
    pretrained_encoder_state = {k.replace('encoder.', ''): v for k, v in pretrained_state.items() if k.startswith('encoder.')}
    # Load the pre-trained encoder's weights into the new model's encoder_pretrained
    model.encoder_pretrained.load_state_dict(pretrained_encoder_state, strict=False)
    print("Pre-trained encoder weights loaded.")
    return model


#funzione per il training dell'autoencoder
def train_autoencoder(model, train_loader, val_loader, optimizer, criterion_reconstruction, criterion_classification, alpha, device, patience=10):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    epoch = 100
    for epoch in range(epoch):
        #impostiamo il modello in modalità training
        model.train()
        #azzeriamo la loss
        total_train_loss = 0
        #iteriamo sul train_dataloader
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            #otteniamo i valori di latent, reconstruction e classification
            latent, reconstruction, classification = model(X_batch)
            loss_reconstruction = criterion_reconstruction(reconstruction, X_batch)
            loss_classification = criterion_classification(classification, y_batch)
            #calcoliamo la loss totale
            loss = alpha * loss_reconstruction + (1 - alpha) * loss_classification
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        #fase di validazione
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                _, reconstruction, classification = model(X_batch)
                loss_reconstruction = criterion_reconstruction(reconstruction, X_batch)
                loss_classification = criterion_classification(classification, y_batch)
                loss = alpha * loss_reconstruction + (1 - alpha) * loss_classification
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break




def sae_cross_validation(X, y, encoding_shape, alpha, num_folds=5, random_state=42,pretrained_path=None):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Convert to tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)  # reshape for classification

    # Set parameters
    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")
    bottleneck_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    
    # impostiamo il seed
    set_seed(random_state)
    
    # metriche da calcolare
    mean_results = {'macro avg precision': [], 'macro avg recall': [], 'macro avg f1-score': [],
                   'weighted avg precision': [], 'weighted avg recall': [], 'weighted avg f1-score': []}
    
    best_f1_score = -1
    best_model = None
    best_fold = -1

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        set_seed(random_state + fold)
        print(f"Fold {fold}:")

        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create data loaders
        train_loader = create_dataloader(X_train, y_train)
        val_loader = create_dataloader(X_val, y_val)

        # Initialize autoencoder
        model = Autoencoder(input_dim=X.shape[1], bottleneck_dim=encoding_shape).to(device)
        if pretrained_path:
            model = load_pretrained_encoder(model, pretrained_path)
            print("Pre-trained encoder weights loaded.")
            # Freeze pre-trained encoder layers except the last one
            for param in model.encoder_pretrained.parameters():
                param.requires_grad = False
            # Unfreeze the last layer of encoder_pretrained for fine-tuning
            for param in model.encoder_pretrained[-1].parameters():
                param.requires_grad = True

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion_reconstruction = nn.MSELoss()
        criterion_classification = nn.BCELoss()

        # Train autoencoder with early stopping
        train_autoencoder(
            model, train_loader, val_loader, optimizer, criterion_reconstruction, criterion_classification, alpha, device
        )

        model.eval()
        with torch.no_grad():
            # Extract latent features
            latent_train = model.encode(torch.tensor(X_train.values, dtype=torch.float32).to(device)).cpu().numpy()
            latent_val = model.encode(torch.tensor(X_val.values, dtype=torch.float32).to(device)).cpu().numpy()

        # Train and evaluate Random Forest on latent features
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        rf.fit(latent_train, y_train)
        y_pred = rf.predict(latent_val)
        report = classification_report(y_val, y_pred, output_dict=True)
        print("classification report:")
        print(report)

        # Collect metrics
        current_f1_score = report['macro avg']['f1-score']
        for metric in mean_results.keys():
            mean_results[metric].append(report['macro avg'][metric.split()[-1]])

        # Save best model based on F1 score
        if current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
            best_model = model
            best_fold = fold

    # Aggregate results
    for metric in mean_results.keys():
        mean_results[metric] = np.mean(mean_results[metric])
    
    print("\nCross-validation results:")
    for metric, value in mean_results.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nBest model was from fold {best_fold} with F1-score: {best_f1_score:.4f}")

    # Save only the autoencoder state dict
    save_path = r"C:\Users\Utente\Desktop\SAE-microbiome\francesco_Manco\models"
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'input_dim': input_dim,
        'bottleneck_dim': encoding_shape,
        'fold': best_fold,
        'f1_score': best_f1_score
    }, f"{save_path}/best_autoencoder_clr_Shotgun_FINE_TUNED_0_7_padding.pt")

    # Save the cross-validation results
    results_df = pd.DataFrame({
        'Metric': ['Best fold', 'Best F1-score'] + list(mean_results.keys()),
        'Value': [best_fold, best_f1_score] + list(mean_results.values())
    })
    results_df.to_csv(r"C:\Users\Utente\Desktop\SAE-microbiome\results\SAE_clr_SHOTGUN_FINE_TUNED_padding.csv", index=False)

    return mean_results

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv(r'C:\Users\Utente\Desktop\SAE-microbiome\francesco_Manco\data\df_MV-ASD_clr_Shotugun.csv')
    X = df.drop(['target'], axis=1)
    y = df['target']

    SEED = 42
    set_seed(SEED)
    encoding_shape = 64
    alpha = 0.6
    pretrained_path = r"C:\Users\Utente\Desktop\SAE-microbiome\francesco_Manco\models\best_autoencoder_clr_0_7_padding.pt"
    sae_cross_validation(X, y, encoding_shape, alpha,random_state=SEED,pretrained_path=pretrained_path)
