import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from Supervised_Autoencoder_SAE import sae, load_dataset
from keras import models
import seaborn as sns

class Experimentor:
    def __init__(self, X, y, name):
        self.X = X
        self.y = y
        self.name = name
        self.result_path = os.path.join('../results', name)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

    def classify(self):
        """
        Performs stratified k-fold cross-validation classification using Random Forest.
        This method implements 5-fold cross-validation with a Random Forest classifier. It:
        1. Splits data into train/test sets using stratified k-fold
        2. Trains Random Forest model on each fold
        3. Makes predictions and calculates performance metrics
        4. Averages metrics across all folds
        5. Saves results to CSV file
        The following metrics are calculated:
        - Macro average precision
        - Macro average recall 
        - Macro average F1-score
        - Weighted average precision
        - Weighted average recall
        - Weighted average F1-score
        Returns:
            None. Results are saved to CSV file and printed to console.
        Example:
            classifier = MyClassifier()
            classifier.classify()
        """

        #implementa la CV e decide le metriche da calcolare
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        mean_results = {'macro avg precision': [], 'macro avg recall': [], 'macro avg f1-score': [],
                        'weighted avg precision': [], 'weighted avg recall': [], 'weighted avg f1-score': []}

        #cicla su ogni fold creata dalla CV
        for fold, (train_index, test_index) in enumerate(skf.split(self.X, self.y), 1):
            print(f"Fold {fold}:")
            X_train, X_val = self.X[train_index], self.X[test_index]
            y_train, y_val = self.y[train_index], self.y[test_index]

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
        """
        Performs classification using PCA dimensionality reduction and Random Forest classifier with 5-fold cross validation.
        This method applies Principal Component Analysis (PCA) for dimensionality reduction followed by
        Random Forest classification. Performance metrics are calculated using 5-fold cross validation.
        Args:
            n_components (int): Number of components to keep after PCA dimensionality reduction.
        Returns:
            None. Results are saved to CSV file and printed to console. The following metrics are computed:
                - Macro average precision
                - Macro average recall  
                - Macro average F1-score
                - Weighted average precision
                - Weighted average recall
                - Weighted average F1-score
        The results are averaged across all 5 folds and saved in 'pca_rf_results.csv' in the result_path directory.
        Example:
            classifier.classify_with_pca(n_components=10)
        """

        #crea la CV e inizializza un dizionario per le metriche
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        mean_results = {'macro avg precision': [], 'macro avg recall': [], 'macro avg f1-score': [],
                        'weighted avg precision': [], 'weighted avg recall': [], 'weighted avg f1-score': []}

        #cicla su ogni fold della CV
        for fold, (train_index, test_index) in enumerate(skf.split(self.X, self.y), 1):
            print(f"Fold {fold}:")
            X_train, X_val = self.X[train_index], self.X[test_index]
            y_train, y_val = self.y[train_index], self.y[test_index]

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

    def sae_experiment(self, encoding_shape0, encoding_shape1, alpha, latent_act):
        """
        Performs Supervised Autoencoder (SAE) experiments with 5-fold cross-validation.
        The method trains a supervised autoencoder on the data, extracts latent representations,
        and uses them to train a Random Forest classifier. Performance metrics are computed
        and averaged across folds.
        Parameters
        ----------
        encoding_shape0 : int
            Number of neurons in the first encoding layer
        encoding_shape1 : int 
            Number of neurons in the second encoding layer (bottleneck layer)
        alpha : float
            Weight parameter balancing reconstruction and classification losses in SAE
        latent_act : str
            Activation function to use in the latent layer
        Returns
        -------
        None
            Results are saved to files:
            - Classification metrics saved to 'sae_results.csv'
            - Loss curves for each fold saved as 'AE_[fold_number].png'
        Notes
        -----
        The method:
        1. Performs 5-fold stratified cross-validation
        2. Trains SAE on each fold
        3. Extracts latent representations
        4. Trains Random Forest on latent space
        5. Computes and saves performance metrics
        6. Plots and saves training/validation loss curves
        """

        #crea la CV e inizializza un dizionario per le metriche
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        mean_results = {'macro avg precision': [], 'macro avg recall': [], 'macro avg f1-score': [],
                        'weighted avg precision': [], 'weighted avg recall': [], 'weighted avg f1-score': []}

        #cicla su ogni fold della CV
        for fold, (train_index, test_index) in enumerate(skf.split(self.X, self.y), 1):
            print(f"Fold {fold}:")
            X_train, X_val = self.X[train_index], self.X[test_index]
            y_train, y_val = self.y[train_index], self.y[test_index]

            #crea e addestra l'autoencoder
            autoencoder = sae(input_shape=X_train.shape[1], encoder_shape0=encoding_shape0,
                              encoder_shape1=encoding_shape1, alpha=alpha)
            history = autoencoder.fit(X_train, (X_train, y_train),
                                      validation_data=(X_val, (X_val, y_val)), epochs=200, batch_size=100, verbose=0)

            #estrae le rappresentazioni latenti e addestra il random forest
            encoder = models.Model(inputs=autoencoder.input,
                                         outputs=autoencoder.get_layer('layer_reduced').output)
            latent_train = encoder.predict(X_train)
            latent_val = encoder.predict(X_val)

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(latent_train, y_train)
            #predice e calcola le metriche di performance del random forest addestrato sulle rappresentazioni latenti
            y_pred = rf.predict(latent_val)

            report = classification_report(y_val, y_pred, output_dict=True)
            for metric in mean_results.keys():
                mean_results[metric].append(report['macro avg'][metric.split()[-1]])

            # Plotting SAE training and validation loss
            plt.figure(figsize=(10, 8))
            sns.set(font_scale=2)
            sns.set_style("white")
            plt.plot(history.history["loss"], label="Training Loss", linewidth=3.0)
            plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=3.0)
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.ylim(0, 3.5)
            plt.savefig(os.path.join(self.result_path, f'AE_{fold}.png'))

        for metric in mean_results.keys():
            mean_results[metric] = np.mean(mean_results[metric])

        results_df = pd.DataFrame.from_dict(mean_results, orient='index', columns=['Mean Value (5 fold)']).T
        results_df.to_csv(os.path.join(self.result_path, 'sae_results.csv'), index=False)
        print(results_df)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument()
    parser.add_argument("--expname", type=str, required=True, help="Experiment name",
                        choices=["Baseline", "PCA_RF", "SAE"])
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--encoding_shape0", type=int, default=None, help="Dimension for the first layer of encoder")
    parser.add_argument("--encoding_shape1", type=int, default=None, help="Dimension for the second layer of encoder")
    parser.add_argument("--alpha", type=float, default=None, help="Alpha value for loss weighting")
    parser.add_argument("--latent_act", action='store_true', help="Activation function in latent layer")
    parser.add_argument("--n_components", type=int, default=None, help="Number of components for PCA")
    args = parser.parse_args()

    # Load data
    df = load_dataset(args.dataset)
    X = df.drop('target', axis=1).values
    y = df['target'].values

    exp = Experimentor(X=X, y=y, name=args.expname)

    if args.expname == "Baseline":
        exp.classify()

    elif args.expname == "PCA_RF":
        if args.n_components is None:
            raise ValueError("Number of components for PCA must be provided for PCA_RF experiment")
        exp.classify_with_pca(n_components=args.n_components)

    elif args.expname == "SAE":
        if args.encoding_shape0 is None or args.encoding_shape1 is None or args.alpha is None:
            raise ValueError("Autoencoder dimensions and alpha must be provided for SAE experiment")
        exp.sae_experiment(encoding_shape0=args.encoding_shape0, encoding_shape1=args.encoding_shape1, alpha=args.alpha,
                           latent_act=args.latent_act)

    print('End of main')