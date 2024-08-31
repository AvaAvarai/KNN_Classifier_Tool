# Import necessary libraries
import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pandas.plotting import parallel_coordinates
from functools import partial

class KNNApp:
    def __init__(self, master):
        """Initialize the KNN Classifier Tool application."""
        self.master = master
        self.master.title("KNN Classifier Tool")
        self.master.geometry("400x550")
        self.center_window()

        # Initialize variables
        self.file_path = tk.StringVar()
        self.test_size = tk.DoubleVar(value=0.2)
        self.distance_metric = tk.StringVar(value="euclidean")
        self.k_value = tk.IntVar(value=5)
        self.preprocessing = tk.StringVar(value="none")
        self.p_value = tk.DoubleVar(value=2)

        self.create_widgets()

    def center_window(self):
        """Center the application window on the screen."""
        self.master.update_idletasks()
        width = self.master.winfo_width()
        height = self.master.winfo_height()
        x = (self.master.winfo_screenwidth() // 2) - (width // 2)
        y = (self.master.winfo_screenheight() // 2) - (height // 2)
        self.master.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def create_widgets(self):
        """Create and arrange the GUI widgets."""
        # File selection button
        ttk.Button(self.master, text="Select CSV File", command=self.load_data).pack(pady=10)
        
        # Test size slider
        ttk.Label(self.master, text="Test Size (0.0 - 1.0):").pack()
        ttk.Scale(self.master, from_=0.1, to=0.5, orient="horizontal", 
                  variable=self.test_size, command=self.update_test_size).pack()
        self.test_size_label = ttk.Label(self.master, text="0.2")
        self.test_size_label.pack()

        # Distance metric selection
        ttk.Label(self.master, text="Distance Metric:").pack()
        self.metric_combobox = ttk.Combobox(self.master, textvariable=self.distance_metric, 
                     values=["euclidean", "manhattan", "minkowski", "diem"])
        self.metric_combobox.pack()
        self.metric_combobox.bind("<<ComboboxSelected>>", self.on_metric_change)

        # K value selection
        ttk.Label(self.master, text="K Value:").pack()
        ttk.Spinbox(self.master, from_=1, to=20, textvariable=self.k_value).pack()

        # P value selection (for Minkowski distance)
        self.p_value_frame = ttk.Frame(self.master)
        self.p_value_frame.pack(pady=5)
        ttk.Label(self.p_value_frame, text="P Value (for Minkowski):").pack(side=tk.LEFT)
        self.p_value_spinbox = ttk.Spinbox(self.p_value_frame, from_=1, to=10, increment=0.1, textvariable=self.p_value, state="disabled")
        self.p_value_spinbox.pack(side=tk.LEFT)
        
        # Preprocessing method selection
        ttk.Label(self.master, text="Preprocessing:").pack()
        ttk.Combobox(self.master, textvariable=self.preprocessing, 
                     values=["none", "standard_scaler", "minmax_scaler"]).pack()

        # Run KNN button
        ttk.Button(self.master, text="Run KNN", command=self.run_knn).pack(pady=10)

    def update_test_size(self, event):
        """Update the test size label when the slider is moved."""
        self.test_size_label.config(text=f"{self.test_size.get():.2f}")

    def on_metric_change(self, event):
        """Enable or disable the P value input based on the selected distance metric."""
        if self.distance_metric.get() == "minkowski":
            self.p_value_spinbox.config(state="normal")
        else:
            self.p_value_spinbox.config(state="disabled")

    def load_data(self):
        """Open a file dialog to select a CSV file and store the file path."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.file_path.set(file_path)
            print(f"File selected: {file_path}")
    
    def diem_metric(self, x, y, v_M, v_m, sigma_n_squared, E_d_n):
        """Calculate the DIEM metric between two points."""
        # Calculate Euclidean distance
        euclidean_dist = np.sqrt(np.sum((x - y) ** 2))
        
        # Calculate the DIEM metric
        diem = ((v_M - v_m) / sigma_n_squared) * (euclidean_dist - E_d_n)
        return diem

    def compute_diem_parameters(self, X):
        """
        Compute the necessary parameters for DIEM:
        v_M, v_m, sigma_n_squared, and E[d(n)]
        based on the entire dataset X.
        """
        n = X.shape[1]
        
        # Calculate v_M and v_m based on the entire dataset
        v_M = np.max(X)
        v_m = np.min(X)
        
        # Estimate sigma(n)^2 and E[d(n)]
        # These calculations might need to be fine-tuned based on your specific data
        E_d_n = np.sqrt(n) * (v_M - v_m) / np.sqrt(6)  # Expected distance, this is an approximation
        sigma_n_squared = n / 6  # Variance of Euclidean distances, this is an approximation
        
        return v_M, v_m, sigma_n_squared, E_d_n

    def run_knn(self):
        """Execute the KNN classification and display results."""
        if not self.file_path.get():
            print("Please select a CSV file first.")
            return

        # Load and prepare data
        df = pd.read_csv(self.file_path.get())
        class_column = next(col for col in df.columns if col.lower() == 'class')
        X = df.drop(columns=[class_column])
        y = df[class_column]

        # Split data into training and testing sets
        test_size = self.test_size.get()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Apply preprocessing
        if self.preprocessing.get() == "standard_scaler":
            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train)
            X_test_processed = scaler.transform(X_test)
        elif self.preprocessing.get() == "minmax_scaler":
            scaler = MinMaxScaler()
            X_train_processed = scaler.fit_transform(X_train)
            X_test_processed = scaler.transform(X_test)
        else:  # No preprocessing
            X_train_processed = X_train
            X_test_processed = X_test
        
        # Precompute DIEM parameters if using DIEM metric
        if self.distance_metric.get() == "diem":
            v_M, v_m, sigma_n_squared, E_d_n = self.compute_diem_parameters(X_train)
            knn_params = {
                'n_neighbors': self.k_value.get(),
                'metric': partial(self.diem_metric, v_M=v_M, v_m=v_m, sigma_n_squared=sigma_n_squared, E_d_n=E_d_n)
            }
        else:
            knn_params = {
                'n_neighbors': self.k_value.get(),
                'metric': self.distance_metric.get()
            }
            if self.distance_metric.get() == "minkowski":
                knn_params['p'] = self.p_value.get()

        # Train and predict
        knn = KNeighborsClassifier(**knn_params)
        knn.fit(X_train_processed, y_train)
        y_pred = knn.predict(X_test_processed)
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Print results
        print(f"Preprocessing: {self.preprocessing.get()}")
        print(f"Distance Metric: {self.distance_metric.get()}")
        if self.distance_metric.get() == "minkowski":
            print(f"P Value: {self.p_value.get()}")
        print(f"K Value: {self.k_value.get()}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        title = f'Confusion Matrix\n(Preprocessing: {self.preprocessing.get()},\nMetric: {self.distance_metric.get()}'
        if self.distance_metric.get() == "minkowski":
            title += f', P: {self.p_value.get()}'
        title += f', K: {self.k_value.get()})'
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Parallel coordinates plot
        plt.subplot(122)
        X_test_df = pd.DataFrame(X_test_processed, columns=X.columns)
        X_test_df['class'] = y_test
        X_test_df['predicted'] = y_pred

        # Create color maps for actual classes and predictions
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        class_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
        class_color_map = dict(zip(unique_classes, class_colors))
        pred_colors = plt.cm.rainbow(np.linspace(0.1, 0.9, len(unique_classes)))
        pred_color_map = dict(zip(unique_classes, pred_colors))

        # Plot actual classes and predictions
        for cls in unique_classes:
            mask = X_test_df['class'] == cls
            parallel_coordinates(X_test_df[mask].drop('predicted', axis=1), 'class', color=[class_color_map[cls]], alpha=0.5)
            
            mask = X_test_df['predicted'] == cls
            parallel_coordinates(X_test_df[mask].drop('class', axis=1), 'predicted', color=[pred_color_map[cls]], alpha=0.5, linestyle='--')

        plt.title('Parallel Coordinates Plot\nSolid: Actual, Dashed: Predicted')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.show()


def main():
    """Create and run the KNN Classifier Tool application."""
    root = tk.Tk()
    app = KNNApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
