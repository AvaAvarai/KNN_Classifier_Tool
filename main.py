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

class KNNApp:
    def __init__(self, master):
        self.master = master
        self.master.title("KNN Classifier Tool")
        self.master.geometry("400x550")
        self.center_window()

        self.file_path = tk.StringVar()
        self.test_size = tk.DoubleVar(value=0.2)
        self.distance_metric = tk.StringVar(value="euclidean")
        self.k_value = tk.IntVar(value=5)
        self.preprocessing = tk.StringVar(value="none")
        self.p_value = tk.DoubleVar(value=2)  # Added p_value for Minkowski distance

        self.create_widgets()

    def center_window(self):
        self.master.update_idletasks()
        width = self.master.winfo_width()
        height = self.master.winfo_height()
        x = (self.master.winfo_screenwidth() // 2) - (width // 2)
        y = (self.master.winfo_screenheight() // 2) - (height // 2)
        self.master.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def create_widgets(self):
        ttk.Button(self.master, text="Select CSV File", command=self.load_data).pack(pady=10)
        
        ttk.Label(self.master, text="Test Size (0.0 - 1.0):").pack()
        ttk.Scale(self.master, from_=0.1, to=0.5, orient="horizontal", 
                  variable=self.test_size, command=self.update_test_size).pack()
        self.test_size_label = ttk.Label(self.master, text="0.2")
        self.test_size_label.pack()

        ttk.Label(self.master, text="Distance Metric:").pack()
        self.metric_combobox = ttk.Combobox(self.master, textvariable=self.distance_metric, 
                     values=["euclidean", "manhattan", "minkowski"])
        self.metric_combobox.pack()
        self.metric_combobox.bind("<<ComboboxSelected>>", self.on_metric_change)

        ttk.Label(self.master, text="K Value:").pack()
        ttk.Spinbox(self.master, from_=1, to=20, textvariable=self.k_value).pack()

        self.p_value_frame = ttk.Frame(self.master)
        self.p_value_frame.pack(pady=5)
        ttk.Label(self.p_value_frame, text="P Value (for Minkowski):").pack(side=tk.LEFT)
        self.p_value_spinbox = ttk.Spinbox(self.p_value_frame, from_=1, to=10, increment=0.1, textvariable=self.p_value, state="disabled")
        self.p_value_spinbox.pack(side=tk.LEFT)
        
        ttk.Label(self.master, text="Preprocessing:").pack()
        ttk.Combobox(self.master, textvariable=self.preprocessing, 
                     values=["none", "standard_scaler", "minmax_scaler"]).pack()

        ttk.Button(self.master, text="Run KNN", command=self.run_knn).pack(pady=10)

    def update_test_size(self, event):
        self.test_size_label.config(text=f"{self.test_size.get():.2f}")

    def on_metric_change(self, event):
        if self.distance_metric.get() == "minkowski":
            self.p_value_spinbox.config(state="normal")
        else:
            self.p_value_spinbox.config(state="disabled")

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.file_path.set(file_path)
            print(f"File selected: {file_path}")

    def run_knn(self):
        if not self.file_path.get():
            print("Please select a CSV file first.")
            return

        df = pd.read_csv(self.file_path.get())
        class_column = next(col for col in df.columns if col.lower() == 'class')
        X = df.drop(columns=[class_column])
        y = df[class_column]

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
        
        # Configure KNN classifier
        knn_params = {
            'n_neighbors': self.k_value.get(),
            'metric': self.distance_metric.get()
        }
        if self.distance_metric.get() == "minkowski":
            knn_params['p'] = self.p_value.get()
        
        knn = KNeighborsClassifier(**knn_params)
        knn.fit(X_train_processed, y_train)
        y_pred = knn.predict(X_test_processed)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Preprocessing: {self.preprocessing.get()}")
        print(f"Distance Metric: {self.distance_metric.get()}")
        if self.distance_metric.get() == "minkowski":
            print(f"P Value: {self.p_value.get()}")
        print(f"K Value: {self.k_value.get()}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        title = f'Confusion Matrix\n(Preprocessing: {self.preprocessing.get()},\nMetric: {self.distance_metric.get()}'
        if self.distance_metric.get() == "minkowski":
            title += f', P: {self.p_value.get()}'
        title += f', K: {self.k_value.get()})'
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()

def main():
    root = tk.Tk()
    app = KNNApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
