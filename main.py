import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class KNNApp:
    def __init__(self, master):
        self.master = master
        self.master.title("KNN Classifier Tool")
        self.master.geometry("400x300")

        self.file_path = tk.StringVar()
        self.test_size = tk.DoubleVar(value=0.2)

        self.create_widgets()

    def create_widgets(self):
        ttk.Button(self.master, text="Select CSV File", command=self.load_data).pack(pady=10)
        
        ttk.Label(self.master, text="Test Size (0.0 - 1.0):").pack()
        ttk.Scale(self.master, from_=0.1, to=0.5, orient="horizontal", 
                  variable=self.test_size, command=self.update_test_size).pack()
        self.test_size_label = ttk.Label(self.master, text="0.2")
        self.test_size_label.pack()

        ttk.Button(self.master, text="Run KNN", command=self.run_knn).pack(pady=10)

    def update_test_size(self, event):
        self.test_size_label.config(text=f"{self.test_size.get():.2f}")

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
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

def main():
    root = tk.Tk()
    app = KNNApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

