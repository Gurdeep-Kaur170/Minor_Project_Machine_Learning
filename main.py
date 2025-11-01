import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
class EmployeeAttritionApp:
    def __init__(self):  # ✅ fixed constructor
        self.root = tk.Tk()
        self.root.title("🔍 Employee Attrition Predictor")
        self.root.geometry("800x900")
        self.main_bg = "#f0f4f7"
        self.frame_bg = "#dfe6e9"
        self.label_color = "#2d3436"
        self.btn_color = "#0984e3"

        # Initialize variables
        self.df = None
        self.label_encoders = {}
        self.scaler = None
        self.model = None
        self.feature_names = []
        self.input_widgets = {}

        self.acc = 0
        self.prec = 0
        self.rec = 0
        self.cm = None

        self.setup_gui()

    def setup_gui(self):
        self.root.configure(bg=self.main_bg)

        # Header
        self.header = tk.Label(
            self.root,
            text="Employee Attrition Predictor",
            bg=self.main_bg,
            fg="#2c3e50",
            font=("Segoe UI", 16, "bold"),
        )
        self.header.pack(pady=10)

        # File loading section
        self.file_frame = tk.Frame(self.root, bg=self.main_bg)
        self.file_frame.pack(pady=10)

        self.load_btn = tk.Button(
            self.file_frame,
            text="📂 Load Dataset",
            command=self.load_dataset,
            bg=self.btn_color,
            fg="white",
            font=("Segoe UI", 10, "bold"),
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.preprocess_btn = tk.Button(
            self.file_frame,
            text="⚙ Preprocess Data",
            command=self.preprocess_data,
            bg="#6c5ce7",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            state=tk.DISABLED,
        )
        self.preprocess_btn.pack(side=tk.LEFT, padx=5)

        # Input features frame with scrollbar
        self.canvas = tk.Canvas(self.root, bg=self.main_bg)
        self.scroll_y = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = tk.Frame(self.canvas, bg=self.frame_bg)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        # Result and metrics
        self.result_label = tk.Label(
            self.root,
            text="",
            bg=self.main_bg,
            font=("Segoe UI", 12, "bold"),
            wraplength=600,
        )
        self.result_label.pack(pady=15)

        # Action buttons
        self.btn_frame = tk.Frame(self.root, bg=self.main_bg)
        self.btn_frame.pack(pady=10)

        self.predict_btn = tk.Button(
            self.btn_frame,
            text="🔮 Predict",
            command=self.predict_attrition,
            bg=self.btn_color,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            state=tk.DISABLED,
        )
        self.predict_btn.grid(row=0, column=0, padx=10)

        self.clear_btn = tk.Button(
            self.btn_frame,
            text="🧹 Clear",
            command=self.clear_inputs,
            bg="#636e72",
            fg="white",
            font=("Segoe UI", 10, "bold"),
        )
        self.clear_btn.grid(row=0, column=1, padx=10)

        self.matrix_btn = tk.Button(
            self.btn_frame,
            text="📊 Show Metrics",
            command=self.show_metrics,
            bg="#6c5ce7",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            state=tk.DISABLED,
        )
        self.matrix_btn.grid(row=0, column=2, padx=10)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            self.df = pd.read_csv(file_path)
            self.preprocess_btn.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Dataset loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")

    def preprocess_data(self):
        try:
            # Drop irrelevant columns if they exist
            cols_to_drop = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
            self.df.drop([c for c in cols_to_drop if c in self.df.columns], axis=1, inplace=True)

            # Encode categorical features
            self.label_encoders = {}
            cat_cols = self.df.select_dtypes(include="object").columns
            for col in cat_cols:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le

            # Features and target
            X = self.df.drop("Attrition", axis=1)
            y = self.df["Attrition"]

            # Scale numerical features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train Random Forest
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

            # Metrics
            y_pred = self.model.predict(X_test)
            self.acc = accuracy_score(y_test, y_pred)
            self.prec = precision_score(y_test, y_pred)
            self.rec = recall_score(y_test, y_pred)
            self.cm = confusion_matrix(y_test, y_pred)

            # Create input fields
            self.create_input_fields()
            self.predict_btn.config(state=tk.NORMAL)
            self.matrix_btn.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Preprocessing and model training completed!")

        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed:\n{str(e)}")

    def create_input_fields(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        tk.Label(
            self.scroll_frame,
            text="Enter Employee Features:",
            bg=self.frame_bg,
            fg=self.label_color,
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, columnspan=2, pady=10)

        self.feature_names = self.df.drop("Attrition", axis=1).columns.tolist()
        self.input_widgets = {}

        for i, col in enumerate(self.feature_names):
            tk.Label(
                self.scroll_frame,
                text=f"{col}:",
                bg=self.frame_bg,
                fg=self.label_color,
                font=("Segoe UI", 10),
            ).grid(row=i + 1, column=0, sticky="e", pady=5, padx=5)

            if col in self.label_encoders:
                widget = ttk.Combobox(
                    self.scroll_frame,
                    values=self.label_encoders[col].classes_.tolist(),
                    state="readonly",
                    width=25,
                )
                widget.grid(row=i + 1, column=1, padx=5, pady=5)
            else:
                widget = tk.Entry(self.scroll_frame, width=27, font=("Segoe UI", 10))
                widget.grid(row=i + 1, column=1, padx=5, pady=5)

            self.input_widgets[col] = widget

    def predict_attrition(self):
        try:
            input_data = []
            for col in self.feature_names:
                widget = self.input_widgets[col]
                value = widget.get()
                if not value:
                    raise ValueError(f"Missing value for {col}")
                if col in self.label_encoders:
                    encoded_value = self.label_encoders[col].transform([value])[0]
                    input_data.append(encoded_value)
                else:
                    input_data.append(float(value))

            # Scale and predict
            input_scaled = self.scaler.transform([input_data])
            prediction = self.model.predict(input_scaled)[0]

            # Display result
            if prediction == 1:
                self.result_label.config(
                    text="⚠ High Attrition Risk: Employee may leave",
                    fg="white",
                    bg="#e74c3c",
                )
            else:
                self.result_label.config(
                    text="✅ Low Risk: Employee likely to stay",
                    fg="white",
                    bg="#2ecc71",
                )

        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid input:\n{str(e)}")

    def show_metrics(self):
        if self.cm is None:
            messagebox.showwarning("Warning", "No metrics available yet!")
            return

        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        sns.heatmap(
            self.cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Stay", "Leave"],
            yticklabels=["Stay", "Leave"],
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.subplot(2, 1, 2)
        metrics = [self.acc, self.prec, self.rec]
        labels = ["Accuracy", "Precision", "Recall"]
        plt.bar(labels, metrics, color=[self.btn_color, "#6c5ce7", "#00b894"])
        plt.ylim(0, 1)
        plt.title("Model Performance Metrics")

        plt.tight_layout()
        plt.show()

    def clear_inputs(self):
        for col, widget in self.input_widgets.items():
            if isinstance(widget, ttk.Combobox):
                widget.set("")
            else:
                widget.delete(0, tk.END)
        self.result_label.config(text="", bg=self.main_bg)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":  # ✅ fixed entry point
    app = EmployeeAttritionApp()
    app.run()
