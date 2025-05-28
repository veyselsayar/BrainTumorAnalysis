BrainTumorAnalysis
BrainTumorAnalysis is an open-source deep learning project for automatic detection and classification of brain tumors in MRI images.
The system provides an easy-to-use REST API for image prediction and real-time summary generation using advanced AI models (HuggingFace, Llama-2/Zephyr).

🚀 Features
Automatic brain tumor detection from MRI images

Tumor type classification: glioma, meningioma, pituitary, no_tumor

REST API for fast predictions (/predict)

Real-time AI summary of recent analysis results (/gemma_summary)

Easy integration with other systems

Result pool for collecting and summarizing past results

HuggingFace LLM (Llama-2, Zephyr) for natural language summary

📦 Directory Structure
graphql
Copy
Edit
project-root/
│
├── app.py                 # Flask API main file
├── ai/
│   ├── add_result.py      # Utility for result pool (JSON)
│   └── gemma_summary.py   # HuggingFace summary module
│
├── models/
│   └── brain_tumor_optimized_cnn_model.keras  # Trained CNN model
│
├── result_havuzu.json     # Collected analysis results (JSON)
├── requirements.txt
└── README.md
⚙️ Installation
Clone the repo:

bash
Copy
Edit
git clone https://github.com/yourusername/BrainTumorAnalysis.git
cd BrainTumorAnalysis
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Set up HuggingFace API key:

Get your token from https://huggingface.co/settings/tokens

Add to a .env file:

ini
Copy
Edit
HF_TOKEN=your_huggingface_token_here
(Optional) Place your trained model in models/ directory.

🖼️ API Usage
Prediction Endpoint
POST /predict

Form-data: "image" (your MRI image file)

Example using curl:

bash
Copy
Edit
curl -X POST -F "image=@/path/to/image.jpg" http://localhost:500/predict
Sample response:

json
Copy
Edit
{
  "predicted_class": "glioma",
  "confidence": 0.94,
  "probabilities": [0.94, 0.02, 0.03, 0.01],
  "timestamp": 1717000000.123
}
Summary Endpoint
GET /gemma_summary

Returns AI-generated summary of the latest predictions.

Sample response:

json
Copy
Edit
{
  "summary": "5 MR görüntüsü incelendi. 2 tanesinde tümör tespit edildi. Glioma ve meningioma saptandı. Hastaların ileri tetkiki önerilir.",
  "result_count": 5
}
📑 Example Usage
Add a new result to pool:

python
Copy
Edit
from ai.add_result import add_result
import time

add_result({
    "predicted_class": "glioma",
    "confidence": 0.93,
    "timestamp": time.time()
})
Generate a summary:

python
Copy
Edit
from ai.gemma_summary import generate_gemma_summary

print(generate_gemma_summary([
    {"predicted_class": "glioma"},
    {"predicted_class": "no_tumor"},
    {"predicted_class": "pituitary"}
]))
🤖 Technologies Used
Python 3.9+

Flask

TensorFlow / Keras

HuggingFace Hub (LLM for summary)

Pillow, NumPy, threading

📜 License
MIT License.
Free for personal and academic use.

✨ Acknowledgements
HuggingFace LLM API

Brain Tumor MRI Dataset (Kaggle)

Open source contributors

👨‍💻 Author
Developed by Veysel SAYAR
Manisa Celal Bayar University – Software Engineering

