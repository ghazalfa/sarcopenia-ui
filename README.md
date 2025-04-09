# AI Powered Sarcopenia Diagnostic Model

##  Getting Started in VS Code

### 1. Clone the repository
```bash
git clone https://github.com/ghazalfa/sarcopenia-ui.git
cd sarcopenia-ui
```

### 2. Open in VS Code
```bash
code .
```

### 3. (Optional) Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```
### 5. Download the Weights from this google drive
https://drive.google.com/drive/folders/1K1V6QDGjldSrVuyLIQlDibykEr7Lqdd5?usp=drive_link

### 6. Updating the file paths to the weights
Update the 3 file paths at the top of the main.py file so that the filepaths correspond to the correct weights.

### 7. Run the app
```bash
streamlit run main.py --server.maxUploadSize 300
```

## Using the UI
Now that the UI is up and running on your computer, if the CT image is upside down, click the checkmark before adding the image into the website.

