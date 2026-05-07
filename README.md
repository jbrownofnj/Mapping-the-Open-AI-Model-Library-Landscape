# How to download and run the code:

## 1. Clone the Repo

```bash
git clone https://github.com/jbrownofnj/Mapping-the-Open-AI-Model-Library-Landscape.git
cd Mapping-the-Open-AI-Model-Library-Landscape
```
## 2. Make a virtual invironment
### Windows
```bash
python -m venv venv
venv\Scripts\activate
```
### Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
## 3. Install packages
```bash 
pip install -r requirements.txt
```

## 4. Add the HF API Key from CodeBench
Create a file in the root directory called .env
insdie the file add the following:
```bash
HF_TOKEN=your_hugging_face_token_here
```

## 5. Run Program
```bash
python src/main.py
```