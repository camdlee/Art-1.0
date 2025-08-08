# Art-Assistant

## Overview
**** Art is an AI powered Streamlit application design to assist architects/interior designs develop their coding skills. 

## Installation
1. Download or clone the repository
2. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    pip install -r requirements-agents.txt
    ```

## Project Structure
- `Home.py` - main application code to start the Streamlit interface
- `requirements.txt` - list of required Python libraries
- `requirements-agents.txt` - list of libraries to deploy agents
- `pages/` - Manages UI for feature functionalities for Art
- `services/` - Manages system prompts and embeddings
- `helpers/` - Contains utility functions for page features
- `data/` - Directory for storing the PDF file and embeddings

## Prerequisites
Ensure that you have Python 3.12 installed

## Dependencies
Ensure that you have the following dependencies/libraries installed:
- streamlit == 1.38.0
- openai==1.45.0
- toml~=0.10.2
- pandas~=2.2.2
- tabulate==0.9.0
- python-dotenv==1.0.1
- streamlit_ace==0.1.1
- PyPDF2 == 1.27.9
- pdf2image == 1.16.0
- pillow==9.5.0
- scikit-learn == 1.3.0
- tiktoken==0.6.0
- numpy==1.26.4
- autogen-agentchat~=0.2
- requests
- streamlit-file-browser
- openai 
- toml 
- pandas 
- tabulate 
- python-dotenv 
- httpx 
- asyncio~=3.4.3 
- pytest 
- fastapi 
- uvicorn 
- gunicorn
- jupyter

## Running Application
To run the application, use the following command:
```bash
streamlit run 🏠_Home.py
```