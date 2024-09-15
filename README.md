# README
## Instructions for use
- Install dependencies with `pip install -r requirements.txt`. I'm using Python 3.10
- Download the `content` folder from Google Drive and put it into the root of this directory (i.e. you should have `./content/tables`, `./content/unstructured`)
- Create a `.env` file which contains an entry for `OPENAI_API_KEY` (and also `LANGCHAIN_API_KEY` if you want to do tracing, alongside `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_ENDPOINT=https://api.smith.langchain.com` \[unsure if the last one is actually needed])
- Run `app.py`
- Open `index.html` in your browser