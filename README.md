# README
## Instructions for use
- Install dependencies with poetry
- Download the `content` folder from Google Drive and put it into the root of this directory (i.e. you should have `./content/tables`, `./content/unstructured`)
- Create a `.env` file which contains an entry for `OPENAI_API_KEY` (and also `LANGCHAIN_API_KEY` if you want to do tracing, alongside `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_ENDPOINT=https://api.smith.langchain.com` \[unsure if the last one is actually needed])
- Run `poetry run python runner.py`
- Open the URL that Streamlit tells you to, probably `http://localhost:8501`

## Notes
- The citations provided are references to the main Airtable with the format `airtable.csv_COLUMN_ROW_CHUNK`.
  - Rows start from zero and ignore the notes (and headers)
  - Chunks don't mean anything at the moment
- There's a problem with using `RunnableWithChatHistory` when using Anthropic models (see [issue](https://github.com/langchain-ai/langchain/issues/26563)), so it's currently implemented in a more manual way.

## Setting up heroku
- Do `git push heroku main` then navigate to the URL to check project

### Database (not actually using the cloud one)
- To reset do `heroku pg:reset DATABASE_URL`
- After that, make sure to run
```
heroku pg:psql
CREATE EXTENSION IF NOT EXISTS vector;
```
- At the moment we're just making a new local Chroma vectorstore every time because I couldn't get the PGVector one to work