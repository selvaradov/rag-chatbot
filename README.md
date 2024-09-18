# README
## Instructions for use locally
- Install dependencies with poetry
- Create a `.env` file which contains an entry for `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (and also `LANGCHAIN_API_KEY` if you want to do tracing, alongside `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_ENDPOINT=https://api.smith.langchain.com` \[unsure if the last one is actually needed])
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
- It's a bit confusing how the database works.
- There will be two tables created: 
  - `langchain_pg_collection` which contains all the different "collections" you've created -- usually you'd only have one, I think. The default name for the collection is just `langchain`
  - `langchain_pg_embedding` which contains all the embeddings for every document across all collections, along with other information (e.g., which collection does it belong to, etc.)
- To reset do `heroku pg:reset DATABASE_URL`
- After that, make sure to run
```
heroku pg:psql
CREATE EXTENSION IF NOT EXISTS vector;
```
- At the moment we're just making a new local Chroma vectorstore every time because I couldn't get the PGVector one to work
- Docker working now for the Postgres:
  - To [set up](https://python.langchain.com/docs/integrations/vectorstores/pgvector/): `docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16`
  - To reset: `docker exec -it pgvector-container psql -U langchain -d langchain -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public; GRANT ALL ON SCHEMA public TO langchain; GRANT ALL ON SCHEMA public TO public;"`
  - To get shell `docker exec -it pgvector-container psql -U langchain -d langchain`