# Docker and VPS deployment

## Prepare
1. Make sure `.env` exists and contains a valid `OPENAI_API_KEY`. `SERPER_API_KEY` is optional if the app's DuckDuckGo fallback is enabled.
2. Never commit `.env`.
3. Create a local data folder and an empty log file before using Compose:
   - Windows CMD: `if not exist data mkdir data`
   - Windows CMD: `if not exist data\answers.txt type nul > data\answers.txt`
   - On Linux VPS: `mkdir -p data && touch data/answers.txt`
4. The application writes interaction logs to `/app/answers.txt`; Compose persists this file at `./data/answers.txt` on the host.

## Build and run locally (Windows CMD)
```cmd
docker build -t multi-agent-customer-support:local .
docker run --rm -p 8501:8501 --env-file .env multi-agent-customer-support:local
```
Open http://localhost:8501.

## Or use Docker Compose
```cmd
if not exist data mkdir data
if not exist data\answers.txt type nul > data\answers.txt
docker compose up --build -d
docker compose ps
docker compose logs -f customer-support
```
Stop with:
```cmd
docker compose down
```

## GitHub
Review `.gitignore` and `.dockerignore`; confirm `.env`, `data/answers.txt`, caches, and private documents are not committed unless you intentionally want public sample docs. Commit the Docker files with your tested app and push to the repository you own.

## VPS deployment
1. SSH to your VPS and clone or pull the repository.
2. Create `.env` on the VPS using a secure editor; never commit it to Git.
3. Create the persistent log file:
   ```bash
   mkdir -p data && touch data/answers.txt
   ```
4. Confirm Docker and the Compose plugin are installed. Ensure port `8501` is available; for public use, prefer a reverse proxy with HTTPS rather than exposing Streamlit directly.
5. Run:
   ```bash
   docker compose up --build -d
   docker compose ps
   docker compose logs --tail=100 customer-support
   ```
6. Validate document upload, sample documents, hybrid retrieval, web fallback, and logging.

## Operational notes
- Uploaded files are held in Streamlit session state in this app. Users may need to upload them again after a session/container restart unless the app is extended to persist uploaded documents and indexes.
- The log is persisted to `./data/answers.txt`. It is excluded from the Docker build context and should remain untracked in Git.
- This Docker setup has been generated from the supplied project files but must still be built and tested on your machine before VPS deployment.
