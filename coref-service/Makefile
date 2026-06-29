COMPOSE ?= docker compose

.DEFAULT_GOAL := help
.PHONY: help build up cascade down restart logs health resolve test download-s2e clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## Build the Docker image
	$(COMPOSE) build

up: ## Start the service (LingMess, default) on :5000
	$(COMPOSE) up -d --build
	@echo "→ http://localhost:5000  (docs at /docs)"

cascade: download-s2e ## Start the LingMess → s2e cascade on :5000
	COREF_MODEL=cascade S2E_MODEL_PATH=/models/s2e $(COMPOSE) up -d --build
	@echo "→ cascade up. Verify with: make health  (expect s2e_active=true)"

down: ## Stop and remove the container
	$(COMPOSE) down

restart: ## Restart the service
	$(COMPOSE) restart

logs: ## Follow service logs
	$(COMPOSE) logs -f

health: ## Print /health as JSON
	@curl -s localhost:5000/health | python3 -m json.tool

resolve: ## Demo: resolve a sample sentence (override with TEXT=...)
	@curl -s -X POST localhost:5000/resolve -H 'Content-Type: application/json' \
	  -d '{"text":"$(or $(TEXT),Rapamycin reduced mTOR. It was effective. The drug helped.)"}' \
	  | python3 -m json.tool

download-s2e: ## Download the s2e checkpoint (~1.6 GB) if missing
	@bash scripts/download_s2e.sh

test: ## Run unit tests (needs: pip install -r requirements-dev.txt)
	pytest -q

clean: ## Remove Python caches
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ cleaned"
