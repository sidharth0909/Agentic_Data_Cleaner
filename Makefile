.PHONY: install install-backend install-frontend dev dev-backend dev-frontend test

# Install everything
install: install-backend install-frontend

install-backend:
	cd backend && python -m venv .venv && .venv/bin/pip install -r requirements.txt

install-frontend:
	cd frontend && npm install

# Run both servers concurrently (requires 'concurrently' or two terminals)
dev:
	@echo "Start two terminals and run:"
	@echo "  Terminal 1: make dev-backend"
	@echo "  Terminal 2: make dev-frontend"

dev-backend:
	cd backend && .venv/bin/uvicorn main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

# Tests
test:
	cd backend && .venv/bin/pytest tests/ -v

# Copy .env examples
setup-env:
	cp backend/.env.example backend/.env
	cp frontend/.env.example frontend/.env.local
	@echo "Edit backend/.env and add your ANTHROPIC_API_KEY"
