.PHONY: help dev-frontend dev-frontend-host dev-backend dev dev-host

help:
	@echo "Available commands:"
	@echo "  make dev-frontend      - Starts the frontend development server (Vite)"
	@echo "  make dev-frontend-host - Starts the frontend with network exposure (--host)"
	@echo "  make dev-backend       - Starts the backend development server (Uvicorn with reload)"
	@echo "  make dev               - Starts both frontend and backend development servers"
	@echo "  make dev-host          - Starts both servers with frontend network exposure"

dev-frontend:
	@echo "Starting frontend development server..."
	@cd frontend && npm run dev

dev-frontend-host:
	@echo "Starting frontend development server with network exposure..."
	@cd frontend && npm run dev -- --host

dev-backend:
	@echo "Starting backend development server..."
	@cd backend && langgraph dev

# Run frontend and backend concurrently
dev:
	@echo "Starting both frontend and backend development servers..."
	@make dev-frontend & make dev-backend 

# Run frontend and backend concurrently with network exposure
dev-host:
	@echo "Starting both frontend and backend development servers with network exposure..."
	@make dev-frontend-host & make dev-backend