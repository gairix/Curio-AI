from fastapi import APIRouter

from backend.app.api.endpoints import status, upload, youtube, chat, actions, system

api_router = APIRouter()

# Register sub-routers
api_router.include_router(status.router)
api_router.include_router(upload.router)
api_router.include_router(youtube.router)
api_router.include_router(chat.router)
api_router.include_router(actions.router)
api_router.include_router(system.router)
